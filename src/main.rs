//! 文字単位の最小 GPT。
//!
//! モデルにやらせたいことは「ここまでの文字列を見て、次の 1 文字の確率分布を出す」の 1 つだけ。
//! 文章を生成したいときはその分布から 1 文字引いて末尾に足し、また次を予測させる（`generate`）。
//! 学習では、正解の次の文字に高い確率を出せるよう重みを少しずつ動かす（`main` の学習ループ）。
//!
//! 処理の流れ:
//!   文字列 → Tokenizer で ID 列 → wte + wpe でベクトル列
//!   → Block × N（Attention で過去から情報を集め、MLP で加工し、残差に書き足す）
//!   → ln_f → lm_head で語彙ごとのスコア（logits）→ softmax で次の文字の確率
//!
//! 対話モデルは 2 段階で作る。重みは最後まで 1 組で、汎用の文章力と対話の振る舞いを同じ重みに持たせる:
//!   train    文章コーパスで次の文字の予測を学習する（事前学習）
//!   finetune train の重みから始め、会話データの AI の返事の部分だけを loss に数えて追加学習する（SFT）
//!   chat     finetune の重みで、入力した発言への返事を生成する

use std::collections::HashMap;
use std::io::{BufRead, IsTerminal, Write};
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::{Args, Parser, Subcommand};

mod chat;
mod viz;
use viz::Viz;

use candle_core::backprop::GradStore;
use candle_core::{D, DType, Device, IndexOp, Result, Tensor, Var};
use candle_nn::{AdamW, Embedding, Init, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap, ops};
use rand::distr::{Distribution, weighted::WeightedIndex};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

// 学習済みファイルから Gpt を組み直すのに必要な値。モデルファイルの meta.config に同じ順序で保存する
#[derive(Clone, Copy)]
struct ModelConfig {
	block_size: usize,
	n_layer: usize,
	n_head: usize,
	n_embd: usize,
}

impl ModelConfig {
	fn to_vec(self) -> Vec<u32> {
		[self.block_size, self.n_layer, self.n_head, self.n_embd]
			.map(|v| v as u32)
			.to_vec()
	}

	fn from_slice(v: &[u32]) -> Result<Self> {
		let &[block_size, n_layer, n_head, n_embd] = v else {
			candle_core::bail!("meta.config の要素数が 4 ではない: {}", v.len());
		};
		let cfg = Self {
			block_size: block_size as usize,
			n_layer: n_layer as usize,
			n_head: n_head as usize,
			n_embd: n_embd as usize,
		};
		if cfg.n_head == 0 || cfg.n_embd % cfg.n_head != 0 {
			candle_core::bail!("n_embd ({}) が n_head ({}) で割り切れない", cfg.n_embd, cfg.n_head);
		}
		Ok(cfg)
	}
}

struct TrainConfig {
	model: ModelConfig,
	// 1 回の重み更新に使う系列の本数。勾配はこの本数ぶんの平均になる
	batch_size: usize,
	// 1 回の forward / backward に載せる本数。batch_size を割り切れる値にする
	micro_batch: usize,
	// lr はウォームアップ後の最大値。cosine で max_steps に向けて min_lr まで下げる
	lr: f64,
	min_lr: f64,
	// 学習初期は AdamW の勾配統計が溜まっておらず更新幅が不安定なので、この step 数をかけて 0 から lr まで線形に上げる
	warmup_steps: usize,
	max_steps: usize,
	// 学習時のみ有効にする。評価と生成では 0 にする
	dropout: f32,
	// 毎 step 重みを (1 - lr * weight_decay) 倍に縮め、大きな重みに頼った丸暗記を抑える。
	// 1 次元の bias と LayerNorm には掛けない（ParamGroups 参照）
	weight_decay: f64,
	// 勾配全体の L2 ノルムがこの値を超えたら、向きを保ったままこの長さまで縮める。0 で縮めない（ノルムが inf や NaN なら 0 でも止める）。
	// AdamW は勾配を自身の大きさの移動平均で割るので、1 step の更新幅はもともと lr 程度に収まり、
	// 毎 step 同じ倍率で縮めても更新は変わらない。効くのは突発的に大きい勾配が出た step で、
	// その 1 回が 1 次・2 次モーメントの移動平均を大きく塗り替え、2 次モーメントが長く膨らんだまま
	// 後続の step の更新幅まで縮めてしまうのを防ぐ
	grad_clip: f64,
	eval_interval: usize,
	eval_iters: usize,
}

impl TrainConfig {
	// nanoGPT の文字単位 Shakespeare 設定と同規模にして、既知の loss（val 1.47 前後）と突き合わせられるようにする
	fn gpu() -> Self {
		Self {
			model: ModelConfig {
				block_size: 256,
				n_layer: 6,
				n_head: 6,
				n_embd: 384,
			},
			batch_size: 64,
			// 64 本を一度に流すと、attention の (B, H, T, T) 行列とその勾配が層ごとに残って 12GB に収まらない。
			// 16 本ずつ 4 回に分けて勾配を足し合わせれば、更新 1 回あたりの勾配は 64 本のときと同じになる
			micro_batch: 16,
			lr: 1e-3,
			min_lr: 1e-4,
			warmup_steps: 100,
			max_steps: 5000,
			// dropout 0 では学習データ約 100 万文字に対して 10M パラメータが丸暗記に回り、val loss が step 1500 前後から上がる
			dropout: 0.2,
			// nanoGPT の既定値。candle の AdamW 既定は 0.01
			weight_decay: 0.1,
			grad_clip: 1.0,
			eval_interval: 250,
			eval_iters: 50,
		}
	}

	// 配線ミスを GPU 無しでも数十秒で見つけたいので、CPU で回る大きさにする
	fn tiny() -> Self {
		Self {
			model: ModelConfig {
				block_size: 32,
				n_layer: 2,
				n_head: 2,
				n_embd: 64,
			},
			batch_size: 16,
			// 勾配累積の配線も tiny で確かめられるよう、1 step を 4 回の backward に分ける
			micro_batch: 4,
			lr: 3e-3,
			min_lr: 3e-4,
			warmup_steps: 20,
			max_steps: 1000,
			// dropout の逆伝播経路も tiny で通すため 0 にしない
			dropout: 0.1,
			weight_decay: 0.1,
			grad_clip: 1.0,
			eval_interval: 200,
			eval_iters: 10,
		}
	}

	// モデルの形は土台のモデルファイルから受け取る。
	// 会話データは約 160 万文字で 1 step（64 本 × block_size 文字）の約 100 倍しかなく、10 epoch で一巡するので、
	// 事前学習より lr を約 1/3 に下げ、step 数も減らす。lr が大きいと事前学習で得た重みが大きく動き、文章力が落ちる
	fn finetune(model: ModelConfig) -> Self {
		Self {
			model,
			batch_size: 64,
			micro_batch: 16,
			lr: 3e-4,
			min_lr: 3e-5,
			warmup_steps: 50,
			max_steps: 1000,
			dropout: 0.1,
			weight_decay: 0.1,
			grad_clip: 1.0,
			eval_interval: 50,
			eval_iters: 20,
		}
	}

	// CLI で上書きした後の値と既定値を同じ条件で検査するため、上書きの適用後に呼ぶ
	fn validate(&self) -> Result<()> {
		if !(0.0..1.0).contains(&self.dropout) {
			candle_core::bail!("--dropout は 0 以上 1 未満である必要がある: {}", self.dropout);
		}
		// 重みは毎 step (1 - lr * weight_decay) 倍されるので、負だと膨らみ続け、lr * weight_decay が 1 以上だと
		// 1 step で 0 か符号反転になり、inf や NaN だと全重みが NaN になる
		if !(self.weight_decay >= 0.0 && self.weight_decay * self.lr < 1.0) {
			candle_core::bail!(
				"--weight-decay は 0 以上で、lr ({}) との積が 1 未満である必要がある: {}",
				self.lr,
				self.weight_decay
			);
		}
		// 負や NaN の上限ではクリップの判定が意味をなさない
		if !(self.grad_clip >= 0.0 && self.grad_clip.is_finite()) {
			candle_core::bail!("--grad-clip は 0 以上の有限値である必要がある: {}", self.grad_clip);
		}
		// 割り切れないと 1 step あたりの本数が batch_size より少なくなり、どちらかが 0 だと勾配が 1 つも作られないので、GPU を確保する前に弾く
		if self.batch_size == 0 || self.micro_batch == 0 || self.batch_size % self.micro_batch != 0 {
			candle_core::bail!(
				"batch_size ({}) と micro_batch ({}) は 1 以上で、batch_size は micro_batch で割り切れる必要がある",
				self.batch_size,
				self.micro_batch
			);
		}
		Ok(())
	}
}

// ニューラルネットは数値しか扱えないので、文字を整数 ID に置き換える必要がある。
// まず「トークン = 1 文字」から始めて、BPE を入れたときの差を後で比較できるようにする
struct Tokenizer {
	itos: Vec<char>,
	stoi: HashMap<char, u32>,
}

impl Tokenizer {
	fn from_texts(texts: &[&str]) -> Self {
		Self::from_chars(distinct_chars(texts))
	}

	// 既存の ID を変えずに、texts に出てきて語彙に無い文字を末尾に足した語彙を作る。
	// 既存の ID が変わらないので、学習済みの wte と lm_head の各行を同じ番号のまま新しい語彙でも使える
	fn extended(&self, texts: &[&str]) -> Self {
		let mut itos = self.itos.clone();
		itos.extend(distinct_chars(texts).into_iter().filter(|c| !self.stoi.contains_key(c)));
		Self::from_chars(itos)
	}

	// 学習時と同じ ID 割り当てを再現するため、保存しておいた itos の並びから組み直す
	fn from_chars(itos: Vec<char>) -> Self {
		let stoi = itos.iter().enumerate().map(|(i, &c)| (c, i as u32)).collect();
		Self { itos, stoi }
	}

	fn vocab_size(&self) -> usize {
		self.itos.len()
	}

	// プロンプトは学習データに無い文字を含みうるので、未知の文字はエラーにする
	fn encode(&self, s: &str) -> Result<Vec<u32>> {
		s.chars()
			.map(|c| {
				self.stoi
					.get(&c)
					.copied()
					.ok_or_else(|| candle_core::Error::Msg(format!("語彙に無い文字: {c:?}")))
			})
			.collect()
	}

	// コーパス全体の ID 列は数十億要素になりうるので、u16 で持ってメモリを u32 の半分に抑える。
	// 1 文字ごとの HashMap 引きも数十億回だと遅いので、文字コードで直接引ける表を使う
	fn encode_corpus(&self, s: &str) -> Result<Vec<u16>> {
		// u16::MAX は表の「語彙に無い」印に使うので、ID は u16::MAX 未満に収まる必要がある
		if self.vocab_size() > u16::MAX as usize {
			candle_core::bail!("語彙が {} 文字あり、u16 の ID に収まらない", self.vocab_size());
		}
		let mut lut = vec![u16::MAX; char::MAX as usize + 1];
		for (i, &c) in self.itos.iter().enumerate() {
			lut[c as usize] = i as u16;
		}
		// 文字数はバイト数以下なので、バイト数で確保すれば途中で再確保（一時的に 2 倍のメモリを使う）が起きない
		let mut ids = Vec::with_capacity(s.len());
		for c in s.chars() {
			match lut[c as usize] {
				u16::MAX => candle_core::bail!("語彙に無い文字: {c:?}"),
				id => ids.push(id),
			}
		}
		ids.shrink_to_fit();
		Ok(ids)
	}

	fn decode(&self, ids: &[u32]) -> String {
		ids.iter().map(|&i| self.itos[i as usize]).collect()
	}
}

// ID の割り当てを実行ごとに変えないよう、文字コード順に並べて返す
fn distinct_chars(texts: &[&str]) -> Vec<char> {
	// 数 GB のコーパスでも 1 パスで速く終わるよう、集合ではなく文字コードで直接引けるフラグ表に出現を記録する
	let mut seen = vec![false; char::MAX as usize + 1];
	for c in texts.iter().flat_map(|t| t.chars()) {
		seen[c as usize] = true;
	}
	(0..seen.len() as u32)
		.filter(|&i| seen[i as usize])
		.filter_map(char::from_u32)
		.collect()
}

// candle_nn の LayerNorm と softmax_last_dim は高速化用の専用カーネルを使う代わりに逆伝播を持たず、
// エラーも出さずに勾配が途切れる（0.11.1 で確認）。学習に使う箇所は基本演算の組み合わせで自前実装する
struct LayerNorm {
	weight: Tensor,
	bias: Tensor,
	eps: f64,
}

impl LayerNorm {
	fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
		// 学習開始時は「正規化するだけ」の恒等変換から始めたいので、スケール 1・シフト 0 で初期化する
		let weight = vb.get_with_hints(dim, "weight", Init::Const(1.0))?;
		let bias = vb.get_with_hints(dim, "bias", Init::Const(0.0))?;
		Ok(Self {
			weight,
			bias,
			eps: 1e-5,
		})
	}

	// 層を重ねるほどベクトルの大きさがばらついて学習が不安定になるので、
	// 各位置のベクトルを平均 0・分散 1 に揃えてから、学習可能なスケールとシフトを掛け直す
	fn forward(&self, x: &Tensor) -> Result<Tensor> {
		let x = x.broadcast_sub(&x.mean_keepdim(D::Minus1)?)?;
		let var = x.sqr()?.mean_keepdim(D::Minus1)?;
		x.broadcast_div(&(var + self.eps)?.sqrt()?)?
			.broadcast_mul(&self.weight)?
			.broadcast_add(&self.bias)
	}
}

// 任意の実数スコアを「全部正で合計 1」の重み・確率として扱いたいので、exp を取って合計で割る
fn softmax(x: &Tensor) -> Result<Tensor> {
	// exp のオーバーフローを避けるため、行ごとの最大値を引いてから指数を取る（結果は数学的に同じ）
	let x = x.broadcast_sub(&x.max_keepdim(D::Minus1)?)?.exp()?;
	x.broadcast_div(&x.sum_keepdim(D::Minus1)?)
}

// ops::dropout は p = 0 でも乱数テンソルとマスクを確保するので、評価・生成用モデルではその確保を省く
fn dropout(x: &Tensor, p: f32) -> Result<Tensor> {
	if p > 0.0 { ops::dropout(x, p) } else { Ok(x.clone()) }
}

// candle の既定の初期値は linear が Kaiming（n_embd 384 で標準偏差約 0.07）、embedding が標準偏差 1 と大きい。
// lm_head の入力は ln_f で分散 1 に揃うので、step 0 の logits の標準偏差は lm_head の重みでほぼ決まり、
// Kaiming だと 0.07 * sqrt(384) ≒ 1.4 になって、loss が一様分布の ln(vocab) より高く出る（65 文字で 4.17 に対して約 5.1）。
// embedding が大きいと、各層が残差に書き足す差分がそれに比べて小さくなり、序盤は層が効きにくい。
// 序盤の step がこれらの修正に費やされないよう、nanoGPT と同じく平均 0・標準偏差 0.02 の正規分布から始め、bias は 0 にする
const INIT_STD: f64 = 0.02;

fn init_normal(stdev: f64) -> Init {
	Init::Randn { mean: 0.0, stdev }
}

// 残差に書き足す出力層（attention と MLP の proj）は 1 層につき 2 回、合計 2 * n_layer 回足し合わされる。
// 独立な値の和の分散は回数に比例して増えるので、標準偏差を 1/sqrt(2 * n_layer) 倍して、層を重ねても残差の大きさを揃える
fn residual_std(cfg: &ModelConfig) -> f64 {
	INIT_STD / (2.0 * cfg.n_layer as f64).sqrt()
}

// 重みとバイアスの名前は candle_nn::linear と同じにして、既存のモデルファイルもそのまま読めるようにする
fn linear_init(in_dim: usize, out_dim: usize, stdev: f64, bias: bool, vb: VarBuilder) -> Result<Linear> {
	let weight = vb.get_with_hints((out_dim, in_dim), "weight", init_normal(stdev))?;
	let bias = bias
		.then(|| vb.get_with_hints(out_dim, "bias", Init::Const(0.0)))
		.transpose()?;
	Ok(Linear::new(weight, bias))
}

fn embedding_init(n: usize, dim: usize, vb: VarBuilder) -> Result<Embedding> {
	Ok(Embedding::new(
		vb.get_with_hints((n, dim), "weight", init_normal(INIT_STD))?,
		dim,
	))
}

// 各位置に「過去のどの位置から情報を持ってくるか」を入力に応じて自分で決めさせたい。
// そのために各位置のベクトルから 3 種類のベクトルを作る:
//   Q（Query）: 自分が探している情報
//   K（Key）  : 自分が持っている情報の見出し
//   V（Value）: 選ばれたときに実際に渡す中身
// 位置 i の Q と各位置の K の内積を関連度スコアにし、softmax で重みに変えて V の重み付き平均を取る。
// どの位置を見るべきか（直前の文字、行頭、対応する引用符など）は学習で獲得させる
struct CausalSelfAttention {
	c_attn: Linear,
	c_proj: Linear,
	n_head: usize,
	dropout: f32,
}

impl CausalSelfAttention {
	fn new(cfg: &ModelConfig, dropout: f32, vb: VarBuilder) -> Result<Self> {
		// Q, K, V を 3 回の matmul に分けるより 1 回にまとめた方が GPU を効率よく使える
		let c_attn = linear_init(cfg.n_embd, 3 * cfg.n_embd, INIT_STD, true, vb.pp("c_attn"))?;
		let c_proj = linear_init(cfg.n_embd, cfg.n_embd, residual_std(cfg), true, vb.pp("c_proj"))?;
		Ok(Self {
			c_attn,
			c_proj,
			n_head: cfg.n_head,
			dropout,
		})
	}

	fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
		let (b, t, c) = x.dims3()?;
		let hd = c / self.n_head;
		let qkv = self.c_attn.forward(x)?;
		// 1 種類の見方だけだと「直前の文字」と「行頭」を同時に参照できないので、
		// n_embd 次元を n_head 個に分けて別々の観点で attention させる（multi-head）。
		// ヘッドごとに独立して計算したいので (B, H, T, hd) に並べ替える
		let split = |i: usize| {
			qkv.narrow(2, i * c, c)?
				.reshape((b, t, self.n_head, hd))?
				.transpose(1, 2)?
				.contiguous()
		};
		let (q, k, v) = (split(0)?, split(1)?, split(2)?);

		// 内積の分散が hd に比例して大きくなり softmax が尖りすぎるのを防ぐため sqrt(hd) で割る
		let att = (q.matmul(&k.t()?.contiguous()?)? / (hd as f64).sqrt())?;
		// 未来の位置のスコアを -inf にしておけば、softmax 後の重みがちょうど 0 になる
		let att = att.broadcast_add(mask)?;
		// 参照先の一部を毎 step ランダムに切り、特定の位置の組み合わせだけに依存する重みを作らせない
		let att = dropout(&softmax(&att)?, self.dropout)?;

		// 各ヘッドが集めた情報を連結し、c_proj で混ぜてから残差に戻す
		let y = att.matmul(&v)?.transpose(1, 2)?.reshape((b, t, c))?;
		dropout(&self.c_proj.forward(&y)?, self.dropout)
	}
}

// attention は位置どうしで情報を混ぜるだけなので、集めた情報を加工する処理を位置ごとに独立して入れる。
// 一度 4 倍の次元に広げて GELU で非線形を挟み、元の次元に戻す。学習した知識の多くはここに入るとされる
struct Mlp {
	fc: Linear,
	proj: Linear,
	dropout: f32,
}

impl Mlp {
	fn new(cfg: &ModelConfig, dropout: f32, vb: VarBuilder) -> Result<Self> {
		Ok(Self {
			fc: linear_init(cfg.n_embd, 4 * cfg.n_embd, INIT_STD, true, vb.pp("fc"))?,
			proj: linear_init(4 * cfg.n_embd, cfg.n_embd, residual_std(cfg), true, vb.pp("proj"))?,
			dropout,
		})
	}

	fn forward(&self, x: &Tensor) -> Result<Tensor> {
		dropout(&self.proj.forward(&self.fc.forward(x)?.gelu()?)?, self.dropout)
	}
}

struct Block {
	ln1: LayerNorm,
	attn: CausalSelfAttention,
	ln2: LayerNorm,
	mlp: Mlp,
}

impl Block {
	fn new(cfg: &ModelConfig, dropout: f32, vb: VarBuilder) -> Result<Self> {
		Ok(Self {
			ln1: LayerNorm::new(cfg.n_embd, vb.pp("ln1"))?,
			attn: CausalSelfAttention::new(cfg, dropout, vb.pp("attn"))?,
			ln2: LayerNorm::new(cfg.n_embd, vb.pp("ln2"))?,
			mlp: Mlp::new(cfg, dropout, vb.pp("mlp"))?,
		})
	}

	fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
		// x を上書きせず `x + f(x)` で差分を書き足す形（残差接続）にする。各層は「今の表現に何を足すか」だけを
		// 学べばよくなり、逆伝播の勾配も足し算を素通りして下の層まで届く。
		// その素通りの経路を邪魔しないよう、正規化は f の入口だけに置く（pre-norm）
		let x = (x + self.attn.forward(&self.ln1.forward(x)?, mask)?)?;
		&x + self.mlp.forward(&self.ln2.forward(&x)?)?
	}
}

struct Gpt {
	wte: Embedding,
	wpe: Embedding,
	blocks: Vec<Block>,
	ln_f: LayerNorm,
	lm_head: Linear,
	dropout: f32,
}

impl Gpt {
	// dropout はモデルファイルに保存しない。学習中の更新用モデルだけが 0 より大きい値を受け取る
	fn new(cfg: &ModelConfig, vocab_size: usize, dropout: f32, vb: VarBuilder) -> Result<Self> {
		let blocks = (0..cfg.n_layer)
			.map(|i| Block::new(cfg, dropout, vb.pp(format!("h.{i}"))))
			.collect::<Result<_>>()?;
		Ok(Self {
			// ID のままだと「3 番と 4 番は近い」という意味のない大小関係がつくので、ID ごとに学習可能なベクトルを割り当てる
			wte: embedding_init(vocab_size, cfg.n_embd, vb.pp("wte"))?,
			// attention は入力を順序のない集合として扱うので、「何文字目か」を位置ごとのベクトルとして足して教える。
			// RoPE と比較するための基準として、最初は GPT-2 と同じ学習型の位置埋め込みにする
			wpe: embedding_init(cfg.block_size, cfg.n_embd, vb.pp("wpe"))?,
			blocks,
			ln_f: LayerNorm::new(cfg.n_embd, vb.pp("ln_f"))?,
			lm_head: linear_init(cfg.n_embd, vocab_size, INIT_STD, false, vb.pp("lm_head"))?,
			dropout,
		})
	}

	fn forward(&self, idx: &Tensor) -> Result<Tensor> {
		let (_b, t) = idx.dims2()?;
		let dev = idx.device();
		let pos = Tensor::arange(0u32, t as u32, dev)?;
		let x = self.wte.forward(idx)?.broadcast_add(&self.wpe.forward(&pos)?)?;
		let mut x = dropout(&x, self.dropout)?;
		let mask = causal_mask(t, dev)?;
		for block in &self.blocks {
			x = block.forward(&x, &mask)?;
		}
		// 最後のベクトルから語彙の数だけスコアを出したいので、vocab_size 次元に写す。softmax は loss 側と生成側でかける
		self.lm_head.forward(&self.ln_f.forward(&x)?)
	}
}

// 未来のトークンを見て答えをカンニングできないよう、softmax 前に右上を -inf で潰す
fn causal_mask(t: usize, dev: &Device) -> Result<Tensor> {
	let mask: Vec<f32> = (0..t)
		.flat_map(|i| (0..t).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
		.collect();
	Tensor::from_vec(mask, (t, t), dev)
}

// 学習と評価で窓を切り出す元の ID 列。
// mask が None なら全文字を正解として loss に数え（事前学習）、Some なら true の文字だけを数える（会話データの AI の返事）
#[derive(Clone, Copy)]
struct Corpus<'a> {
	ids: &'a [u16],
	mask: Option<&'a [bool]>,
}

impl<'a> Corpus<'a> {
	fn plain(ids: &'a [u16]) -> Self {
		Self { ids, mask: None }
	}
}

struct Batch {
	x: Tensor,
	y: Tensor,
	// 位置ごとの loss の重み（1 か 0）。形は (B * T)
	weight: Tensor,
	// weight の合計。loss の平均を取るときの分母
	n_targets: f64,
}

// sources は (抽出元, 抽出割合) の組。1 本ごとに割合に従って抽出元を選ぶ。
// finetune で会話データと文章コーパスを混ぜるのに使う。会話データだけで学習すると、事前学習で得た文章力が失われていく
fn get_batch(sources: &[(Corpus, f64)], b: usize, t: usize, rng: &mut StdRng, dev: &Device) -> Result<Batch> {
	let total: f64 = sources.iter().map(|(_, w)| w).sum();
	let mut xs = Vec::with_capacity(b * t);
	let mut ys = Vec::with_capacity(b * t);
	let mut ws = Vec::with_capacity(b * t);
	for _ in 0..b {
		// 抽出元が 1 つのときは乱数を引かない。引くと同じ --seed でも窓の位置の列が変わり、抽選を入れる前の train の結果を再現できない
		let src = if sources.len() == 1 {
			sources[0].0
		} else {
			let mut r = rng.random::<f64>() * total;
			// 浮動小数の丸めで r が合計をわずかに超えた場合は最後の抽出元を使う
			sources
				.iter()
				.find(|(_, w)| {
					r -= w;
					r < 0.0
				})
				.unwrap_or(&sources[sources.len() - 1])
				.0
		};
		let data = src.ids;
		let i = rng.random_range(0..data.len() - t - 1);
		xs.extend(data[i..i + t].iter().map(|&id| id as u32));
		// 全位置で同時に「次の 1 トークン」を予測させるため、入力を 1 つずらしたものを正解にする。
		// causal mask があるので、長さ T の 1 本から「1 文字目まで → 2 文字目」…「T 文字目まで → T+1 文字目」の T 問を 1 回で解かせられる
		ys.extend(data[i + 1..i + t + 1].iter().map(|&id| id as u32));
		// mask は正解側の文字に対応させる。入力の文字ではなく、予測する文字が AI の返事かどうかで数えるかを決める
		match src.mask {
			Some(m) => ws.extend(m[i + 1..i + t + 1].iter().map(|&c| if c { 1.0f32 } else { 0.0 })),
			None => ws.extend(std::iter::repeat_n(1.0f32, t)),
		}
	}
	let n_targets = ws.iter().map(|&w| w as f64).sum();
	Ok(Batch {
		x: Tensor::from_vec(xs, (b, t), dev)?,
		y: Tensor::from_vec(ys, (b * t, 1), dev)?,
		weight: Tensor::from_vec(ws, b * t, dev)?,
		n_targets,
	})
}

// 「正解の文字につけた確率の -log」（cross entropy）を最小化したい。正解に確率 1 を出せば 0 になる。
// 全文字を当てずっぽうにすると ln(vocab) で、65 文字なら 4.17。loss 2.0 は e^2 ≒ 7 択まで絞れている状態。
// 返り値は weight を掛けた合計で、平均にはしない。勾配累積では 1 step 分の n_targets の合計で割り、
// micro batch ごとに数える文字数が違っても、1 step 内の全対象文字の平均 loss の勾配にする
fn loss_sum(model: &Gpt, batch: &Batch) -> Result<Tensor> {
	let logits = model.forward(&batch.x)?;
	let (b, t, v) = logits.dims3()?;
	// candle_nn::loss::cross_entropy と同じ計算で、平均を取る前の位置ごとの値を得るために展開している
	let nll = ops::log_softmax(&logits.reshape((b * t, v))?, D::Minus1)?
		.gather(&batch.y, 1)?
		.flatten_all()?
		.neg()?;
	(nll * &batch.weight)?.sum_all()
}

// 後半も lr が大きいままだと学習データへの過適合が進むので、ウォームアップ後は cosine で min_lr まで下げる
fn lr_at(cfg: &TrainConfig, step: usize) -> f64 {
	if step < cfg.warmup_steps {
		return cfg.lr * (step + 1) as f64 / cfg.warmup_steps as f64;
	}
	let decay_steps = cfg.max_steps.saturating_sub(cfg.warmup_steps).max(1);
	let progress = ((step - cfg.warmup_steps) as f64 / decay_steps as f64).min(1.0);
	let coeff = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
	cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)
}

// candle_nn::AdamW は全パラメータに同じ weight_decay を掛けるので、掛けたい重みと掛けたくない重みで AdamW を 2 つに分ける。
// 行列（linear の重みと埋め込み）は値が大きいほど特定の入力を強く覚え込むので縮める。
// 1 次元のもの（bias と LayerNorm のスケール・シフト）は出力の位置と大きさを合わせる値で、パラメータ全体のごく一部しかない。
// 縮めても丸暗記はほとんど減らず、正規化後のスケールが 0 に引っ張られて学習の邪魔になるだけなので、nanoGPT に合わせて外す
struct ParamGroups {
	decay: AdamW,
	no_decay: AdamW,
}

impl ParamGroups {
	fn new(vars: Vec<Var>, params: ParamsAdamW) -> Result<Self> {
		let (decay, no_decay): (Vec<Var>, Vec<Var>) = vars.into_iter().partition(|v| v.rank() >= 2);
		Ok(Self {
			decay: AdamW::new(decay, params.clone())?,
			no_decay: AdamW::new(
				no_decay,
				ParamsAdamW {
					weight_decay: 0.0,
					..params
				},
			)?,
		})
	}

	fn learning_rate(&self) -> f64 {
		self.decay.learning_rate()
	}

	fn set_learning_rate(&mut self, lr: f64) {
		self.decay.set_learning_rate(lr);
		self.no_decay.set_learning_rate(lr);
	}

	fn step(&mut self, grads: &GradStore) -> Result<()> {
		self.decay.step(grads)?;
		self.no_decay.step(grads)
	}
}

// 全パラメータの勾配を 1 本の長いベクトルとみなした L2 ノルムを求め、max_norm を超えていれば全勾配に同じ倍率を掛けて縮める。
// パラメータごとに別々に縮めると更新の向きが変わってしまうので、全体で 1 つの倍率にする。返り値は縮める前のノルム
fn clip_grad_norm(grads: &mut GradStore, vars: &[Tensor], max_norm: f64) -> Result<f32> {
	let sq = vars
		.iter()
		.filter_map(|v| grads.get(v))
		.map(|g| g.sqr()?.sum_all())
		.collect::<Result<Vec<_>>>()?;
	let norm = Tensor::stack(&sq, 0)?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
	// NaN は大小比較が常に偽でクリップを素通りし、AdamW が全重みとモーメントを NaN で上書きする。
	// その後は最終 step まで NaN のまま計算が続くだけなので、ここで止める
	if !norm.is_finite() {
		candle_core::bail!("勾配ノルムが有限の値でない（勾配に NaN か inf があるか、二乗和が f32 であふれた）: {norm}");
	}
	if max_norm > 0.0 && norm as f64 > max_norm {
		let scale = max_norm / norm as f64;
		for v in vars {
			if let Some(g) = grads.remove(v) {
				grads.insert(v, (g * scale)?);
			}
		}
	}
	Ok(norm)
}

// 1 バッチの loss はノイズが大きく改良の効果と見分けにくいので、複数バッチの平均で比べる。
// mask 付きのデータではバッチごとに数える文字数が違うので、バッチの平均の平均ではなく、全対象文字の平均にする。
// model には計算グラフを作らない評価用モデル（fit の eval_model）を渡す
fn estimate_loss(model: &Gpt, data: Corpus, cfg: &TrainConfig, rng: &mut StdRng, dev: &Device) -> Result<f32> {
	let (mut sum, mut n) = (0.0f64, 0.0f64);
	for _ in 0..cfg.eval_iters {
		let batch = get_batch(&[(data, 1.0)], cfg.batch_size, cfg.model.block_size, rng, dev)?;
		sum += loss_sum(model, &batch)?.to_scalar::<f32>()? as f64;
		n += batch.n_targets;
	}
	// 0 件を loss 0 として返すと、fit がそれを最小値として保存し、以後の重みが保存されなくなる
	if n == 0.0 {
		candle_core::bail!("評価に使ったバッチに、loss に数える文字（AI の発言）が 1 文字も無い");
	}
	Ok((sum / n) as f32)
}

// on_token は 1 文字引くたびに呼ばれ、CLI はここで逐次表示する。
// Break を返すとその文字を末尾に含めたまま生成を止める。chat は返事の終わりの印で止めるのに使う
#[allow(clippy::too_many_arguments)]
fn generate(
	model: &Gpt,
	start: &[u32],
	max_new: usize,
	temperature: f64,
	block_size: usize,
	rng: &mut StdRng,
	dev: &Device,
	mut on_token: impl FnMut(u32) -> Result<ControlFlow<()>>,
) -> Result<Vec<u32>> {
	// 空の文脈では forward に長さ 0 の系列が渡り、最後の位置が存在しない
	if start.is_empty() {
		candle_core::bail!("生成の開始文字列が空");
	}
	let mut ids = start.to_vec();
	for _ in 0..max_new {
		// 仕組みを追いやすくするため毎回全文脈を計算し直す。KV キャッシュは推論側を作るときに入れる
		let ctx = &ids[ids.len().saturating_sub(block_size)..];
		let logits = model.forward(&Tensor::new(ctx, dev)?.unsqueeze(0)?)?;
		// 次の文字の予測に使うのは最後の位置の出力だけ。
		// temperature で割ってから softmax にかけ、1 未満なら確率の高い文字に寄せ、1 より大きければばらつかせる
		let last = (logits.i((0, ctx.len() - 1))? / temperature)?;
		let probs = softmax(&last)?.to_vec1::<f32>()?;
		let dist = WeightedIndex::new(&probs).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
		let id = dist.sample(rng) as u32;
		ids.push(id);
		if on_token(id)?.is_break() {
			break;
		}
	}
	Ok(ids)
}

// モデルファイルには重みと一緒に、生成時に Gpt と Tokenizer を組み直すための値を入れる。
// 学習データのテキストが無くても generate できるよう、語彙もここに持たせる
const META_CONFIG: &str = "meta.config";
const META_VOCAB: &str = "meta.vocab";

fn save_model(path: &Path, varmap: &VarMap, cfg: &ModelConfig, tok: &Tokenizer) -> Result<()> {
	let mut tensors: HashMap<String, Tensor> = varmap
		.data()
		.lock()
		.unwrap()
		.iter()
		.map(|(name, var)| (name.clone(), var.as_tensor().clone()))
		.collect();
	tensors.insert(
		META_CONFIG.to_string(),
		Tensor::new(cfg.to_vec().as_slice(), &Device::Cpu)?,
	);
	let vocab: Vec<u32> = tok.itos.iter().map(|&c| c as u32).collect();
	tensors.insert(META_VOCAB.to_string(), Tensor::new(vocab.as_slice(), &Device::Cpu)?);
	candle_core::safetensors::save(&tensors, path)
}

fn load_model(path: &Path, dev: &Device) -> Result<(Gpt, ModelConfig, Tokenizer)> {
	let (tensors, cfg, tok) = load_tensors(path, dev)?;
	// Var を経由せずテンソルから直接組むので、生成中に計算グラフは作られない
	let model = Gpt::new(
		&cfg,
		tok.vocab_size(),
		0.0,
		VarBuilder::from_tensors(tensors, DType::F32, dev),
	)?;
	Ok((model, cfg, tok))
}

// 返り値の HashMap は meta.* を取り除いた重みだけを持つ。finetune はこれを学習用の Var に書き写す
fn load_tensors(path: &Path, dev: &Device) -> Result<(HashMap<String, Tensor>, ModelConfig, Tokenizer)> {
	let mut tensors = candle_core::safetensors::load(path, dev)
		.map_err(|e| candle_core::Error::Msg(format!("{}: {e}", path.display())))?;
	let mut take_meta = |key: &str| -> Result<Vec<u32>> {
		tensors
			.remove(key)
			.ok_or_else(|| {
				candle_core::Error::Msg(format!(
					"{}: {key} が無い。train サブコマンドで保存し直す必要がある",
					path.display()
				))
			})?
			.to_vec1::<u32>()
	};
	let cfg = ModelConfig::from_slice(&take_meta(META_CONFIG)?)?;
	let itos = take_meta(META_VOCAB)?
		.into_iter()
		.map(|v| {
			char::from_u32(v).ok_or_else(|| candle_core::Error::Msg(format!("{META_VOCAB} に不正な文字コード: {v}")))
		})
		.collect::<Result<Vec<char>>>()?;
	Ok((tensors, cfg, Tokenizer::from_chars(itos)))
}

#[derive(Parser)]
#[command(version, about = "文字単位の最小 GPT を学習し、文章の生成と対話を行う")]
struct Cli {
	/// CUDA が使える環境でも CPU で実行する
	#[arg(long, global = true)]
	cpu: bool,

	#[command(subcommand)]
	command: Command,
}

#[derive(Subcommand)]
enum Command {
	/// テキストファイルでモデルを学習し、重みを保存する
	Train(TrainArgs),
	/// train で保存したモデルを会話データで追加学習し、対話用の重みを保存する
	Finetune(FinetuneArgs),
	/// 保存したモデルで文章を生成する
	Generate(GenerateArgs),
	/// finetune で保存したモデルと対話する
	Chat(ChatArgs),
}

// train と finetune で共通の学習オプション
#[derive(Args)]
struct LoopArgs {
	/// dropout 率（省略時は設定ごとの既定値）。データが十分大きく丸暗記が起きないなら 0 でよい
	#[arg(long)]
	dropout: Option<f32>,

	/// 行列の重みに掛ける weight decay（省略時は設定ごとの既定値）。bias と LayerNorm には掛けない
	#[arg(long)]
	weight_decay: Option<f64>,

	/// 勾配ノルムの上限（省略時は設定ごとの既定値）。0 でクリップしない
	#[arg(long)]
	grad_clip: Option<f64>,

	/// 学習ステップ数（省略時は設定ごとの既定値）
	#[arg(long)]
	steps: Option<usize>,

	/// バッチの抽出と学習後のサンプル生成に使う乱数シード
	#[arg(long, default_value_t = 1337)]
	seed: u64,

	/// 学習後に生成するサンプルの文字数（0 で生成しない）
	#[arg(long, default_value_t = 500)]
	sample_len: usize,

	/// 予測の当たり具合で検証データを色分けする表示を止め、評価ログを 1 行ずつ流す（標準出力が端末でないときは指定しなくても止まる）
	#[arg(long)]
	no_viz: bool,

	/// 色分け表示の描き直し間隔（step 数）。表示が止まっているときは使わない
	#[arg(long, default_value_t = 10)]
	viz_every: usize,
}

impl LoopArgs {
	// 指定された値で cfg を上書きして検査し、色分け表示を使うかどうかを返す
	fn apply(&self, cfg: &mut TrainConfig) -> Result<bool> {
		if let Some(steps) = self.steps {
			cfg.max_steps = steps;
		}
		if let Some(p) = self.dropout {
			cfg.dropout = p;
		}
		if let Some(wd) = self.weight_decay {
			cfg.weight_decay = wd;
		}
		if let Some(c) = self.grad_clip {
			cfg.grad_clip = c;
		}
		cfg.validate()?;
		// ファイルやパイプへのリダイレクト先と、エスケープシーケンスを解釈しない TERM=dumb の端末には、描き直しの制御コードを書き込まない
		let use_viz =
			!self.no_viz && std::io::stdout().is_terminal() && std::env::var_os("TERM").is_none_or(|t| t != "dumb");
		if use_viz && self.viz_every == 0 {
			candle_core::bail!("--viz-every は 1 以上である必要がある");
		}
		Ok(use_viz)
	}
}

#[derive(Args)]
struct TrainArgs {
	/// 学習データのテキストファイル
	#[arg(default_value = "input.txt")]
	data: PathBuf,

	/// 検証データのテキストファイル（省略時は学習データの末尾 1 割を検証に回す）
	#[arg(long)]
	val: Option<PathBuf>,

	/// CPU で数十秒で終わる小さい設定を使う
	#[arg(long)]
	tiny: bool,

	/// 学習済みモデルの保存先
	#[arg(short, long, default_value = "model.safetensors")]
	out: PathBuf,

	#[command(flatten)]
	common: LoopArgs,
}

#[derive(Args)]
struct FinetuneArgs {
	/// 会話データの JSONL（1 行 1 会話。scripts/download-chat.bat で作れる）
	#[arg(default_value = "data/chat-train.jsonl")]
	data: PathBuf,

	/// 検証用の会話データの JSONL（省略時は会話データの末尾 1 割を検証に回す）
	#[arg(long)]
	val: Option<PathBuf>,

	/// train で保存した土台のモデルファイル。モデルの形と語彙はこのファイルから引き継ぐ
	#[arg(long)]
	base: PathBuf,

	/// 会話データに混ぜる文章コーパスのテキストファイル。事前学習で得た文章力が失われるのを抑える。
	/// テキストと ID 列を同時に持つので、ファイルの大きさの約 3 倍のメモリを使う
	#[arg(long)]
	mix: Option<PathBuf>,

	/// 1 step の系列の本数のうち --mix の文章コーパスから取る割合。
	/// 会話データは AI の発言の文字だけを loss に数えるので、loss に数える文字数に占める文章コーパスの割合はこの値より大きくなる
	/// （会話データの約 7 割が AI の発言のとき、0.25 で約 3 割）
	#[arg(long, default_value_t = 0.25, requires = "mix")]
	mix_ratio: f64,

	/// 追加学習したモデルの保存先
	#[arg(short, long, default_value = "chat.safetensors")]
	out: PathBuf,

	#[command(flatten)]
	common: LoopArgs,
}

#[derive(Args)]
struct GenerateArgs {
	/// train で保存したモデルファイル
	#[arg(short, long, default_value = "model.safetensors")]
	model: PathBuf,

	/// 生成の書き出し（省略時は語彙の先頭の文字）
	#[arg(short, long)]
	prompt: Option<String>,

	/// 生成する文字数
	#[arg(short = 'n', long, default_value_t = 500)]
	len: usize,

	/// 1 未満で確率の高い文字に寄り、1 より大きいとばらつく
	#[arg(short, long, default_value_t = 0.8)]
	temperature: f64,

	/// 乱数シード（省略時は実行ごとに変わる）
	#[arg(long)]
	seed: Option<u64>,
}

#[derive(Args)]
struct ChatArgs {
	/// finetune で保存したモデルファイル
	#[arg(short, long, default_value = "chat.safetensors")]
	model: PathBuf,

	/// 1 回の返事の最大文字数。返事の終わりの印が出なくてもこの文字数で打ち切る
	#[arg(short = 'n', long, default_value_t = 300)]
	max_len: usize,

	/// 1 未満で確率の高い文字に寄り、1 より大きいとばらつく
	#[arg(short, long, default_value_t = 0.8)]
	temperature: f64,

	/// 乱数シード（省略時は実行ごとに変わる）
	#[arg(long)]
	seed: Option<u64>,
}

fn select_device(cpu: bool) -> Result<Device> {
	if cpu {
		Ok(Device::Cpu)
	} else {
		Device::cuda_if_available(0)
	}
}

fn read_text(path: &Path) -> Result<String> {
	std::fs::read_to_string(path).map_err(|e| candle_core::Error::Msg(format!("{}: {e}", path.display())))
}

// get_batch は block_size + 1 文字の窓を取るので、それより長くないと窓の開始位置の範囲が空になる
fn ensure_len(what: &str, len: usize, block_size: usize) -> Result<()> {
	if len <= block_size + 1 {
		candle_core::bail!(
			"{what} が短すぎる（{len} 文字。block_size + 1 = {} 文字を超える必要がある）",
			block_size + 1
		);
	}
	Ok(())
}

// train と finetune で共通の学習ループに渡す値。重みの初期値とデータの用意は呼び出し側が行う
struct Fit<'a> {
	cfg: &'a TrainConfig,
	tok: &'a Tokenizer,
	// 空なら fit が Gpt::new の初期値で Var を作る。土台のモデルから始める場合は、呼び出し側が先に Var を作って値を書き込んでおく
	varmap: &'a VarMap,
	// 重みの更新に使う (抽出元, 抽出割合)
	train: &'a [(Corpus<'a>, f64)],
	// 評価ログの train loss を測るデータ。finetune では文章コーパスを含めず、val と同じく会話データの AI の返事の loss にする
	train_eval: Corpus<'a>,
	val: Corpus<'a>,
	out: &'a Path,
	// 色分け表示を使う場合の (描き直し間隔, 表示を始める位置の直前の文字)
	viz: Option<(usize, Option<char>)>,
	summary: String,
}

// 返り値は保存したモデルの val loss
fn fit(f: Fit, rng: &mut StdRng, dev: &Device) -> Result<f32> {
	let cfg = f.cfg;
	let model = Gpt::new(
		&cfg.model,
		f.tok.vocab_size(),
		cfg.dropout,
		VarBuilder::from_varmap(f.varmap, DType::F32, dev),
	)?;
	// Var から計算した結果は逆伝播用に入力への参照を持ち続けるので、forward だけでも中間テンソルが最後まで解放されない。
	// 評価と生成は逆伝播しないので、detach した重みで同じ構造のモデルをもう 1 つ組んでそちらを使う。
	// detach は Var と同じストレージを指し、AdamW の更新（Var::set）はストレージを書き換えるので、毎 step の重みがそのまま見える
	let detached: HashMap<String, Tensor> = f
		.varmap
		.data()
		.lock()
		.unwrap()
		.iter()
		.map(|(name, var)| (name.clone(), var.as_tensor().detach()))
		.collect();
	let eval_model = Gpt::new(
		&cfg.model,
		f.tok.vocab_size(),
		0.0,
		VarBuilder::from_tensors(detached, DType::F32, dev),
	)?;
	let n_params: usize = f.varmap.all_vars().iter().map(|v| v.elem_count()).sum();
	let summary = format!("{} / params: {:.2}M", f.summary, n_params as f64 / 1e6);
	println!("{summary}");

	let mut opt = ParamGroups::new(
		f.varmap.all_vars(),
		ParamsAdamW {
			lr: lr_at(cfg, 0),
			// 1 step あたりのトークン数が少なく勾配のばらつきが大きいので、nanoGPT の Shakespeare 設定に合わせて
			// 2 次モーメントの平均を既定の 0.999 より短い期間で取る
			beta2: 0.99,
			weight_decay: cfg.weight_decay,
			..Default::default()
		},
	)?;
	// 勾配が全パラメータに届いているかの検査に使う
	let vars: Vec<Tensor> = f.varmap.all_vars().iter().map(|v| v.as_tensor().clone()).collect();
	let t0 = Instant::now();
	// 最終 step の重みは過学習で val loss 最小時より悪化しうるので、val loss が最小値を更新するたびに保存する
	let mut best_val = f32::INFINITY;
	// 直前の step のクリップ前の勾配ノルム。評価ログに出し、クリップがどのくらいの頻度で効いているかの目安にする
	let mut grad_norm: Option<f32> = None;
	let mut viz = f
		.viz
		.map(|(_, start_after)| Viz::new(f.tok, cfg.model.block_size, f.val.ids, start_after))
		.transpose()?;
	// Viz::new が画面を消すので、学習後に出力する評価ログの先頭に同じ行を入れる
	if let Some(v) = &mut viz {
		v.push_log(summary);
	}

	for step in 0..=cfg.max_steps {
		if step % cfg.eval_interval == 0 || step == cfg.max_steps {
			let tr = estimate_loss(&eval_model, f.train_eval, cfg, rng, dev)?;
			let va = estimate_loss(&eval_model, f.val, cfg, rng, dev)?;
			let improved = va < best_val;
			if improved {
				best_val = va;
				save_model(f.out, f.varmap, &cfg.model, f.tok)?;
			}
			let line = format!(
				"step {step:5} | train {tr:.4} | val {va:.4} | lr {:.2e} | gnorm {} | {:.1}s{}",
				opt.learning_rate(),
				grad_norm.map_or_else(|| "-".to_string(), |n| format!("{n:.2}")),
				t0.elapsed().as_secs_f32(),
				if improved { " | saved" } else { "" }
			);
			// 可視化中は画面を描き直すので、評価ログは流さずフレームの上部に表示させる
			match &mut viz {
				Some(v) => v.push_log(line),
				None => println!("{line}"),
			}
		}
		if let (Some(v), Some((every, _))) = (&viz, f.viz)
			&& (step % every == 0 || step == cfg.max_steps)
		{
			let progress = format!(
				"step {step:5}/{} | lr {:.2e} | {:.1}s",
				cfg.max_steps,
				opt.learning_rate(),
				t0.elapsed().as_secs_f32()
			);
			v.draw(&eval_model, &progress, dev)?;
		}
		if step == cfg.max_steps {
			break;
		}
		// loss を下げるには各重みをどちらにどれだけ動かせばいいかを、逆伝播で全パラメータ分まとめて求める。
		// micro_batch ずつ backward して勾配を足し合わせる。各 loss を 1 step 分の対象文字数で割っておけば、
		// 合計は batch_size 本の全対象文字の平均 loss の勾配と同じになる。
		// loss（計算グラフ）は 1 回ごとに捨てるので、同時に VRAM に載る中間テンソルは micro_batch 本ぶんだけで済む
		let grad_accum = cfg.batch_size / cfg.micro_batch;
		let batches = (0..grad_accum)
			.map(|_| get_batch(f.train, cfg.micro_batch, cfg.model.block_size, rng, dev))
			.collect::<Result<Vec<_>>>()?;
		// finetune は AI の発言が無い会話データを事前に弾くが、ランダムな窓が偶然すべて User の発言に収まる step はありうる。
		// その step は 0 除算を避けて loss と勾配を 0 にし、AdamW は weight decay だけを掛ける
		let n_targets = batches.iter().map(|b| b.n_targets).sum::<f64>().max(1.0);
		let mut grads = GradStore::default();
		for batch in &batches {
			let loss = (loss_sum(&model, batch)? / n_targets)?;
			// extend は同じ TensorId の勾配を足し合わせ、無ければそのまま入れる
			grads.extend(loss.backward()?)?;
		}
		// 部品を差し替えたときに勾配がエラーも出さず途切れる事故を検出したい。勾配の無い重みは AdamW が黙って更新を飛ばし、
		// clip_grad_norm もノルムに数えないので、気づかないまま固まり続ける。確認は表の引きだけで軽いので毎 step 行う。
		// ただし同じ重みに別経路から勾配が届く場合（Q/K 側だけ切れて V 側は生きている等）は素通りするので、
		// 怪しいときは loss の下がり方で確認する
		let missing = vars.iter().filter(|v| grads.get(v).is_none()).count();
		if missing > 0 {
			candle_core::bail!("step {step}: {missing} 個のパラメータに勾配が届いていない");
		}
		// 評価ログは eval_interval ごとにしか出ないので、どの step で壊れたかをエラーに入れる
		let norm = clip_grad_norm(&mut grads, &vars, cfg.grad_clip)
			.map_err(|e| candle_core::Error::Msg(format!("step {step}: {e}")))?;
		grad_norm = Some(norm);
		// 勾配そのままだと重みごとにスケールがばらばらなので、AdamW で過去の勾配の平均と大きさを使って歩幅を揃えて更新する
		opt.set_learning_rate(lr_at(cfg, step));
		opt.step(&grads)?;
	}

	// フレームには評価ログの末尾しか出ないので、学習後に全体を残す
	if let Some(v) = &viz {
		println!();
		for line in v.log() {
			println!("{line}");
		}
	}
	println!("best val {best_val:.4}: {}", f.out.display());
	Ok(best_val)
}

fn train(args: TrainArgs, dev: &Device) -> Result<()> {
	let mut cfg = if args.tiny {
		TrainConfig::tiny()
	} else {
		TrainConfig::gpu()
	};
	let use_viz = args.common.apply(&mut cfg)?;

	let text = read_text(&args.data)?;
	let val_text = args.val.as_deref().map(read_text).transpose()?;
	// 検証データにしか出てこない文字があると encode が失敗するので、語彙は両方から作る
	let texts: Vec<&str> = std::iter::once(text.as_str()).chain(val_text.as_deref()).collect();
	let tok = Tokenizer::from_texts(&texts);
	let data = tok.encode_corpus(&text)?;
	let val_data = val_text.as_deref().map(|t| tok.encode_corpus(t)).transpose()?;
	// 数 GB のテキストを ID 列と二重に持たないよう、encode し終えたら手放す
	drop(texts);
	drop(text);
	drop(val_text);
	let (train, val) = match &val_data {
		Some(v) => (data.as_slice(), v.as_slice()),
		// 学習に使っていない部分で測らないと、丸暗記しただけでも loss が下がって見えてしまう
		None => data.split_at(data.len() * 9 / 10),
	};
	let block = cfg.model.block_size;
	ensure_len(&format!("学習データ {}", args.data.display()), train.len(), block)?;
	match &args.val {
		Some(p) => ensure_len(&format!("検証データ {}", p.display()), val.len(), block)?,
		None => ensure_len(
			&format!("検証データ（{} の末尾 1 割）", args.data.display()),
			val.len(),
			block,
		)?,
	}

	let varmap = VarMap::new();
	let mut rng = StdRng::seed_from_u64(args.common.seed);
	let summary = format!(
		"device: {:?} / chars: train {} + val {} / vocab: {}",
		dev,
		train.len(),
		val.len(),
		tok.vocab_size()
	);
	fit(
		Fit {
			cfg: &cfg,
			tok: &tok,
			varmap: &varmap,
			train: &[(Corpus::plain(train), 1.0)],
			train_eval: Corpus::plain(train),
			val: Corpus::plain(val),
			out: &args.out,
			viz: use_viz.then_some((args.common.viz_every, Some('\n'))),
			summary,
		},
		&mut rng,
		dev,
	)?;

	if args.common.sample_len > 0 {
		// eval_model は最終 step の重みを参照するので、保存済みファイルと同じ重みで生成するために読み直す
		let (best_model, _, _) = load_model(&args.out, dev)?;
		let out = generate(
			&best_model,
			&[0],
			args.common.sample_len,
			0.8,
			block,
			&mut rng,
			dev,
			|_| Ok(ControlFlow::Continue(())),
		)?;
		println!("----\n{}", tok.decode(&out));
	}
	Ok(())
}

// 土台のモデルの重みを varmap の Var に書き写す。
// 語彙を足した分の wte と lm_head の行は、Gpt::new の初期値（平均 0・標準偏差 0.02 の正規分布）のまま残す
fn init_from_base(
	varmap: &VarMap,
	mut base: HashMap<String, Tensor>,
	cfg: &ModelConfig,
	base_vocab: usize,
	vocab: usize,
	dev: &Device,
) -> Result<()> {
	Gpt::new(cfg, vocab, 0.0, VarBuilder::from_varmap(varmap, DType::F32, dev))?;
	for (name, var) in varmap.data().lock().unwrap().iter() {
		let src = base
			.remove(name)
			.ok_or_else(|| candle_core::Error::Msg(format!("土台のモデルに {name} が無い")))?;
		// 行 i は ID i の文字に対応する。Tokenizer::extended は既存の ID を変えないので、末尾に新しい行を足すだけで対応が保たれる
		let src = if (name == "wte.weight" || name == "lm_head.weight") && vocab > base_vocab {
			Tensor::cat(&[&src, &var.as_tensor().narrow(0, base_vocab, vocab - base_vocab)?], 0)?
		} else {
			src
		};
		var.set(&src)
			.map_err(|e| candle_core::Error::Msg(format!("{name}: {e}")))?;
	}
	if let Some(name) = base.keys().next() {
		candle_core::bail!("土台のモデルに、このモデルの構造に無い重み {name} がある");
	}
	Ok(())
}

fn finetune(args: FinetuneArgs, dev: &Device) -> Result<()> {
	if !(0.0..1.0).contains(&args.mix_ratio) {
		candle_core::bail!("--mix-ratio は 0 以上 1 未満である必要がある: {}", args.mix_ratio);
	}
	let (base, model_cfg, base_tok) = load_tensors(&args.base, dev)?;
	let mut cfg = TrainConfig::finetune(model_cfg);
	let use_viz = args.common.apply(&mut cfg)?;

	let mut convs = chat::load_jsonl(&args.data)?;
	let val_convs = match &args.val {
		Some(p) => chat::load_jsonl(p)?,
		// 学習に使っていない会話で測らないと、丸暗記しただけでも loss が下がって見えてしまう
		None => {
			if convs.len() < 2 {
				candle_core::bail!(
					"{}: 検証に回す会話が無い（会話が 1 つだけ）。--val を指定する",
					args.data.display()
				);
			}
			let n = (convs.len() / 10).max(1);
			convs.split_off(convs.len() - n)
		}
	};
	let (text, mask) = chat::render(&convs);
	let (val_text, val_mask) = chat::render(&val_convs);
	// AI の発言が無いと loss に数える文字が 0 になり、学習も評価も成り立たない
	let val_name = args
		.val
		.as_ref()
		.map_or_else(|| "検証用の会話データ".to_string(), |p| p.display().to_string());
	for (name, m) in [(args.data.display().to_string(), &mask), (val_name, &val_mask)] {
		if !m.contains(&true) {
			candle_core::bail!("{name}: AI（assistant）の発言が 1 つも無い");
		}
	}
	if args.mix.is_none() {
		eprintln!("--mix を指定していないので、会話データだけで学習する。事前学習で得た文章力は失われていく");
	}
	let mix_text = args.mix.as_deref().map(read_text).transpose()?;
	// 区切りの印と、土台の語彙に無い文字（会話データにだけ出てくる記号など）を語彙の末尾に足す
	let markers: String = chat::MARKERS.iter().collect();
	let texts: Vec<&str> = [markers.as_str(), text.as_str(), val_text.as_str()]
		.into_iter()
		.chain(mix_text.as_deref())
		.collect();
	let tok = base_tok.extended(&texts);
	drop(texts);
	let data = tok.encode_corpus(&text)?;
	let val_data = tok.encode_corpus(&val_text)?;
	let mix_data = mix_text.as_deref().map(|t| tok.encode_corpus(t)).transpose()?;
	drop(mix_text);

	let block = cfg.model.block_size;
	ensure_len(&format!("会話データ {}", args.data.display()), data.len(), block)?;
	ensure_len("検証用の会話データ", val_data.len(), block)?;
	if let (Some(p), Some(m)) = (&args.mix, &mix_data) {
		ensure_len(&format!("文章コーパス {}", p.display()), m.len(), block)?;
	}

	let varmap = VarMap::new();
	init_from_base(&varmap, base, &cfg.model, base_tok.vocab_size(), tok.vocab_size(), dev)?;

	let chat_train = Corpus {
		ids: &data,
		mask: Some(&mask),
	};
	let chat_val = Corpus {
		ids: &val_data,
		mask: Some(&val_mask),
	};
	let mut sources = vec![(chat_train, 1.0 - args.mix_ratio)];
	if let Some(m) = &mix_data {
		sources.push((Corpus::plain(m), args.mix_ratio));
	}
	let mut rng = StdRng::seed_from_u64(args.common.seed);
	let summary = format!(
		"device: {:?} / base: {} / vocab: {} → {} / conversations: train {} + val {} / chars: train {} + val {}{}",
		dev,
		args.base.display(),
		base_tok.vocab_size(),
		tok.vocab_size(),
		convs.len(),
		val_convs.len(),
		data.len(),
		val_data.len(),
		mix_data
			.as_ref()
			.map(|m| format!(" / mix: {} chars × {}", m.len(), args.mix_ratio))
			.unwrap_or_default()
	);
	fit(
		Fit {
			cfg: &cfg,
			tok: &tok,
			varmap: &varmap,
			train: &sources,
			train_eval: chat_train,
			val: chat_val,
			out: &args.out,
			viz: use_viz.then_some((args.common.viz_every, None)),
			summary,
		},
		&mut rng,
		dev,
	)?;

	// 検証用の最初の会話の最初の User の発言に返事をさせ、学習結果の目安として表示する
	let first_user = val_convs
		.iter()
		.flat_map(|c| &c.messages)
		.find(|m| m.role == chat::Role::User);
	if args.common.sample_len > 0
		&& let Some(m) = first_user
	{
		let (best_model, _, _) = load_model(&args.out, dev)?;
		let ids = ChatIds::new(&tok, &args.out)?;
		let mut history = Vec::new();
		ids.push_user(&tok, &mut history, &m.content)?;
		let mut reply = Vec::new();
		ids.reply(
			&best_model,
			&mut history,
			args.common.sample_len,
			0.8,
			block,
			&mut rng,
			dev,
			|id| {
				reply.push(id);
				Ok(())
			},
		)?;
		println!("----\nUser: {}\nAI: {}", m.content, tok.decode(&reply));
	}
	Ok(())
}

// 会話の区切りの印の ID。印は finetune で語彙に足すので、train だけのモデルには無い
struct ChatIds {
	user: u32,
	assistant: u32,
	end: u32,
}

impl ChatIds {
	fn new(tok: &Tokenizer, model_path: &Path) -> Result<Self> {
		let [user, assistant, end] = chat::MARKERS.map(|c| tok.stoi.get(&c).copied());
		let (Some(user), Some(assistant), Some(end)) = (user, assistant, end) else {
			candle_core::bail!(
				"{}: 会話の区切りの印が語彙に無い。finetune で追加学習したモデルを指定する",
				model_path.display()
			);
		};
		Ok(Self { user, assistant, end })
	}

	// User の発言を履歴に足し、続けて AI の発言の印を置いて返事を生成できる状態にする。
	// 語彙に無い文字や区切りの印を含む発言はエラーにし、履歴は変えない
	fn push_user(&self, tok: &Tokenizer, history: &mut Vec<u32>, text: &str) -> Result<()> {
		if text.contains(chat::MARKERS) {
			candle_core::bail!("会話の区切り用の文字（U+E000〜U+E002）は入力できない");
		}
		let ids = tok.encode(text)?;
		history.push(self.user);
		history.extend(ids);
		history.extend([self.end, self.assistant]);
		Ok(())
	}

	// history の末尾に AI の返事を生成して書き足す。区切りの印のいずれかが出るか max_len 文字に達したら止め、
	// どの場合も history の末尾を終わりの印にして、次の User の発言を続けられる形にする。
	// 終わりの印以外の印（User や AI の発言の開始）は返事の中に置く位置が無いので、終わりの印に置き換える。
	// on_char は返事の文字ごとに呼ばれ、印では呼ばれない
	#[allow(clippy::too_many_arguments)]
	fn reply(
		&self,
		model: &Gpt,
		history: &mut Vec<u32>,
		max_len: usize,
		temperature: f64,
		block_size: usize,
		rng: &mut StdRng,
		dev: &Device,
		mut on_char: impl FnMut(u32) -> Result<()>,
	) -> Result<()> {
		// generate は直近 block_size 文字だけを文脈にするので、そのままでは文脈が発言の途中から始まることがある。
		// 文脈に収まる範囲で最も古い User の発言の開始位置より前を捨て、発言の境界から始まる文脈にする。
		// 収まる範囲に User の発言の開始が無い場合（最後の発言だけで block_size を超える）は捨てずに渡す
		if history.len() > block_size {
			let from = history.len() - block_size;
			if let Some(i) = history[from..].iter().position(|&id| id == self.user) {
				history.drain(..from + i);
			}
		}
		let out = generate(model, history, max_len, temperature, block_size, rng, dev, |id| {
			if self.is_marker(id) {
				return Ok(ControlFlow::Break(()));
			}
			on_char(id)?;
			Ok(ControlFlow::Continue(()))
		})?;
		*history = out;
		if history.last().is_some_and(|&id| self.is_marker(id)) {
			history.pop();
		}
		history.push(self.end);
		Ok(())
	}

	fn is_marker(&self, id: u32) -> bool {
		[self.user, self.assistant, self.end].contains(&id)
	}
}

fn io_err(e: std::io::Error) -> candle_core::Error {
	candle_core::Error::Msg(e.to_string())
}

// 区切りの印は私用領域の文字で字形が無いので、端末には User: / AI: の表記で出す
fn printable(tok: &Tokenizer, id: u32) -> String {
	let c = tok.itos[id as usize];
	chat::label(c).map_or_else(|| c.to_string(), str::to_string)
}

fn generate_cmd(args: GenerateArgs, dev: &Device) -> Result<()> {
	// temperature で logits を割るので、0 以下だと確率分布にならない
	if !(args.temperature > 0.0) {
		candle_core::bail!("temperature は 0 より大きい必要がある: {}", args.temperature);
	}
	let (model, cfg, tok) = load_model(&args.model, dev)?;
	let start = match &args.prompt {
		Some(p) => tok.encode(p)?,
		None => vec![0],
	};
	let seed = args.seed.unwrap_or_else(rand::random);
	// 出力を標準出力に流すので、同じ結果を再現するためのシードは標準エラーに出す
	if args.seed.is_none() {
		eprintln!("seed: {seed}");
	}
	let mut rng = StdRng::seed_from_u64(seed);

	let mut stdout = std::io::stdout().lock();
	for &id in &start {
		write!(stdout, "{}", printable(&tok, id)).map_err(io_err)?;
	}
	stdout.flush().map_err(io_err)?;
	generate(
		&model,
		&start,
		args.len,
		args.temperature,
		cfg.block_size,
		&mut rng,
		dev,
		|id| {
			write!(stdout, "{}", printable(&tok, id)).map_err(io_err)?;
			stdout.flush().map_err(io_err)?;
			Ok(ControlFlow::Continue(()))
		},
	)?;
	writeln!(stdout).map_err(io_err)?;
	Ok(())
}

fn chat_cmd(args: ChatArgs, dev: &Device) -> Result<()> {
	// temperature で logits を割るので、0 以下だと確率分布にならない
	if !(args.temperature > 0.0) {
		candle_core::bail!("temperature は 0 より大きい必要がある: {}", args.temperature);
	}
	let (model, cfg, tok) = load_model(&args.model, dev)?;
	let ids = ChatIds::new(&tok, &args.model)?;
	let seed = args.seed.unwrap_or_else(rand::random);
	if args.seed.is_none() {
		eprintln!("seed: {seed}");
	}
	let mut rng = StdRng::seed_from_u64(seed);
	// generate は直近の block_size 文字だけを文脈にするので、それより前の発言は返事に反映されない
	eprintln!(
		"/reset で会話を消去、/exit か空行で終了。返事の生成に使うのは直近の {} 文字まで",
		cfg.block_size
	);

	let mut stdout = std::io::stdout();
	let mut lines = std::io::stdin().lock().lines();
	let mut history = Vec::new();
	loop {
		write!(stdout, "User: ").map_err(io_err)?;
		stdout.flush().map_err(io_err)?;
		let Some(line) = lines.next() else {
			writeln!(stdout).map_err(io_err)?;
			break;
		};
		let line = line.map_err(io_err)?;
		let text = line.trim();
		match text {
			"" | "/exit" => {
				writeln!(stdout).map_err(io_err)?;
				break;
			}
			"/reset" => {
				history.clear();
				writeln!(stdout, "（会話を消去した）").map_err(io_err)?;
				continue;
			}
			_ => {}
		}
		// 入力の誤りで対話を終了させないよう、エラーを表示して次の入力を待つ
		if let Err(e) = ids.push_user(&tok, &mut history, text) {
			writeln!(stdout, "{e}").map_err(io_err)?;
			continue;
		}
		write!(stdout, "AI: ").map_err(io_err)?;
		stdout.flush().map_err(io_err)?;
		ids.reply(
			&model,
			&mut history,
			args.max_len,
			args.temperature,
			cfg.block_size,
			&mut rng,
			dev,
			|id| {
				write!(stdout, "{}", printable(&tok, id)).map_err(io_err)?;
				stdout.flush().map_err(io_err)
			},
		)?;
		writeln!(stdout).map_err(io_err)?;
	}
	Ok(())
}

fn main() -> Result<()> {
	let cli = Cli::parse();
	let dev = select_device(cli.cpu)?;
	match cli.command {
		Command::Train(args) => train(args, &dev),
		Command::Finetune(args) => finetune(args, &dev),
		Command::Generate(args) => generate_cmd(args, &dev),
		Command::Chat(args) => chat_cmd(args, &dev),
	}
}
