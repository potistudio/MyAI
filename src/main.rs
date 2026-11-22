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

use std::collections::{BTreeSet, HashMap};
use std::time::Instant;

use candle_core::{D, DType, Device, IndexOp, Result, Tensor};
use candle_nn::{
	AdamW, Embedding, Init, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap, embedding, linear,
	linear_no_bias, loss,
};
use rand::distr::{Distribution, weighted::WeightedIndex};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

struct Config {
	block_size: usize,
	batch_size: usize,
	n_layer: usize,
	n_head: usize,
	n_embd: usize,
	lr: f64,
	max_steps: usize,
	eval_interval: usize,
	eval_iters: usize,
}

impl Config {
	// nanoGPT の文字単位 Shakespeare 設定と同規模にして、既知の loss（val 1.47 前後）と突き合わせられるようにする
	fn gpu() -> Self {
		Self {
			block_size: 256,
			batch_size: 64,
			n_layer: 6,
			n_head: 6,
			n_embd: 384,
			lr: 1e-3,
			max_steps: 5000,
			eval_interval: 250,
			eval_iters: 50,
		}
	}

	// 配線ミスを GPU 無しでも数十秒で見つけたいので、CPU で回る大きさにする
	fn tiny() -> Self {
		Self {
			block_size: 32,
			batch_size: 16,
			n_layer: 2,
			n_head: 2,
			n_embd: 64,
			lr: 3e-3,
			max_steps: 1000,
			eval_interval: 200,
			eval_iters: 10,
		}
	}
}

// ニューラルネットは数値しか扱えないので、文字を整数 ID に置き換える必要がある。
// まず「トークン = 1 文字」から始めて、BPE を入れたときの差を後で比較できるようにする
struct Tokenizer {
	itos: Vec<char>,
	stoi: HashMap<char, u32>,
}

impl Tokenizer {
	fn from_text(text: &str) -> Self {
		// ID の割り当てを実行ごとに変えたくないので、ソート済みの集合から作る
		let itos: Vec<char> = text.chars().collect::<BTreeSet<_>>().into_iter().collect();
		let stoi = itos.iter().enumerate().map(|(i, &c)| (c, i as u32)).collect();
		Self { itos, stoi }
	}

	fn vocab_size(&self) -> usize {
		self.itos.len()
	}

	fn encode(&self, s: &str) -> Vec<u32> {
		s.chars().map(|c| self.stoi[&c]).collect()
	}

	fn decode(&self, ids: &[u32]) -> String {
		ids.iter().map(|&i| self.itos[i as usize]).collect()
	}
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
}

impl CausalSelfAttention {
	fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
		// Q, K, V を 3 回の matmul に分けるより 1 回にまとめた方が GPU を効率よく使える
		let c_attn = linear(cfg.n_embd, 3 * cfg.n_embd, vb.pp("c_attn"))?;
		let c_proj = linear(cfg.n_embd, cfg.n_embd, vb.pp("c_proj"))?;
		Ok(Self {
			c_attn,
			c_proj,
			n_head: cfg.n_head,
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
		let att = softmax(&att)?;

		// 各ヘッドが集めた情報を連結し、c_proj で混ぜてから残差に戻す
		let y = att.matmul(&v)?.transpose(1, 2)?.reshape((b, t, c))?;
		self.c_proj.forward(&y)
	}
}

// attention は位置どうしで情報を混ぜるだけなので、集めた情報を加工する処理を位置ごとに独立して入れる。
// 一度 4 倍の次元に広げて GELU で非線形を挟み、元の次元に戻す。学習した知識の多くはここに入るとされる
struct Mlp {
	fc: Linear,
	proj: Linear,
}

impl Mlp {
	fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
		Ok(Self {
			fc: linear(cfg.n_embd, 4 * cfg.n_embd, vb.pp("fc"))?,
			proj: linear(4 * cfg.n_embd, cfg.n_embd, vb.pp("proj"))?,
		})
	}

	fn forward(&self, x: &Tensor) -> Result<Tensor> {
		self.proj.forward(&self.fc.forward(x)?.gelu()?)
	}
}

struct Block {
	ln1: LayerNorm,
	attn: CausalSelfAttention,
	ln2: LayerNorm,
	mlp: Mlp,
}

impl Block {
	fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
		Ok(Self {
			ln1: LayerNorm::new(cfg.n_embd, vb.pp("ln1"))?,
			attn: CausalSelfAttention::new(cfg, vb.pp("attn"))?,
			ln2: LayerNorm::new(cfg.n_embd, vb.pp("ln2"))?,
			mlp: Mlp::new(cfg, vb.pp("mlp"))?,
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
}

impl Gpt {
	fn new(cfg: &Config, vocab_size: usize, vb: VarBuilder) -> Result<Self> {
		let blocks = (0..cfg.n_layer)
			.map(|i| Block::new(cfg, vb.pp(format!("h.{i}"))))
			.collect::<Result<_>>()?;
		Ok(Self {
			// ID のままだと「3 番と 4 番は近い」という意味のない大小関係がつくので、ID ごとに学習可能なベクトルを割り当てる。
			// candle の初期値は標準偏差 1 と大きめで、step 0 の loss が一様分布の ln(vocab) より高く出る（初期化の実験ネタ）
			wte: embedding(vocab_size, cfg.n_embd, vb.pp("wte"))?,
			// attention は入力を順序のない集合として扱うので、「何文字目か」を位置ごとのベクトルとして足して教える。
			// RoPE と比較するための基準として、最初は GPT-2 と同じ学習型の位置埋め込みにする
			wpe: embedding(cfg.block_size, cfg.n_embd, vb.pp("wpe"))?,
			blocks,
			ln_f: LayerNorm::new(cfg.n_embd, vb.pp("ln_f"))?,
			lm_head: linear_no_bias(cfg.n_embd, vocab_size, vb.pp("lm_head"))?,
		})
	}

	fn forward(&self, idx: &Tensor) -> Result<Tensor> {
		let (_b, t) = idx.dims2()?;
		let dev = idx.device();
		let pos = Tensor::arange(0u32, t as u32, dev)?;
		let mut x = self.wte.forward(idx)?.broadcast_add(&self.wpe.forward(&pos)?)?;
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

fn get_batch(data: &[u32], cfg: &Config, rng: &mut StdRng, dev: &Device) -> Result<(Tensor, Tensor)> {
	let (b, t) = (cfg.batch_size, cfg.block_size);
	let mut xs = Vec::with_capacity(b * t);
	let mut ys = Vec::with_capacity(b * t);
	for _ in 0..b {
		let i = rng.random_range(0..data.len() - t - 1);
		xs.extend_from_slice(&data[i..i + t]);
		// 全位置で同時に「次の 1 トークン」を予測させるため、入力を 1 つずらしたものを正解にする。
		// causal mask があるので、長さ T の 1 本から「1 文字目まで → 2 文字目」…「T 文字目まで → T+1 文字目」の T 問を 1 回で解かせられる
		ys.extend_from_slice(&data[i + 1..i + t + 1]);
	}
	Ok((Tensor::from_vec(xs, (b, t), dev)?, Tensor::from_vec(ys, (b, t), dev)?))
}

// 「正解の文字につけた確率の -log」（cross entropy）を最小化したい。正解に確率 1 を出せば 0 になる。
// 全文字を当てずっぽうにすると ln(vocab) で、65 文字なら 4.17。loss 2.0 は e^2 ≒ 7 択まで絞れている状態
fn compute_loss(model: &Gpt, x: &Tensor, y: &Tensor) -> Result<Tensor> {
	let logits = model.forward(x)?;
	let (b, t, v) = logits.dims3()?;
	loss::cross_entropy(&logits.reshape((b * t, v))?, &y.reshape(b * t)?)
}

// 1 バッチの loss はノイズが大きく改良の効果と見分けにくいので、複数バッチの平均で比べる
fn estimate_loss(model: &Gpt, data: &[u32], cfg: &Config, rng: &mut StdRng, dev: &Device) -> Result<f32> {
	let mut total = 0.0;
	for _ in 0..cfg.eval_iters {
		let (x, y) = get_batch(data, cfg, rng, dev)?;
		total += compute_loss(model, &x, &y)?.to_scalar::<f32>()?;
	}
	Ok(total / cfg.eval_iters as f32)
}

fn generate(
	model: &Gpt,
	start: &[u32],
	max_new: usize,
	temperature: f64,
	cfg: &Config,
	rng: &mut StdRng,
	dev: &Device,
) -> Result<Vec<u32>> {
	let mut ids = start.to_vec();
	for _ in 0..max_new {
		// 仕組みを追いやすくするため毎回全文脈を計算し直す。KV キャッシュは推論側を作るときに入れる
		let ctx = &ids[ids.len().saturating_sub(cfg.block_size)..];
		let logits = model.forward(&Tensor::new(ctx, dev)?.unsqueeze(0)?)?;
		// 次の文字の予測に使うのは最後の位置の出力だけ。
		// temperature で割ってから softmax にかけ、1 未満なら確率の高い文字に寄せ、1 より大きければばらつかせる
		let last = (logits.i((0, ctx.len() - 1))? / temperature)?;
		let probs = softmax(&last)?.to_vec1::<f32>()?;
		let dist = WeightedIndex::new(&probs).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
		ids.push(dist.sample(rng) as u32);
	}
	Ok(ids)
}

fn main() -> Result<()> {
	let args: Vec<String> = std::env::args().skip(1).collect();
	let tiny = args.iter().any(|a| a == "--tiny");
	let path = args
		.iter()
		.find(|a| !a.starts_with("--"))
		.map(String::as_str)
		.unwrap_or("input.txt");
	let cfg = if tiny { Config::tiny() } else { Config::gpu() };

	let dev = Device::cuda_if_available(0)?;
	let text = std::fs::read_to_string(path).map_err(|e| candle_core::Error::Msg(format!("{path}: {e}")))?;
	let tok = Tokenizer::from_text(&text);
	let data = tok.encode(&text);
	// 学習に使っていない部分で測らないと、丸暗記しただけでも loss が下がって見えてしまう
	let n_train = data.len() * 9 / 10;
	let (train, val) = data.split_at(n_train);

	let varmap = VarMap::new();
	let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
	let model = Gpt::new(&cfg, tok.vocab_size(), vb)?;
	let n_params: usize = varmap.all_vars().iter().map(|v| v.elem_count()).sum();
	println!(
		"device: {:?} / chars: {} / vocab: {} / params: {:.2}M",
		dev,
		data.len(),
		tok.vocab_size(),
		n_params as f64 / 1e6
	);

	let mut opt = AdamW::new(
		varmap.all_vars(),
		ParamsAdamW {
			lr: cfg.lr,
			..Default::default()
		},
	)?;
	let mut rng = StdRng::seed_from_u64(1337);
	let t0 = Instant::now();

	for step in 0..=cfg.max_steps {
		if step % cfg.eval_interval == 0 || step == cfg.max_steps {
			let tr = estimate_loss(&model, train, &cfg, &mut rng, &dev)?;
			let va = estimate_loss(&model, val, &cfg, &mut rng, &dev)?;
			println!(
				"step {step:5} | train {tr:.4} | val {va:.4} | {:.1}s",
				t0.elapsed().as_secs_f32()
			);
		}
		if step == cfg.max_steps {
			break;
		}
		let (x, y) = get_batch(train, &cfg, &mut rng, &dev)?;
		let loss = compute_loss(&model, &x, &y)?;
		// loss を下げるには各重みをどちらにどれだけ動かせばいいかを、逆伝播で全パラメータ分まとめて求める
		let grads = loss.backward()?;
		// 部品を差し替えたときに勾配がエラーも出さず途切れる事故を、最初の 1 step で検出したい。
		// ただし同じ重みに別経路から勾配が届く場合（Q/K 側だけ切れて V 側は生きている等）は素通りするので、
		// 怪しいときは loss の下がり方で確認する
		if step == 0 {
			let missing = varmap
				.all_vars()
				.iter()
				.filter(|v| grads.get(v.as_tensor()).is_none())
				.count();
			if missing > 0 {
				candle_core::bail!("{missing} 個のパラメータに勾配が届いていない");
			}
		}
		// 勾配そのままだと重みごとにスケールがばらばらなので、AdamW で過去の勾配の平均と大きさを使って歩幅を揃えて更新する
		opt.step(&grads)?;
	}

	varmap.save("model.safetensors")?;

	let out = generate(&model, &data[..1], 500, 0.8, &cfg, &mut rng, &dev)?;
	println!("----\n{}", tok.decode(&out));
	Ok(())
}
