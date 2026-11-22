//! 学習中のモデルが検証データの各文字に付けた確率を、一定 step ごとにターミナルの同じ位置へ色で描き直す。
//!
//! 描画には学習ループの評価用モデル（重みを共有し、dropout 0）を使うので、更新された重みの効果がそのまま表示に出る。
//! 入力には検証データの区間を使い、丸暗記した文字列で予測が当たって見える状態を避ける。

use std::io::Write as _;

use candle_core::{Device, Result, Tensor};

use crate::{Gpt, Tokenizer, softmax};

// 描画ごとに表示する評価ログの行数。全行を出すとフレームが端末の高さを超え、カーソルを先頭に戻す描き直しが崩れる
const LOG_TAIL: usize = 5;

pub struct Viz<'a> {
	tok: &'a Tokenizer,
	block_size: usize,
	// 行頭から始まる検証データの区間。毎回同じ区間を入力し、変化の原因を重みの更新だけにする
	sample: Vec<u32>,
	log: Vec<String>,
}

impl<'a> Viz<'a> {
	pub fn new(tok: &'a Tokenizer, block_size: usize, val: &[u16]) -> Result<Self> {
		// Windows の従来コンソールは既定でエスケープシーケンスを解釈しないので、仮想端末処理を有効にする。
		// 有効にできない端末でも学習自体は続けられるので、結果は確認しない
		let _ = anstyle_query::windows::enable_ansi_colors();
		// 区間が単語の途中から始まると文脈表示が読みにくいので、最初の改行の直後から使う。
		// ただし get_batch と同じく block_size + 1 文字を確保できない場合は先頭から使う
		let start = tok
			.stoi
			.get(&'\n')
			.and_then(|&nl| val.iter().position(|&id| id as u32 == nl))
			.map(|i| i + 1)
			.filter(|&i| val.len() - i > block_size + 1)
			.unwrap_or(0);
		print!("\x1b[2J");
		Ok(Self {
			tok,
			block_size,
			// 各描画が読むのは先頭から最大 block_size + 256 文字なので、検証データ全体ではなく余裕を持たせた分だけ u32 に写す
			sample: val[start..val.len().min(start + 2 * block_size + 512)]
				.iter()
				.map(|&id| id as u32)
				.collect(),
			log: Vec::new(),
		})
	}

	pub fn push_log(&mut self, line: String) {
		self.log.push(line);
	}

	pub fn log(&self) -> &[String] {
		&self.log
	}

	pub fn draw(&self, model: &Gpt, progress: &str, dev: &Device) -> Result<()> {
		let body = self.heat(model, dev)?;
		let mut frame = String::from("\x1b[H");
		let tail = &self.log[self.log.len().saturating_sub(LOG_TAIL)..];
		for line in tail
			.iter()
			.map(String::as_str)
			.chain([progress, ""])
			.chain(body.iter().map(String::as_str))
		{
			// 前回のフレームの方が長い行の残りを消すため、行末までを消去してから改行する
			frame.push_str(line);
			frame.push_str("\x1b[K\n");
		}
		// 前回のフレームの方が行数が多い場合の残りを消す
		frame.push_str("\x1b[J");
		let mut out = std::io::stdout().lock();
		out.write_all(frame.as_bytes())
			.and_then(|_| out.flush())
			.map_err(|e| candle_core::Error::Msg(e.to_string()))
	}

	// 各文字について、直前までの文脈から正解のその文字に付けた確率で背景色を塗る。
	// 学習が進むと、空白と改行、頻出する単語、単語の途中の文字の順に明るくなる
	fn heat(&self, model: &Gpt, dev: &Device) -> Result<Vec<String>> {
		const SHOWN: usize = 256;

		// モデルは block_size 文字までしか文脈に取れないので、表示区間を block_size 文字ずつの窓に分けて 1 バッチで流す。
		// 各窓の先頭付近は文脈が短く、窓の終わりより予測が外れやすい
		let t = self.block_size.min(self.sample.len() - 1);
		let windows = SHOWN.div_ceil(t).min((self.sample.len() - 1) / t);
		let n = windows * t;
		let x = Tensor::new(&self.sample[..n], dev)?.reshape((windows, t))?;
		let y = Tensor::new(&self.sample[1..=n], dev)?.reshape((n, 1))?;
		let logits = model.forward(&x)?;
		let probs = softmax(&logits.reshape((n, ()))?)?
			.gather(&y, 1)?
			.flatten_all()?
			.to_vec1::<f32>()?;
		let loss = probs.iter().map(|p| -p.ln()).sum::<f32>() / n as f32;
		let hits = probs.iter().filter(|&&p| p >= 0.5).count();

		let mut lines = vec![
			format!(
				"検証データ {n} 文字の予測  loss {loss:.3} / 確率 50% 以上を付けた文字 {:.1}%",
				hits as f32 * 100.0 / n as f32
			),
			format!(
				"正解の文字に付けた確率: 0% {} 100%",
				(0..=10).map(|i| paint(' ', i as f32 / 10.0)).collect::<String>()
			),
			String::new(),
		];
		// 先頭の 1 文字は予測の対象ではないので塗らない
		let mut line = visible_cell(self.tok.itos[self.sample[0] as usize]).to_string();
		for (&id, &p) in self.sample[1..=n].iter().zip(&probs) {
			let c = self.tok.itos[id as usize];
			line.push_str(&paint(visible_cell(c), p));
			if c == '\n' {
				lines.push(std::mem::take(&mut line));
			}
		}
		lines.push(line);
		Ok(lines)
	}
}

// 背景色で塗るので、空白は空白のままでも位置が見える。改行だけ記号に置き換える
fn visible_cell(c: char) -> char {
	if c == '\n' { '↵' } else { c }
}

// 確率 0 を暗い灰色、1 を明るい黄色とし、その間を線形に補間した 24 bit カラーの背景で文字を描く。
// 背景が明るい側では白文字が読めないので、文字色を黒に切り替える
fn paint(c: char, p: f32) -> String {
	const LOW: [f32; 3] = [40.0, 40.0, 52.0];
	const HIGH: [f32; 3] = [255.0, 214.0, 10.0];
	let p = p.clamp(0.0, 1.0);
	let [r, g, b] = [0, 1, 2].map(|i| (LOW[i] + (HIGH[i] - LOW[i]) * p) as u8);
	let fg = if p > 0.5 { 30 } else { 37 };
	format!("\x1b[{fg};48;2;{r};{g};{b}m{c}\x1b[0m")
}
