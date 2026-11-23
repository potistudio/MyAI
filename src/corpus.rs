//! 学習データのファイルの読み書き。
//!
//! 入力は次の 3 形式で、拡張子で区別する:
//!   .bin    tokenizer encode が書き出した ID 列。ヘッダーのトークナイザーの fingerprint が一致しない場合は読み込まない
//!   .jsonl  1 行 1 文書の {"text": "..."}。本文が空白だけの文書は捨てる
//!   その他  プレーンテキスト。`<|endoftext|>` の文字列で文書に区切る（TinyStories の形式）
//!
//! .bin の形式（数値はすべてリトルエンディアン）:
//!   0..8    識別子 b"MYAITOK\0"
//!   8..12   形式の版（u32）
//!   12..16  語彙数（u32）
//!   16..24  トークナイザーの fingerprint（u64）
//!   24..32  トークン数（u64）
//!   32..    ID 列（u16 × トークン数）

use std::io::{Read as _, Write as _};
use std::path::Path;

use candle_core::Result;
use serde::Deserialize;

use crate::tokenizer::Tokenizer;

const MAGIC: &[u8; 8] = b"MYAITOK\0";
const BIN_VERSION: u32 = 1;
const HEADER_LEN: usize = 32;
const DOC_SEPARATOR: &str = "<|endoftext|>";

/// 文書の本文を 1 本の文字列に並べ、文書ごとの範囲を持つ。文書ごとに String を確保しないので、
/// 数 GB のファイルでもメモリはファイルの大きさ程度で済む
pub struct Documents {
	text: String,
	spans: Vec<(usize, usize)>,
}

impl Documents {
	pub fn strs(&self) -> Vec<&str> {
		self.spans.iter().map(|&(s, e)| &self.text[s..e]).collect()
	}

	pub fn len(&self) -> usize {
		self.spans.len()
	}

	/// 全文書の本文のバイト数の合計
	pub fn bytes(&self) -> usize {
		self.spans.iter().map(|&(s, e)| e - s).sum()
	}

	/// 先頭から、本文のバイト数の合計が limit を超えない範囲の文書だけを残す。1 つ目の文書は limit を超えても残す。
	/// 残らなかった文書の本文はメモリから解放する
	pub fn truncate_bytes(&mut self, limit: usize) {
		let mut acc = 0;
		let keep = self
			.spans
			.iter()
			.position(|&(s, e)| {
				acc += e - s;
				acc > limit
			})
			.map_or(self.spans.len(), |i| i.max(1));
		debug_assert!(
			self.spans.windows(2).all(|w| w[0].1 <= w[1].0),
			"spans は text の中で前から順に並ぶ"
		);
		self.spans.truncate(keep);
		// spans は text の中で前から順に並ぶので、最後に残った文書の終わりより後ろは捨ててよい
		let end = self.spans.last().map_or(0, |&(_, e)| e);
		self.text.truncate(end);
		self.text.shrink_to_fit();
	}
}

#[derive(Deserialize)]
struct JsonlDoc {
	text: String,
}

fn read_err(path: &Path, e: impl std::fmt::Display) -> candle_core::Error {
	candle_core::Error::Msg(format!("{}: {e}", path.display()))
}

fn is_ext(path: &Path, ext: &str) -> bool {
	path.extension().is_some_and(|e| e.eq_ignore_ascii_case(ext))
}

/// .jsonl かプレーンテキストを文書の列として読む。
/// プレーンテキストの文書は区切りの前後の空白と改行を取り除き、空になった文書は捨てる
pub fn load_documents(path: &Path) -> Result<Documents> {
	if is_ext(path, "bin") {
		candle_core::bail!("{}: ID 列のファイルはテキストとして読めない", path.display());
	}
	let raw = std::fs::read_to_string(path).map_err(|e| read_err(path, e))?;
	if !is_ext(path, "jsonl") {
		let base = raw.as_ptr() as usize;
		let spans = raw
			.split(DOC_SEPARATOR)
			.map(str::trim)
			.filter(|d| !d.is_empty())
			.map(|d| {
				let s = d.as_ptr() as usize - base;
				(s, s + d.len())
			})
			.collect();
		return Ok(Documents { text: raw, spans });
	}
	let mut text = String::new();
	let mut spans = Vec::new();
	for (i, line) in raw.lines().enumerate() {
		if line.trim().is_empty() {
			continue;
		}
		let doc: JsonlDoc = serde_json::from_str(line).map_err(|e| read_err(path, format!("{} 行目: {e}", i + 1)))?;
		// プレーンテキストと同じく空の文書は捨て、文書の区切りが 2 つ続く並びを学習データに入れない
		if doc.text.trim().is_empty() {
			continue;
		}
		let s = text.len();
		text.push_str(&doc.text);
		spans.push((s, text.len()));
	}
	Ok(Documents { text, spans })
}

/// 学習用の ID 列を読む。.bin はそのまま読み、テキストは tok で encode する
pub fn load_tokens(path: &Path, tok: &Tokenizer) -> Result<Vec<u16>> {
	if is_ext(path, "bin") {
		read_bin(path, tok)
	} else {
		Ok(tok.encode_documents(&load_documents(path)?.strs()))
	}
}

pub fn write_bin(path: &Path, tok: &Tokenizer, ids: &[u16]) -> Result<()> {
	let file = std::fs::File::create(path).map_err(|e| read_err(path, e))?;
	let mut w = std::io::BufWriter::new(file);
	let mut header = Vec::with_capacity(HEADER_LEN);
	header.extend_from_slice(MAGIC);
	header.extend_from_slice(&BIN_VERSION.to_le_bytes());
	header.extend_from_slice(&(tok.vocab_size() as u32).to_le_bytes());
	header.extend_from_slice(&tok.fingerprint().to_le_bytes());
	header.extend_from_slice(&(ids.len() as u64).to_le_bytes());
	w.write_all(&header).map_err(|e| read_err(path, e))?;
	// 1 回の write に渡すバイト列を一定の大きさで区切り、ID 列全体のバイト列を別に確保しない
	let mut buf = Vec::with_capacity(1 << 20);
	for chunk in ids.chunks(1 << 19) {
		buf.clear();
		for &id in chunk {
			buf.extend_from_slice(&id.to_le_bytes());
		}
		w.write_all(&buf).map_err(|e| read_err(path, e))?;
	}
	w.flush().map_err(|e| read_err(path, e))
}

// ID 列は読みながら u16 に変換し、ファイル全体のバイト列と ID 列を同時にメモリに持たない
pub fn read_bin(path: &Path, tok: &Tokenizer) -> Result<Vec<u16>> {
	let mut file = std::fs::File::open(path).map_err(|e| read_err(path, e))?;
	let not_bin = || {
		candle_core::Error::Msg(format!(
			"{}: tokenizer encode で書き出した ID 列のファイルではない",
			path.display()
		))
	};
	let mut header = [0u8; HEADER_LEN];
	file.read_exact(&mut header).map_err(|_| not_bin())?;
	if &header[..8] != MAGIC {
		return Err(not_bin());
	}
	let u32_at = |i: usize| u32::from_le_bytes(header[i..i + 4].try_into().expect("4 バイト"));
	let u64_at = |i: usize| u64::from_le_bytes(header[i..i + 8].try_into().expect("8 バイト"));
	let version = u32_at(8);
	if version != BIN_VERSION {
		candle_core::bail!(
			"{}: 形式の版 {version} には対応していない（対応は {BIN_VERSION}）",
			path.display()
		);
	}
	// 別のトークナイザーで作った ID 列は、同じ ID が別のトークンを指すので、学習を始める前に弾く
	if u64_at(16) != tok.fingerprint() || u32_at(12) as usize != tok.vocab_size() {
		candle_core::bail!(
			"{}: 指定したトークナイザーとは別のトークナイザーで作られている。同じ tokenizer.json で encode し直す",
			path.display()
		);
	}
	let n = u64_at(24);
	// FIFO などの通常のファイルでないものはヘッダーを読めても大きさが 0 になるので、引き算の桁あふれもファイルの誤りとして扱う
	let body_len = file
		.metadata()
		.map_err(|e| read_err(path, e))?
		.len()
		.checked_sub(HEADER_LEN as u64)
		.ok_or_else(not_bin)?;
	// 壊れたヘッダーの n では 2 倍が桁あふれして長さの検査を素通りしうるので、桁あふれも不一致として扱う
	if n.checked_mul(2) != Some(body_len) {
		candle_core::bail!(
			"{}: ヘッダーのトークン数 {n} と本体の大きさ {body_len} バイトが合わない",
			path.display()
		);
	}
	let mut ids = Vec::with_capacity(n as usize);
	let mut buf = vec![0u8; 1 << 20];
	let mut left = body_len as usize;
	while left > 0 {
		let chunk = &mut buf[..left.min(1 << 20)];
		file.read_exact(chunk).map_err(|e| read_err(path, e))?;
		ids.extend(chunk.as_chunks::<2>().0.iter().map(|&b| u16::from_le_bytes(b)));
		left -= chunk.len();
	}
	debug_assert_eq!(ids.len() as u64, n);
	Ok(ids)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::tokenizer::count_pieces;

	fn scratch(name: &str) -> std::path::PathBuf {
		std::env::temp_dir().join(format!("my-ai-test-{}-{name}", std::process::id()))
	}

	#[test]
	fn text_splits_on_separator() {
		let p = scratch("docs.txt");
		std::fs::write(&p, "first story.\n<|endoftext|>\n\n second\n<|endoftext|>\n").unwrap();
		let docs = load_documents(&p).unwrap();
		assert_eq!(docs.strs(), ["first story.", "second"]);
		std::fs::remove_file(&p).unwrap();
	}

	#[test]
	fn jsonl_reads_text_field() {
		let p = scratch("docs.jsonl");
		std::fs::write(
			&p,
			"{\"text\": \"a b\"}\n\n{\"text\": \" \"}\n{\"text\": \"c\", \"id\": 1}\n",
		)
		.unwrap();
		let mut docs = load_documents(&p).unwrap();
		assert_eq!(docs.strs(), ["a b", "c"]);
		docs.truncate_bytes(1);
		assert_eq!(docs.strs(), ["a b"]);
		assert_eq!(docs.text, "a b");
		std::fs::remove_file(&p).unwrap();
	}

	#[test]
	fn bin_roundtrip_and_fingerprint_check() {
		let docs = ["hello hello world"];
		let tok = Tokenizer::train(&count_pieces(&docs), 300, |_, _| {}).unwrap();
		let other = Tokenizer::train(&count_pieces(&["other text"]), 300, |_, _| {}).unwrap();
		let ids = tok.encode_documents(&docs);
		let p = scratch("ids.bin");
		write_bin(&p, &tok, &ids).unwrap();
		assert_eq!(read_bin(&p, &tok).unwrap(), ids);
		assert!(read_bin(&p, &other).is_err());
		std::fs::remove_file(&p).unwrap();
	}
}
