//! 対話形式の学習データと、会話を 1 本の ID 列に並べる書式。
//!
//! 会話は発言ごとに「役割の特殊トークン + 本文 + 発言の終わりの特殊トークン」を並べた 1 本の ID 列にする:
//!   <|user|>Hi!<|end|><|assistant|>Hello! How can I help you today?<|end|><|user|>...
//! 特殊トークンはテキストの encode からは生成されないので、本文や利用者の入力が発言の境界を作ることはない。
//! そのため 1 組の重みのまま、特殊トークンを含む入力には AI の発言として返事を書かせられる。
//! 特殊トークンを含まない入力で事前学習と同じ文章の続きを書く力は、finetune で文章コーパスを混ぜた（--mix）場合に保たれる。
//! chat サブコマンドは <|assistant|> の後ろを生成し、<|end|> が出た時点で生成を止める。

use std::path::Path;

use candle_core::Result;
use serde::Deserialize;

use crate::tokenizer::{Special, Tokenizer};

#[derive(Deserialize)]
pub struct Conversation {
	pub messages: Vec<Message>,
}

#[derive(Deserialize)]
pub struct Message {
	pub role: Role,
	pub content: String,
}

#[derive(Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
	User,
	Assistant,
}

impl Role {
	fn special(self) -> Special {
		match self {
			Role::User => Special::User,
			Role::Assistant => Special::Assistant,
		}
	}
}

/// 1 行 1 会話の JSONL を読む。形式は {"messages": [{"role": "user" | "assistant", "content": "..."}, ...]}
pub fn load_jsonl(path: &Path) -> Result<Vec<Conversation>> {
	let text =
		std::fs::read_to_string(path).map_err(|e| candle_core::Error::Msg(format!("{}: {e}", path.display())))?;
	let mut convs = Vec::new();
	for (i, line) in text.lines().enumerate() {
		if line.trim().is_empty() {
			continue;
		}
		let at = || format!("{}:{}", path.display(), i + 1);
		let conv: Conversation =
			serde_json::from_str(line).map_err(|e| candle_core::Error::Msg(format!("{}: {e}", at())))?;
		if conv.messages.is_empty() {
			candle_core::bail!("{}: messages が空", at());
		}
		convs.push(conv);
	}
	if convs.is_empty() {
		candle_core::bail!("{}: 会話が 1 つも無い", path.display());
	}
	Ok(convs)
}

/// 会話を並べた 1 本の ID 列と、各トークンを loss に数えるかどうかの列（ID 列と同じ長さ）を返す。
/// 数えるのは AI の発言の本文とその直後の <|end|>。<|end|> を数えることで、返事を終える位置も学習させる。
/// User の発言と役割の特殊トークンは文脈として入力するだけで、loss には数えない。
/// 数えると、chat の推論時には入力として与えられる User の発言を生成する重みの更新に、学習の一部が使われる
pub fn render(tok: &Tokenizer, convs: &[Conversation]) -> (Vec<u16>, Vec<bool>) {
	let mut ids = Vec::new();
	let mut mask = Vec::new();
	// 語彙は u16 に収まる大きさに制限されているので、ID は u16 に変換できる
	let mut push = |id: u32, counted: bool| {
		ids.push(id as u16);
		mask.push(counted);
	};
	let end = tok.special(Special::End);
	for conv in convs {
		for m in &conv.messages {
			let counted = m.role == Role::Assistant;
			push(tok.special(m.role.special()), false);
			for id in tok.encode(&m.content) {
				push(id, counted);
			}
			push(end, counted);
		}
	}
	(ids, mask)
}

/// 端末に出力するときの名前付きの特殊トークンの表記。index は特殊トークンの枠の中での番号で、予備の枠は None
pub fn label(index: usize) -> Option<&'static str> {
	Some(match Special::from_index(index)? {
		Special::User => "\nUser: ",
		Special::Assistant => "\nAI: ",
		Special::End => "\n",
		Special::EndOfText => "\n<|endoftext|>\n",
	})
}

/// 1 文字を 1 マスに描く表示（viz）での特殊トークンの表記
pub fn symbol(index: usize) -> char {
	match Special::from_index(index) {
		Some(Special::User) => '▷',
		Some(Special::Assistant) => '▶',
		Some(Special::End) => '■',
		Some(Special::EndOfText) => '¶',
		None => '◇',
	}
}
