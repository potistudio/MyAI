//! 対話形式の学習データと、会話を 1 本の文字列に並べる書式。
//!
//! 会話は発言ごとに「役割の印 + 本文 + 終わりの印」を並べた 1 本の文字列にする:
//!   [USER]Hi![END][ASSISTANT]Hello! How can I help you today?[END][USER]...
//! 印は Unicode の私用領域の文字で、通常の文章には現れない。
//! そのため 1 組の重みのまま、印を含む入力には AI の発言として返事を書かせられる。
//! 印を含まない入力で事前学習と同じ文章の続きを書く力は、finetune で文章コーパスを混ぜた（--mix）場合に保たれる。
//! chat サブコマンドは [ASSISTANT] の後ろを生成し、[END] が出た時点で生成を止める。

use std::path::Path;

use candle_core::Result;
use serde::Deserialize;

pub const USER: char = '\u{E000}';
pub const ASSISTANT: char = '\u{E001}';
pub const END: char = '\u{E002}';
pub const MARKERS: [char; 3] = [USER, ASSISTANT, END];

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
	fn marker(self) -> char {
		match self {
			Role::User => USER,
			Role::Assistant => ASSISTANT,
		}
	}
}

/// 1 行 1 会話の JSONL を読む。形式は {"messages": [{"role": "user" | "assistant", "content": "..."}, ...]}。
/// 本文に印の文字があると発言の境界が本文の途中に生じるので、その会話を含むファイルはエラーにする
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
		if conv.messages.iter().any(|m| m.content.contains(MARKERS)) {
			candle_core::bail!("{}: 本文に会話の区切り用の文字（U+E000〜U+E002）が含まれる", at());
		}
		convs.push(conv);
	}
	if convs.is_empty() {
		candle_core::bail!("{}: 会話が 1 つも無い", path.display());
	}
	Ok(convs)
}

/// 会話を並べた 1 本の文字列と、各文字を loss に数えるかどうかの列（文字数と同じ長さ）を返す。
/// 数えるのは AI の発言の本文とその直後の [END]。[END] を数えることで、返事を終える位置も学習させる。
/// User の発言と役割の印は文脈として入力するだけで、loss には数えない。
/// 数えると、chat の推論時には入力として与えられる User の発言を生成する重みの更新に、学習の一部が使われる
pub fn render(convs: &[Conversation]) -> (String, Vec<bool>) {
	let mut text = String::new();
	let mut mask = Vec::new();
	let mut push = |c: char, counted: bool| {
		text.push(c);
		mask.push(counted);
	};
	for conv in convs {
		for m in &conv.messages {
			let counted = m.role == Role::Assistant;
			push(m.role.marker(), false);
			for c in m.content.chars() {
				push(c, counted);
			}
			push(END, counted);
		}
	}
	(text, mask)
}

/// 端末に出力するときの印の表記。私用領域の文字は多くのフォントで字形が無く、表示されないか豆腐になる
pub fn label(c: char) -> Option<&'static str> {
	match c {
		USER => Some("\nUser: "),
		ASSISTANT => Some("\nAI: "),
		END => Some("\n"),
		_ => None,
	}
}

/// 1 文字を 1 マスに描く表示（viz）での印の表記
pub fn symbol(c: char) -> Option<char> {
	match c {
		USER => Some('▷'),
		ASSISTANT => Some('▶'),
		END => Some('■'),
		_ => None,
	}
}
