//! バイトレベル BPE のトークナイザー。
//!
//! ID の割り当て:
//!   0〜255                 UTF-8 のバイト 1 個
//!   256〜(256 + マージ数 - 1) 学習したマージで作るトークン。ID 256 + i は i 番目に学習したマージに対応する
//!   その後ろの N_SPECIAL 個   特殊トークン。テキストの encode からは生成されず、呼び出し側が ID を直接置く
//!
//! encode は文字列を pretokenize で塊に分け、塊ごとにバイト列へ学習順の早いマージから順に適用する。
//! マージは塊の境界をまたがないので、同じ塊は前後の文脈によらず同じ ID 列になる。CorpusEncoder のキャッシュはこの性質に依存する。
//! バイトから始めるので、どんな文字列も語彙に無い文字を出さずに encode できる。

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::path::Path;

use candle_core::Result;
use serde::{Deserialize, Serialize};

/// 前処理の規則の版。規則を変えると同じ文字列でも塊の切り方が変わり、学習済みのマージ表と合わなくなるので、
/// tokenizer.json に記録し、読み込み時に一致を検査する
pub const PRETOKENIZER: &str = "en-v1";
const FORMAT_VERSION: u32 = 1;

// encode_piece が全体を走査する方法から差分更新の方法に切り替える塊のバイト数。
// 英語の単語はほぼこれ以下で、短い塊では全体を走査する方法の方がヒープを確保しない分速い
const LONG_PIECE: usize = 64;

/// 特殊トークンの枠の数。名前付きの 4 個以外は予備で、後から用途を割り当てても既存の ID が変わらない
pub const N_SPECIAL: usize = 16;
const NAMED_SPECIALS: [&str; 4] = ["<|endoftext|>", "<|user|>", "<|assistant|>", "<|end|>"];

/// ID を u16 に収めるための語彙数の上限
pub const MAX_VOCAB: usize = u16::MAX as usize + 1;
/// バイト 256 個と特殊トークンの枠だけで、マージが 0 個の語彙
pub const MIN_VOCAB: usize = 256 + N_SPECIAL;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Special {
	/// 文書の区切り。学習データでは各文書の直前に置く
	EndOfText = 0,
	User = 1,
	Assistant = 2,
	/// 発言の終わり
	End = 3,
}

impl Special {
	pub fn from_index(i: usize) -> Option<Self> {
		[Self::EndOfText, Self::User, Self::Assistant, Self::End]
			.get(i)
			.copied()
	}
}

fn special_names() -> Vec<String> {
	(0..N_SPECIAL)
		.map(|i| {
			NAMED_SPECIALS
				.get(i)
				.map_or_else(|| format!("<|reserved_{i}|>"), |s| s.to_string())
		})
		.collect()
}

pub struct Tokenizer {
	merges: Vec<(u32, u32)>,
	// ペアから、マージ後の ID を引く。ID はマージの学習順と同じ順に並ぶので、小さいほど先に適用する
	ranks: HashMap<(u32, u32), u32>,
	// 特殊トークン以外の ID ごとのバイト列
	pieces: Vec<Vec<u8>>,
}

#[derive(Serialize, Deserialize)]
struct TokenizerFile {
	version: u32,
	pretokenizer: String,
	specials: Vec<String>,
	merges: Vec<[u32; 2]>,
}

impl Tokenizer {
	fn from_merges(merges: Vec<(u32, u32)>) -> Result<Self> {
		if 256 + merges.len() + N_SPECIAL > MAX_VOCAB {
			candle_core::bail!(
				"マージが {} 個あり、語彙が u16 の ID（{MAX_VOCAB} 個）に収まらない",
				merges.len()
			);
		}
		let mut pieces: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
		let mut ranks = HashMap::with_capacity(merges.len());
		for (i, &(a, b)) in merges.iter().enumerate() {
			let id = (256 + i) as u32;
			// 後のマージを参照するマージは、encode で適用順が決まらない
			if a >= id || b >= id {
				candle_core::bail!("マージ {i} が、それより後に作られる ID を参照している: ({a}, {b})");
			}
			if ranks.insert((a, b), id).is_some() {
				candle_core::bail!("マージ {i} のペア ({a}, {b}) が重複している");
			}
			let mut p = pieces[a as usize].clone();
			p.extend_from_slice(&pieces[b as usize]);
			pieces.push(p);
		}
		Ok(Self { merges, ranks, pieces })
	}

	pub fn vocab_size(&self) -> usize {
		self.pieces.len() + N_SPECIAL
	}

	pub fn special(&self, s: Special) -> u32 {
		(self.pieces.len() + s as usize) as u32
	}

	/// 特殊トークンなら、特殊トークンの枠の中での番号を返す
	pub fn special_index(&self, id: u32) -> Option<usize> {
		(id as usize).checked_sub(self.pieces.len()).filter(|&i| i < N_SPECIAL)
	}

	/// 特殊トークンは元の文字列のバイトを持たないので、空を返す
	pub fn token_bytes(&self, id: u32) -> &[u8] {
		self.pieces.get(id as usize).map_or(&[], Vec::as_slice)
	}

	/// 特殊トークンは名前（`<|endoftext|>` など）で書き出す。UTF-8 として不正なバイト列は U+FFFD に置き換える
	pub fn decode(&self, ids: &[u32]) -> String {
		let mut out = String::new();
		let mut dec = StreamDecoder::default();
		for &id in ids {
			match self.special_index(id) {
				Some(i) => {
					out.push_str(&dec.finish());
					out.push_str(&special_names()[i]);
				}
				None => out.push_str(&dec.push(self.token_bytes(id))),
			}
		}
		out.push_str(&dec.finish());
		out
	}

	/// 特殊トークンの名前と同じ文字列も通常のテキストとして encode する。
	/// 入力の文字列から特殊トークンの ID が作られないので、利用者の入力で会話の区切りを偽装できない
	pub fn encode(&self, text: &str) -> Vec<u32> {
		let mut out = Vec::new();
		for piece in pretokenize(text) {
			self.encode_piece(piece.as_bytes(), &mut out);
		}
		out
	}

	// 学習時は全単語に対してマージを学習順に 1 つずつ適用した。ここでも塊の中にある順位の最も小さいペアから適用すると、
	// 学習時と同じ順でマージがかかり、学習データの単語は学習時の最終的な分割と一致する
	fn encode_piece(&self, bytes: &[u8], out: &mut Vec<u32>) {
		// 下の走査はマージ 1 回ごとに塊全体を見直すので O(マージ数 × 塊の長さ) になる。
		// 空白を含まない長い行（minify した JS、base64 など）は 1 つの塊になり、長さに上限が無いので、長い塊は差分更新の方法で処理する
		if bytes.len() > LONG_PIECE {
			self.encode_long_piece(bytes, out);
		} else {
			self.encode_short_piece(bytes, out);
		}
	}

	fn encode_short_piece(&self, bytes: &[u8], out: &mut Vec<u32>) {
		let mut ids: Vec<u32> = bytes.iter().map(|&b| b as u32).collect();
		while let Some(new_id) = ids
			.windows(2)
			.filter_map(|w| self.ranks.get(&(w[0], w[1])).copied())
			.min()
		{
			merge_pair(&mut ids, self.merges[new_id as usize - 256], new_id);
		}
		out.extend(ids);
	}

	// encode_piece と同じ分割を O(n log n) で求める。隣り合うペアの候補を (マージ後の ID, 位置) の小さい順に取り出し、
	// マージした位置の前後だけ候補を足す。同じ ID の候補は左から順に取り出されるので、merge_pair の「左から重ならないように」と一致する。
	// マージで新しくできるペアのマージ後の ID は、今適用している ID より必ず大きい（from_merges の検査で保証される）ので、
	// 小さい ID のマージを後から適用し直すことは起きず、学習順に 1 つずつ全体へ適用した結果と同じになる
	// 作業用の配列は入力 1 バイトあたり約 21 バイト（ids・next・prev・alive）と、候補のヒープ（最大で約 2n 件 × 16 バイト）を使う
	fn encode_long_piece(&self, bytes: &[u8], out: &mut Vec<u32>) {
		const NONE: usize = usize::MAX;
		let n = bytes.len();
		// ペアが無いので、バイトのまま返す。下の 0..n - 1 は n が 0 だと桁あふれする
		if n < 2 {
			out.extend(bytes.iter().map(|&b| b as u32));
			return;
		}
		let mut ids: Vec<u32> = bytes.iter().map(|&b| b as u32).collect();
		let mut next: Vec<usize> = (1..=n).map(|i| if i < n { i } else { NONE }).collect();
		let mut prev: Vec<usize> = (0..n).map(|i| i.checked_sub(1).unwrap_or(NONE)).collect();
		let mut alive = vec![true; n];
		let mut heap = BinaryHeap::new();
		let push = |heap: &mut BinaryHeap<_>, ids: &[u32], i: usize, j: usize| {
			if let Some(&id) = self.ranks.get(&(ids[i], ids[j])) {
				heap.push(Reverse((id, i)));
			}
		};
		for i in 0..n - 1 {
			push(&mut heap, &ids, i, i + 1);
		}
		while let Some(Reverse((id, i))) = heap.pop() {
			let j = next[i];
			// 候補を積んだ後に、どちらかの位置が別のマージに使われていれば、その候補は無効
			if !alive[i] || j == NONE || (ids[i], ids[j]) != self.merges[id as usize - 256] {
				continue;
			}
			ids[i] = id;
			alive[j] = false;
			next[i] = next[j];
			if next[i] != NONE {
				prev[next[i]] = i;
				push(&mut heap, &ids, i, next[i]);
			}
			if prev[i] != NONE {
				push(&mut heap, &ids, prev[i], i);
			}
		}
		out.extend((0..n).filter(|&i| alive[i]).map(|i| ids[i]));
	}

	/// 各文書の直前に文書の区切りを置いて並べた ID 列を返す。文書をスレッド数のまとまりに分けて並列に encode し、
	/// 出力の順序は入力の順序と同じにする
	pub fn encode_documents(&self, docs: &[&str]) -> Vec<u16> {
		let parts: Vec<Vec<u16>> = std::thread::scope(|s| {
			let handles: Vec<_> = balanced_groups(docs, n_threads())
				.into_iter()
				.map(|group| {
					s.spawn(move || {
						let mut enc = CorpusEncoder::new(self);
						let mut out = Vec::new();
						for doc in group {
							enc.push_document(doc, &mut out);
						}
						out
					})
				})
				.collect();
			handles
				.into_iter()
				.map(|h| h.join().expect("encode のスレッドが panic した"))
				.collect()
		});
		let mut ids = Vec::with_capacity(parts.iter().map(Vec::len).sum());
		for p in parts {
			ids.extend(p);
		}
		ids
	}

	/// 学習データの塊ごとの出現回数から、語彙が vocab_size 個（特殊トークンの枠を含む）になるまでマージを学習する。
	/// 隣り合うペアが 1 つも残らなくなった場合は、その時点で止めるので語彙は vocab_size より小さくなる。
	/// 回数が同じペアは ID の小さいペアを先に選ぶので、counts の中身が同じなら結果のマージ表も同じになる。
	/// progress は (学習済みのマージ数, 目標のマージ数) でマージのたびに呼ばれる
	pub fn train(
		counts: &HashMap<&str, u64>,
		vocab_size: usize,
		mut progress: impl FnMut(usize, usize),
	) -> Result<Self> {
		if !(MIN_VOCAB..=MAX_VOCAB).contains(&vocab_size) {
			candle_core::bail!("語彙数は {MIN_VOCAB} 以上 {MAX_VOCAB} 以下である必要がある: {vocab_size}");
		}
		let n_merges = vocab_size - MIN_VOCAB;
		// 1 バイトの塊はペアを持たず、どのマージにも関わらない
		let (mut words, word_counts): (Vec<Vec<u32>>, Vec<u64>) = counts
			.iter()
			.filter(|(w, _)| w.len() >= 2)
			.map(|(w, &c)| (w.bytes().map(u32::from).collect(), c))
			.unzip();

		let mut pair_counts: HashMap<(u32, u32), u64> = HashMap::new();
		// ペアを含む単語の番号。マージで単語からペアが消えても取り除かないので、使う側で単語が今もペアを含むかを確かめる
		let mut pair_words: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
		for (i, w) in words.iter().enumerate() {
			for p in w.windows(2) {
				*pair_counts.entry((p[0], p[1])).or_default() += word_counts[i];
				pair_words.entry((p[0], p[1])).or_default().push(i as u32);
			}
		}
		// 回数の多い順、同じ回数なら ID の小さい順に取り出す。既存のペアの回数はマージで減る一方なので、
		// ヒープの値は常に実際の回数以上になる。取り出した値が実際の回数と違えば、実際の回数で入れ直す
		let mut heap: BinaryHeap<(u64, Reverse<(u32, u32)>)> =
			pair_counts.iter().map(|(&p, &c)| (c, Reverse(p))).collect();

		let mut merges = Vec::with_capacity(n_merges);
		while merges.len() < n_merges {
			let Some((c, Reverse(pair))) = heap.pop() else {
				break;
			};
			let current = pair_counts.get(&pair).copied().unwrap_or(0);
			if c != current {
				if current > 0 {
					heap.push((current, Reverse(pair)));
				}
				continue;
			}
			let new_id = (256 + merges.len()) as u32;
			merges.push(pair);

			let mut targets = pair_words.remove(&pair).unwrap_or_default();
			targets.sort_unstable();
			targets.dedup();
			// マージで新しくできるペアは必ず new_id を含む。それ以外のペアの回数は減るだけなので、ヒープに入れ直すのは new_id を含むペアだけでよい
			let mut created = Vec::new();
			for wi in targets {
				let w = &mut words[wi as usize];
				if !w.windows(2).any(|p| (p[0], p[1]) == pair) {
					continue;
				}
				let n = word_counts[wi as usize];
				for p in w.windows(2) {
					let key = (p[0], p[1]);
					let e = pair_counts.get_mut(&key).expect("単語に含まれるペアは回数表にある");
					*e -= n;
					if *e == 0 {
						pair_counts.remove(&key);
					}
				}
				merge_pair(w, pair, new_id);
				for p in w.windows(2) {
					let key = (p[0], p[1]);
					*pair_counts.entry(key).or_default() += n;
					if key.0 == new_id || key.1 == new_id {
						pair_words.entry(key).or_default().push(wi);
						created.push(key);
					}
				}
			}
			created.sort_unstable();
			created.dedup();
			for p in created {
				if let Some(&c) = pair_counts.get(&p) {
					heap.push((c, Reverse(p)));
				}
			}
			progress(merges.len(), n_merges);
		}
		Self::from_merges(merges)
	}

	pub fn to_json(&self) -> String {
		let file = TokenizerFile {
			version: FORMAT_VERSION,
			pretokenizer: PRETOKENIZER.to_string(),
			specials: special_names(),
			merges: self.merges.iter().map(|&(a, b)| [a, b]).collect(),
		};
		serde_json::to_string(&file).expect("TokenizerFile は JSON に変換できる")
	}

	pub fn from_json(s: &str) -> Result<Self> {
		let file: TokenizerFile =
			serde_json::from_str(s).map_err(|e| candle_core::Error::Msg(format!("トークナイザーの JSON: {e}")))?;
		if file.version != FORMAT_VERSION {
			candle_core::bail!(
				"トークナイザーの形式の版 {} には対応していない（対応は {FORMAT_VERSION}）",
				file.version
			);
		}
		if file.pretokenizer != PRETOKENIZER {
			candle_core::bail!(
				"前処理の規則 {} には対応していない（対応は {PRETOKENIZER}）",
				file.pretokenizer
			);
		}
		if file.specials != special_names() {
			candle_core::bail!("特殊トークンの並びがこのプログラムの定義と一致しない");
		}
		Self::from_merges(file.merges.into_iter().map(|[a, b]| (a, b)).collect())
	}

	pub fn save(&self, path: &Path) -> Result<()> {
		std::fs::write(path, self.to_json()).map_err(|e| candle_core::Error::Msg(format!("{}: {e}", path.display())))
	}

	pub fn load(path: &Path) -> Result<Self> {
		let s = std::fs::read_to_string(path).map_err(|e| {
			candle_core::Error::Msg(format!(
				"{}: {e}（tokenizer train サブコマンドで作成する）",
				path.display()
			))
		})?;
		Self::from_json(&s).map_err(|e| candle_core::Error::Msg(format!("{}: {e}", path.display())))
	}

	/// ID 列のファイルに記録し、読み込み時に別のトークナイザーで作った ID 列を検出するための値。
	/// to_json の出力の FNV-1a ハッシュで、マージ表・特殊トークン・前処理の版のどれかが違えば値が変わる
	pub fn fingerprint(&self) -> u64 {
		self.to_json().bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
			(h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3)
		})
	}
}

// ids の中の (a, b) を左から順に重ならないように c に置き換える。"a a a" に (a, a) なら "c a" になる。
// 学習と encode で同じ関数を使い、両者の分割を一致させる
fn merge_pair(ids: &mut Vec<u32>, (a, b): (u32, u32), c: u32) {
	let (mut r, mut w) = (0, 0);
	while r < ids.len() {
		if r + 1 < ids.len() && ids[r] == a && ids[r + 1] == b {
			ids[w] = c;
			r += 2;
		} else {
			ids[w] = ids[r];
			r += 1;
		}
		w += 1;
	}
	ids.truncate(w);
}

/// コーパスの encode 用。同じ塊（" the" など）はコーパス中に何度も現れるので、塊ごとの結果をキャッシュして再利用する
pub struct CorpusEncoder<'a> {
	tok: &'a Tokenizer,
	cache: HashMap<Box<[u8]>, Box<[u16]>>,
	scratch: Vec<u32>,
}

impl<'a> CorpusEncoder<'a> {
	// 出現が 1 回だけの塊が多いコーパスでもメモリが増え続けないよう、この件数を超えたらキャッシュを空にする。
	// キャッシュに入れる塊は LONG_PIECE バイト以下で、1 件はアロケーターの管理領域を含めて英単語程度の塊で約 130 バイト、
	// 64 バイトの塊で約 290 バイト。スレッドごとに持つので、1 スレッドあたり典型的に約 30MB、最大で約 75MB になる
	const CACHE_LIMIT: usize = 1 << 18;

	pub fn new(tok: &'a Tokenizer) -> Self {
		Self {
			tok,
			cache: HashMap::new(),
			scratch: Vec::new(),
		}
	}

	pub fn push_document(&mut self, doc: &str, out: &mut Vec<u16>) {
		out.push(self.tok.special(Special::EndOfText) as u16);
		for piece in pretokenize(doc) {
			let key = piece.as_bytes();
			if let Some(ids) = self.cache.get(key) {
				out.extend_from_slice(ids);
				continue;
			}
			self.scratch.clear();
			self.tok.encode_piece(key, &mut self.scratch);
			// 語彙は MAX_VOCAB 以下なので ID は u16 に収まる
			let ids: Box<[u16]> = self.scratch.iter().map(|&id| id as u16).collect();
			out.extend_from_slice(&ids);
			// 1 件の大きさは塊の長さに比例し、CACHE_LIMIT は件数の上限なので、長い塊を入れるとメモリの上限が無くなる。
			// 長い塊はコーパス中で繰り返し現れることがほとんどないので、キャッシュに入れない
			if key.len() > LONG_PIECE {
				continue;
			}
			if self.cache.len() >= Self::CACHE_LIMIT {
				self.cache.clear();
			}
			self.cache.insert(key.into(), ids);
		}
	}
}

/// 文書ごとの塊の出現回数を、文書をスレッド数のまとまりに分けて並列に数える
pub fn count_pieces<'a>(docs: &[&'a str]) -> HashMap<&'a str, u64> {
	let maps: Vec<HashMap<&'a str, u64>> = std::thread::scope(|s| {
		let handles: Vec<_> = balanced_groups(docs, n_threads())
			.into_iter()
			.map(|group| {
				s.spawn(move || {
					let mut m: HashMap<&'a str, u64> = HashMap::new();
					for &doc in group {
						for piece in pretokenize(doc) {
							*m.entry(piece).or_default() += 1;
						}
					}
					m
				})
			})
			.collect();
		handles
			.into_iter()
			.map(|h| h.join().expect("集計のスレッドが panic した"))
			.collect()
	});
	let mut iter = maps.into_iter();
	let mut total = iter.next().unwrap_or_default();
	for m in iter {
		for (k, v) in m {
			*total.entry(k).or_default() += v;
		}
	}
	total
}

fn n_threads() -> usize {
	std::thread::available_parallelism().map_or(1, |n| n.get())
}

// docs を連続した n 個以下のまとまりに分ける。各まとまりのバイト数がほぼ等しくなるように区切り、順序は保つ
fn balanced_groups<'s, 'a>(docs: &'s [&'a str], n: usize) -> Vec<&'s [&'a str]> {
	let total: usize = docs.iter().map(|d| d.len()).sum();
	let target = total.div_ceil(n.max(1)).max(1);
	let mut groups = Vec::new();
	let (mut start, mut acc) = (0, 0);
	for (i, d) in docs.iter().enumerate() {
		acc += d.len();
		if acc >= target {
			groups.push(&docs[start..=i]);
			start = i + 1;
			acc = 0;
		}
	}
	if start < docs.len() {
		groups.push(&docs[start..]);
	}
	groups
}

/// 文字列を、マージがまたいではいけない境界で塊に分ける。塊を全部つなげると元の文字列に戻る。
/// GPT-4（cl100k）の分割の正規表現を基にした規則を、先頭から順に試して最初に当てはまったものを使う:
///   1. `'s` `'t` `'re` `'ve` `'m` `'ll` `'d`（大文字小文字を区別しない）
///   2. 先頭に空白 1 個を付けてもよい英字の並び
///   3. 1〜3 桁の数字
///   4. 先頭に空白 1 個を付けてもよい記号の並び（空白・英字・数字以外）
///   5. 改行を含む空白の並びは、最後の改行まで
///   6. 空白の並びが文字列の終わりまで続くなら全部、そうでなければ最後の 1 文字を残した残り
///   7. 空白 1 文字
///
/// 6 で空白を 1 文字残すのは、次の塊が 2 か 4 の「空白 1 個 + 単語」になり、" the" のように空白込みのトークンを使えるようにするため。
/// 正規表現の `\s+(?!\S)` は Rust の regex クレートが先読みに対応しないので使えず、文字種の判定で書く。
/// 英字は char::is_alphabetic、数字は char::is_numeric、空白は char::is_whitespace で判定する
pub fn pretokenize(text: &str) -> Pretokens<'_> {
	Pretokens { text, pos: 0 }
}

pub struct Pretokens<'a> {
	text: &'a str,
	pos: usize,
}

impl<'a> Iterator for Pretokens<'a> {
	type Item = &'a str;

	fn next(&mut self) -> Option<&'a str> {
		if self.pos >= self.text.len() {
			return None;
		}
		let end = self.pos + piece_len(&self.text[self.pos..]);
		let piece = &self.text[self.pos..end];
		self.pos = end;
		Some(piece)
	}
}

fn is_letter(c: char) -> bool {
	c.is_alphabetic()
}

fn is_symbol(c: char) -> bool {
	!c.is_whitespace() && !c.is_alphabetic() && !c.is_numeric()
}

// rest の先頭から f を満たす文字が続くバイト数
fn run_len(rest: &str, f: impl Fn(char) -> bool) -> usize {
	rest.find(|c| !f(c)).unwrap_or(rest.len())
}

// rest は空でない。先頭の塊のバイト数を返す
fn piece_len(rest: &str) -> usize {
	let c = rest.chars().next().expect("rest は空でない");
	let next = rest[c.len_utf8()..].chars().next();

	if c == '\''
		&& let Some(n) = contraction_len(&rest[1..])
	{
		return 1 + n;
	}
	for class in [is_letter as fn(char) -> bool, is_symbol] {
		if class(c) {
			return run_len(rest, class);
		}
		if c == ' ' && next.is_some_and(class) {
			return 1 + run_len(&rest[1..], class);
		}
	}
	if c.is_numeric() {
		return rest
			.char_indices()
			.take(3)
			.take_while(|&(_, d)| d.is_numeric())
			.last()
			.map_or(c.len_utf8(), |(i, d)| i + d.len_utf8());
	}

	// is_letter・is_symbol・is_numeric・is_whitespace で全文字を分類するので、ここに来るのは c が空白のとき。
	// 分類の定義を変えて空白以外の文字がここに来ても、塊が空になって Pretokens::next が先に進まなくならないよう、
	// 返す長さは c の 1 文字分以上にする
	let ws = run_len(rest, char::is_whitespace);
	let run = &rest[..ws];
	if let Some(i) = run.rfind(['\r', '\n']) {
		return i + 1;
	}
	if ws == rest.len() {
		return ws;
	}
	match run.char_indices().last() {
		Some((last, _)) if last > 0 => last,
		_ => ws.max(c.len_utf8()),
	}
}

// `'` の直後の文字列が縮約形の語尾で始まるなら、その語尾のバイト数を返す
fn contraction_len(after: &str) -> Option<usize> {
	["s", "t", "re", "ve", "m", "ll", "d"].into_iter().find_map(|suffix| {
		after
			.get(..suffix.len())
			.filter(|head| head.eq_ignore_ascii_case(suffix))
			.map(|_| suffix.len())
	})
}

/// トークンのバイト列を 1 つずつ受け取り、UTF-8 として完成した文字だけを返す。
/// 1 つの文字が複数のトークンに分かれることがあるので、文字の途中で終わったバイトは次の push まで持ち越す。
/// UTF-8 として不正なバイトは U+FFFD に置き換える
#[derive(Default)]
pub struct StreamDecoder {
	pending: Vec<u8>,
}

impl StreamDecoder {
	pub fn push(&mut self, bytes: &[u8]) -> String {
		self.pending.extend_from_slice(bytes);
		let mut out = String::new();
		loop {
			match std::str::from_utf8(&self.pending) {
				Ok(s) => {
					out.push_str(s);
					self.pending.clear();
					return out;
				}
				Err(e) => {
					let valid = e.valid_up_to();
					out.push_str(std::str::from_utf8(&self.pending[..valid]).expect("valid_up_to までは UTF-8"));
					match e.error_len() {
						Some(n) => {
							out.push('\u{FFFD}');
							self.pending.drain(..valid + n);
						}
						// 末尾が文字の途中で終わっている。続きのバイトが次の push で来る可能性がある
						None => {
							self.pending.drain(..valid);
							return out;
						}
					}
				}
			}
		}
	}

	/// 持ち越しているバイトを U+FFFD に置き換えて返し、空にする。生成の終わりと特殊トークンの前で呼ぶ
	pub fn finish(&mut self) -> String {
		let s = String::from_utf8_lossy(&self.pending).into_owned();
		self.pending.clear();
		s
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use rand::rngs::StdRng;
	use rand::{RngExt, SeedableRng};

	fn pieces(s: &str) -> Vec<&str> {
		pretokenize(s).collect()
	}

	fn trained(text: &str, vocab: usize) -> Tokenizer {
		let docs = [text];
		Tokenizer::train(&count_pieces(&docs), vocab, |_, _| {}).unwrap()
	}

	const SAMPLE: &str = "Once upon a time, there was a little girl named Lily. She liked to play outside.\n\
		One day, Lily's mom said, \"Don't go far!\" Lily said, \"I won't.\" They played 123 games and 4567 more.\n\n\
		The end.   Then   more    spaces\tand\ttabs.\r\n";

	#[test]
	fn pretokenize_rules() {
		assert_eq!(pieces("Hello world"), ["Hello", " world"]);
		assert_eq!(pieces("a   b"), ["a", "  ", " b"]);
		assert_eq!(pieces("don't I'LL"), ["don", "'t", " I", "'LL"]);
		assert_eq!(pieces("12345 x"), ["123", "45", " x"]);
		assert_eq!(pieces("Hi!! ok?\n\nNo"), ["Hi", "!!", " ok", "?", "\n\n", "No"]);
		assert_eq!(pieces("x  \n  y"), ["x", "  \n", " ", " y"]);
		assert_eq!(pieces("end  "), ["end", "  "]);
		assert_eq!(pieces("a\tb"), ["a", "\t", "b"]);
		assert_eq!(pieces(" 7"), [" ", "7"]);
		assert_eq!(pieces("\"quote\""), ["\"", "quote", "\""]);
	}

	#[test]
	fn pretokenize_concatenates_to_input() {
		let mut rng = StdRng::seed_from_u64(0);
		let alphabet: Vec<char> = "ab Z9'\n\r\t.!あ😀 ".chars().collect();
		for _ in 0..2000 {
			let len = rng.random_range(0..40);
			let s: String = (0..len)
				.map(|_| alphabet[rng.random_range(0..alphabet.len())])
				.collect();
			assert_eq!(pieces(&s).concat(), s);
			assert!(pieces(&s).iter().all(|p| !p.is_empty()));
		}
	}

	#[test]
	fn merges_follow_pair_counts() {
		let counts: HashMap<&str, u64> = [("ab", 5), ("abc", 3), ("bc", 1)].into_iter().collect();
		let tok = Tokenizer::train(&counts, MIN_VOCAB + 2, |_, _| {}).unwrap();
		let (a, b, c) = (b'a' as u32, b'b' as u32, b'c' as u32);
		assert_eq!(tok.merges, [(a, b), (256, c)]);
		assert_eq!(tok.encode("abc"), [257]);
		assert_eq!(tok.encode("bc"), [b, c]);
	}

	#[test]
	fn roundtrip_any_text() {
		let tok = trained(SAMPLE, 400);
		assert!(tok.vocab_size() > MIN_VOCAB);
		for s in [
			SAMPLE,
			"",
			"日本語と絵文字😀も通る",
			"\u{0}\u{7f} control",
			"<|endoftext|>",
		] {
			let ids = tok.encode(s);
			assert_eq!(tok.decode(&ids), s);
			assert!(ids.iter().all(|&id| tok.special_index(id).is_none()));
		}
		assert!(tok.encode(SAMPLE).len() < SAMPLE.len());
	}

	#[test]
	fn training_is_deterministic() {
		// HashMap の走査順は実行ごと・インスタンスごとに変わるので、2 回の学習で走査順の影響が無いことを確かめる
		let a = trained(SAMPLE, 350);
		let b = trained(SAMPLE, 350);
		assert_eq!(a.merges, b.merges);
	}

	#[test]
	fn stops_when_no_pairs_left() {
		let tok = trained("ab ab", 1000);
		assert!(tok.vocab_size() < 1000);
		assert_eq!(tok.encode(" ab").len(), 1);
	}

	#[test]
	fn corpus_encoder_matches_encode() {
		let tok = trained(SAMPLE, 400);
		let docs = ["Lily played.", "", "The end.   Then"];
		let ids = tok.encode_documents(&docs);
		let eot = tok.special(Special::EndOfText);
		let mut expected = Vec::new();
		for d in docs {
			expected.push(eot);
			expected.extend(tok.encode(d));
		}
		assert_eq!(ids.iter().map(|&id| id as u32).collect::<Vec<_>>(), expected);
		assert_eq!(tok.special_index(eot), Some(0));
		assert_eq!(tok.decode(&expected[..1]), "<|endoftext|>");
	}

	#[test]
	fn json_roundtrip_keeps_fingerprint() {
		let tok = trained(SAMPLE, 400);
		let back = Tokenizer::from_json(&tok.to_json()).unwrap();
		assert_eq!(back.merges, tok.merges);
		assert_eq!(back.fingerprint(), tok.fingerprint());
		assert_ne!(trained(SAMPLE, 350).fingerprint(), tok.fingerprint());
	}

	#[test]
	fn long_piece_path_matches_short_path() {
		// 2 文字だけの単語で学習すると同じペアが重なって現れ（"aaa" の (a, a) など）、マージの適用順の違いが分割の違いとして表に出やすい
		let mut rng = StdRng::seed_from_u64(1);
		let text: String = (0..3000)
			.map(|_| {
				let len = rng.random_range(1..12);
				let w: String = (0..len).map(|_| if rng.random_bool(0.6) { 'a' } else { 'b' }).collect();
				w + " "
			})
			.collect();
		let tok = trained(&text, 600);
		for _ in 0..500 {
			let len = rng.random_range(2..400);
			let w: Vec<u8> = (0..len)
				.map(|_| if rng.random_bool(0.6) { b'a' } else { b'b' })
				.collect();
			let (mut short, mut long) = (Vec::new(), Vec::new());
			tok.encode_short_piece(&w, &mut short);
			tok.encode_long_piece(&w, &mut long);
			assert_eq!(short, long);
		}
	}

	#[test]
	fn long_piece_handles_tiny_inputs() {
		let tok = trained(SAMPLE, 400);
		for bytes in [&b""[..], b"a"] {
			let mut out = Vec::new();
			tok.encode_long_piece(bytes, &mut out);
			assert_eq!(out, bytes.iter().map(|&b| b as u32).collect::<Vec<_>>());
		}
	}

	#[test]
	fn corpus_encoder_handles_uncached_long_pieces() {
		// LONG_PIECE を超える塊はキャッシュに入れずに出力だけする。改行で区切って同じ長い塊を 2 回置き、
		// 2 回とも出力に残ることと、キャッシュには短い塊だけが入ることを確かめる
		let long = "thethethe".repeat(12);
		let doc = format!("{long}\n{long}.");
		let tok = trained(&doc, 400);
		let mut expected = vec![tok.special(Special::EndOfText)];
		expected.extend(tok.encode(&doc));
		let mut enc = CorpusEncoder::new(&tok);
		let mut out = Vec::new();
		enc.push_document(&doc, &mut out);
		assert_eq!(out.iter().map(|&id| id as u32).collect::<Vec<_>>(), expected);
		assert_eq!(tok.decode(&expected[1..]), doc);
		// キャッシュに入るのは "\n" と "." だけ
		assert!(enc.cache.keys().all(|k| k.len() <= LONG_PIECE));
		assert_eq!(enc.cache.len(), 2);
	}

	#[test]
	fn rejects_merge_referring_to_later_id() {
		assert!(Tokenizer::from_merges(vec![(256, 97)]).is_err());
	}

	#[test]
	fn stream_decoder_holds_partial_chars() {
		let mut d = StreamDecoder::default();
		let bytes = "あ".as_bytes();
		assert_eq!(d.push(&bytes[..1]), "");
		assert_eq!(d.push(&bytes[1..]), "あ");
		assert_eq!(d.push(&[0xff, b'a']), "\u{FFFD}a");
		assert_eq!(d.push(&bytes[..2]), "");
		assert_eq!(d.finish(), "\u{FFFD}");
		assert_eq!(d.finish(), "");
	}
}
