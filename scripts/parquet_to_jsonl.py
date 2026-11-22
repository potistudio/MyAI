"""Hugging Face の parquet を、finetune サブコマンドが読む JSONL に変換する。

1 行 1 会話で、{"messages": [{"role": "user" | "assistant", "content": "..."}, ...]} の形にする。
messages 列以外は finetune で使わないので出力しない。

usage: uv run --with pyarrow python scripts/parquet_to_jsonl.py IN.parquet OUT.jsonl
"""

import json
import sys


def main() -> None:
	# 標準エラーの文字コードは端末ごとに違い、日本語の docstring は文字化けしうるので、使い方は ASCII で出す。
	# pyarrow が無い環境でも使い方を表示できるよう、import は引数の検査の後に行う
	if len(sys.argv) != 3:
		sys.exit("usage: python scripts/parquet_to_jsonl.py IN.parquet OUT.jsonl")
	src, dst = sys.argv[1], sys.argv[2]

	import pyarrow.parquet as pq

	rows = pq.read_table(src, columns=["messages"]).to_pylist()
	# finetune が読めない会話は、parquet の行番号が分かるこの時点で止め、出力ファイルは作らない
	for i, row in enumerate(rows):
		if not row["messages"]:
			sys.exit(f"{src}: row {i}: messages is empty")
		for m in row["messages"]:
			if m["role"] not in ("user", "assistant"):
				sys.exit(f"{src}: row {i}: unsupported role {m['role']!r}")
			if not isinstance(m["content"], str):
				sys.exit(f"{src}: row {i}: content is not a string")
	with open(dst, "w", encoding="utf-8", newline="\n") as f:
		for row in rows:
			messages = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
			# ensure_ascii=False で、非 ASCII 文字を \uXXXX にせず UTF-8 のまま書き出す
			f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
	print(f"{dst}: {len(rows)} conversations")


if __name__ == "__main__":
	main()
