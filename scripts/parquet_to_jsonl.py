"""Hugging Face の parquet を、finetune サブコマンドが読む JSONL に変換する。

1 行 1 会話で、{"messages": [{"role": "user" | "assistant", "content": "..."}, ...]} の形にする。
messages 列以外は finetune で使わないので出力しない。

usage: uv run --with pyarrow python scripts/parquet_to_jsonl.py IN.parquet OUT.jsonl
"""

import json
import sys

import pyarrow.parquet as pq


def main() -> None:
	# 標準エラーの文字コードは端末ごとに違い、日本語の docstring は文字化けしうるので、使い方は ASCII で出す
	if len(sys.argv) != 3:
		sys.exit("usage: python scripts/parquet_to_jsonl.py IN.parquet OUT.jsonl")
	src, dst = sys.argv[1], sys.argv[2]
	rows = pq.read_table(src, columns=["messages"]).to_pylist()
	# finetune は user と assistant 以外の role を読めないので、変換の時点で行番号付きで止める
	for i, row in enumerate(rows):
		for m in row["messages"]:
			if m["role"] not in ("user", "assistant"):
				sys.exit(f"{src}: row {i}: unsupported role {m['role']!r}")
	with open(dst, "w", encoding="utf-8", newline="\n") as f:
		for row in rows:
			messages = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
			# ensure_ascii=False で、非 ASCII 文字を \uXXXX にせず UTF-8 のまま書き出す
			f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
	print(f"{dst}: {len(rows)} conversations")


if __name__ == "__main__":
	main()
