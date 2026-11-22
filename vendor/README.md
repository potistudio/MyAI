# vendor/candle-kernels

`candle-kernels` 0.11.1 の複製に修正を加えたディレクトリ。
ルート `Cargo.toml` の `[patch.crates-io]` から path 参照される。

## 来歴

| 項目 | 値 |
| --- | --- |
| クレート | `candle-kernels` 0.11.1 (crates.io) |
| 上流リポジトリ | https://github.com/huggingface/candle |
| 上流コミット | `59f8e903ca80a32ea7f8514b39748fc979e69e82` (`.cargo_vcs_info.json` に記録) |
| ライセンス | MIT OR Apache-2.0 (下記「ライセンス」を参照) |

## crates.io の tarball との差分

1. `src/compatibility.cuh`: `atomicAdd(double, double)` シムのガード（1 行＋理由コメント 4 行）。
2. `src/compatibility.cuh`: `__hmax_nan` / `__hmin_nan` シムのガードに `defined(__CUDA_ARCH__)` を追加（1 行＋理由コメント 3 行）。
3. `src/compatibility.cuh`: ファイル冒頭の改変表示 3 行（Apache-2.0 §4(b) 用。本リポジトリが付加したもので、上流には存在しない）。
4. `LICENSE-MIT` / `LICENSE-APACHE`: 上流リポジトリの同一コミットから追加（crates.io の tarball には同梱されていない）。
5. `Cargo.lock`: tarball 同梱のものを削除。依存解決はルートの `Cargo.lock` が行う。

上記以外のファイルは crates.io の tarball と一致する。

### 差分 1: `atomicAdd(double, double)`

```diff
-#if __CUDA_ARCH__ < 600
+#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
```

CUDA 13.x の `sm_60_atomic_functions.h` は、ホストパスでも `atomicAdd(double*, double)` を宣言する（`__DEF_IF_HOST` によりボディは空）。
ホストパスでは `__CUDA_ARCH__` が未定義であり、`#if __CUDA_ARCH__ < 600` の評価では 0 として扱われるため、
無条件のガードはホストパスでもシムを有効化し、ツールキット側の宣言と衝突する。
`defined(__CUDA_ARCH__)` を要求することで、シムをデバイスコンパイルに限定する。

### 差分 2: `__hmax_nan` / `__hmin_nan`

```diff
-#if CANDLE_CUDA_VERSION < 12020 && __CUDA_ARCH__ < 800
+#if CANDLE_CUDA_VERSION < 12020 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
```

CUDA 12.2 以降の `cuda_fp16.hpp` は `__hmax_nan` / `__hmin_nan` を全アーキで定義するため、
バージョン条件でシムを無効化できる。一方、CUDA 12.1 以前の `cuda_fp16.hpp` は
`#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)` の下でこれらを定義しており、ホストパスにも定義が存在する。
ホストパスでは `__CUDA_ARCH__` が未定義で 0 と評価されるため、バージョン条件だけではシムがホストパスで有効になり、
ツールキット側の定義と再定義になる。差分 1 と同型の衝突であり、`defined(__CUDA_ARCH__)` の追加で解消する。

確認した宣言位置:

| ツールキット | `cuda_fp16.hpp` の `__hmax_nan` | ホストパスでの可視性 |
| --- | --- | --- |
| 11.8 | `#if !defined(__CUDA_ARCH__) \|\| (__CUDA_ARCH__ >= 800)` (2517 行) | あり |
| 12.1 | `#if defined(_NVHPC_CUDA) \|\| !defined(__CUDA_ARCH__) \|\| (__CUDA_ARCH__ >= 800)` (2641 行) | あり |
| 13.3 | 無条件（`__CUDA_HOSTDEVICE_FP16_DECL__`、3309 行） | あり |

本リポジトリは CUDA 13.3 / sm_86 構成（`.cargo/config.toml`）であり、バージョン条件が偽のためこの差分は no-op。
CUDA 12.1 以前の構成でのみ効果を持つ。

### 上流への反映

crates.io 上の最新版は 0.11.1 であり、上記 2 差分はいずれも含まれていない。上流 `main` にも未反映（2026-11 時点で確認）。
huggingface/candle へ issue / PR を起票して取り込ませた後、このディレクトリと `[patch.crates-io]` を削除する。

## 変更しないと判断した箇所

- `src/compatibility.cuh` の `atomicMaxf` / `atomicMinf` 内の `#if __CUDA_ARCH__ < 700`。
  ホストパスでは pre-sm_70 側に入るが、クレート内に呼び出し元が無く、コンパイルも正常のため上流のままとする。
  `defined(__CUDA_ARCH__)` を一括適用してはならない。条件の妥当性はツールキットごとの宣言規則に依存し、箇所ごとの検証が必要になる。

## ライセンス

上流の宣言は `MIT OR Apache-2.0`。上流に `NOTICE` ファイルは存在しない。

上流の `LICENSE-MIT` は MIT 本文のみで著作権者表示行を含まない。
MIT は「著作権表示と許諾表示の同梱」を条件とするため、この本文だけでは MIT 単独での充足ができない。
本リポジトリは Apache-2.0 側の条件を満たす形で再配布する。

- §4(a) ライセンス文の同梱: `LICENSE-APACHE` を同梱。
- §4(b) 変更したファイルへの表示: `src/compatibility.cuh` 冒頭に改変表示を記載。
- §4(c) 著作権・特許・商標・帰属表示の保持: ソース内の帰属コメント（`moe_align.cu` の vLLM、`mmvq_gguf.cu` / `mmq_quantize.cu` / `moe/gguf.cuh` の llama.cpp、`moe/moe_gguf.cu` の attention.rs など）はすべて改変せず保持している。
- §4(d) `NOTICE` の伝播: 上流に `NOTICE` が存在しないため義務は発生しない。

上流のクレジット: candle project (Hugging Face), https://github.com/huggingface/candle

## 取り扱い

- `cargo vendor` でこのディレクトリを再生成しない。再生成すると上記の修正が失われる。
- 上流ファイルを更新する場合は、本差分を再適用し、`LICENSE-MIT` / `LICENSE-APACHE` を新しい上流コミットから取得し直す。
- `Cargo.toml.orig` と `.cargo_vcs_info.json` は crates.io 配布物の一部であり、来歴の証跡として保持する。
