@echo off
rem Download HuggingFaceTB/everyday-conversations-llama3.1-2k (about 2.2k simple User/AI chats, Apache-2.0)
rem and convert it to JSONL for the finetune subcommand. Requires uv (pyarrow is fetched on demand).
rem The language level matches TinyStories, so finetune a model pretrained on it:
rem   target\release\my-ai.exe finetune data\chat-train.jsonl --val data\chat-valid.jsonl --base model.safetensors
setlocal

set "BASE=https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k/resolve/main/data"

pushd "%~dp0.."
if not exist data mkdir data
curl -fL --retry 5 --retry-delay 5 --connect-timeout 30 -o data\chat-train.parquet "%BASE%/train_sft-00000-of-00001.parquet"
if not "%ERRORLEVEL%"=="0" goto :failed
curl -fL --retry 5 --retry-delay 5 --connect-timeout 30 -o data\chat-valid.parquet "%BASE%/test_sft-00000-of-00001.parquet"
if not "%ERRORLEVEL%"=="0" goto :failed
uv run --no-project --with pyarrow python scripts\parquet_to_jsonl.py data\chat-train.parquet data\chat-train.jsonl
if not "%ERRORLEVEL%"=="0" goto :failed
uv run --no-project --with pyarrow python scripts\parquet_to_jsonl.py data\chat-valid.parquet data\chat-valid.jsonl
if not "%ERRORLEVEL%"=="0" goto :failed
popd
exit /b 0

:failed
echo ERROR: download or conversion failed 1>&2
popd
exit /b 1
