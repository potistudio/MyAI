@echo off
rem Download TinyStories V2 (GPT-4 generated stories, about 2.2GB train + 22MB valid) into data\.
rem Train a tokenizer, convert the corpus to token IDs once, then train on the .bin files.
rem The train command also accepts the .txt files directly, but doing so re-encodes them on every run.
rem With --sample-mb 4096, the merges are learned from the whole train file (the default reads only the first 1024 MB).
rem   target\release\my-ai.exe tokenizer train data\TinyStoriesV2-GPT4-train.txt --vocab 8192 --sample-mb 4096 -o tokenizer.json
rem   target\release\my-ai.exe tokenizer encode data\TinyStoriesV2-GPT4-train.txt -o data\ts-train.bin
rem   target\release\my-ai.exe tokenizer encode data\TinyStoriesV2-GPT4-valid.txt -o data\ts-valid.bin
rem   target\release\my-ai.exe train data\ts-train.bin --val data\ts-valid.bin --dropout 0
rem TinyStories uses a small vocabulary, so 8192 tokens is enough (32768 is meant for the large general corpus).
rem The corpus is far larger than the default 5000 steps can cover (about 15% of one epoch), so dropout is unnecessary.
rem Re-running resumes a partial download (-C -).
setlocal

set "BASE=https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"

pushd "%~dp0.."
if not exist data mkdir data
curl -fL -C - --retry 5 --retry-delay 5 --connect-timeout 30 -o data\TinyStoriesV2-GPT4-valid.txt "%BASE%/TinyStoriesV2-GPT4-valid.txt"
if not "%ERRORLEVEL%"=="0" goto :failed
curl -fL -C - --retry 5 --retry-delay 5 --connect-timeout 30 -o data\TinyStoriesV2-GPT4-train.txt "%BASE%/TinyStoriesV2-GPT4-train.txt"
if not "%ERRORLEVEL%"=="0" goto :failed
popd
exit /b 0

:failed
echo ERROR: download failed 1>&2
popd
exit /b 1
