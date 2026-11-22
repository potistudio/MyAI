@echo off
rem Download TinyStories V2 (GPT-4 generated stories, about 2.2GB train + 22MB valid) into data\.
rem Train with:
rem   target\release\my-ai.exe train data\TinyStoriesV2-GPT4-train.txt --val data\TinyStoriesV2-GPT4-valid.txt --dropout 0
rem The corpus is far larger than the default 5000 steps can cover (about 4% of one epoch), so dropout is unnecessary.
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
