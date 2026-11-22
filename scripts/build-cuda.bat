@echo off
rem CUDA-enabled build.
rem nvcc resolves cl.exe through PATH, so the MSVC environment has to be loaded first.
rem A plain shell fails with: nvcc fatal : Cannot find compiler 'cl.exe' in PATH
rem Note: parenthesized blocks are avoided because the (x86) path segment is
rem expanded inside them and terminates the block early.
setlocal

set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

if not exist "%VCVARS%" goto :no_vcvars
call "%VCVARS%" >nul 2>&1
if errorlevel 1 goto :vcvars_failed

pushd "%~dp0.."
cargo build --release --features cuda %*
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%

:no_vcvars
echo ERROR: vcvars64.bat not found: %VCVARS% 1>&2
exit /b 1

:vcvars_failed
echo ERROR: vcvars64.bat failed 1>&2
exit /b 1
