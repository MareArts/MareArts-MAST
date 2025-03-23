@echo off
echo Setting up environment and compiling CUDA file...

rem Setup Visual Studio x64 environment
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
)

rem Compile with CUDA nvcc
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe" ^
  b1_test_cuda.cu ^
  -o b1_test_cuda.exe ^
  -Xcompiler "/wd4819" ^
  -Wno-deprecated-gpu-targets

if %ERRORLEVEL% NEQ 0 (
    echo Compilation failed!
    exit /b %ERRORLEVEL%
)

echo Compilation complete.
echo Running executable...
b1_test_cuda.exe

echo Done.