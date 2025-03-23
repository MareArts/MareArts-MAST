@echo off
echo Compiling b2_test_opencv.cpp...

rem IMPORTANT: Force x64 environment by calling the x64 native tools command prompt
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
)

rem Set the paths to include directories and libraries
set "OPENCV_INCLUDE=C:\opencv\include"
set "OPENCV_LIB=C:\opencv\x64\vc17\lib"
set "CUDA_LIB_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64"
set "MAST_S2_DLL_PATH=..\MAST_S2_BUILD\MAST_S2_DLL_WIN_CV481_CUDA12.8"
set "TBB_DLL_PATH=C:\Program Files (x86)\Intel\oneAPI\tbb\latest\redist\intel64\vc14"
set "CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"

rem Compile the program 
cl /Fob2_test_opencv.obj /c /EHsc /MD /I%OPENCV_INCLUDE% b2_test_opencv.cpp 

rem Link the program
link /OUT:b2_test_opencv.exe b2_test_opencv.obj ^
  /LIBPATH:%OPENCV_LIB% ^
  opencv_core481.lib ^
  opencv_highgui481.lib ^
  opencv_imgcodecs481.lib ^
  opencv_imgproc481.lib ^
  opencv_cudaarithm481.lib ^
  opencv_cudawarping481.lib

if %ERRORLEVEL% NEQ 0 (
    echo Compilation or linking failed!
    exit /b %ERRORLEVEL%
)

echo Compilation complete.

echo Running b2_test_opencv.exe...
cmd /c "set PATH=%MAST_S2_DLL_PATH%;C:\opencv\x64\vc17\bin;%TBB_DLL_PATH%;%CUDA_BIN%;C:\Windows\system32 && b2_test_opencv.exe"

echo Done.