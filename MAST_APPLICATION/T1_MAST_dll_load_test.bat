@echo off
setlocal

echo Compiling T1_MAST_dll_load_test.cpp...

echo Setting up Visual Studio x64 environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"


rem Set the paths to include directories and libraries
set "OPENCV_INCLUDE=C:\opencv\include"
set "OPENCV_LIB=C:\opencv\x64\vc17\lib"
set "CUDA_LIB_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64"
set "CUDA_INCLUDE_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include"
set "CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
set "MAST_S2_DLL_PATH=..\MAST_S2_BUILD\MAST_S2_DLL_WIN_CV481_CUDA12.8"
set "TBB_DLL_PATH=C:\Program Files (x86)\Intel\oneAPI\tbb\latest\redist\intel64\vc14"

rem Check if source file exists
if not exist T1_MAST_dll_load_test.cpp (
    echo ERROR: Source file T1_MAST_dll_load_test.cpp not found!
    exit /b 1
)

rem Compile the program
cl T1_MAST_dll_load_test.cpp /Fe:T1_MAST_dll_load_test.exe ^
   /I "%OPENCV_INCLUDE%" ^
   /I "%CUDA_INCLUDE_PATH%" ^
   /I "%MAST_S2_DLL_PATH%" ^
   /EHsc /MD ^
   /link ^
   /LIBPATH:"%OPENCV_LIB%" /LIBPATH:"%CUDA_LIB_PATH%" /LIBPATH:"%MAST_S2_DLL_PATH%" ^
   opencv_core481.lib ^
   opencv_imgcodecs481.lib ^
   opencv_cudaarithm481.lib ^
   opencv_videoio481.lib ^
   opencv_highgui481.lib ^
   opencv_imgproc481.lib ^
   opencv_cudaimgproc481.lib ^
   opencv_stitching481.lib ^
   opencv_cudafeatures2d481.lib ^
   opencv_features2d481.lib ^
   opencv_cudalegacy481.lib ^
   opencv_cudawarping481.lib ^
   opencv_calib3d481.lib ^
   opencv_flann481.lib ^
   opencv_cudafilters481.lib ^
   opencv_xfeatures2d481.lib ^
   cudart.lib ^
   "%MAST_S2_DLL_PATH%\MareArtsStitcher.lib"

if %ERRORLEVEL% NEQ 0 (
    echo Compilation failed!
    exit /b %ERRORLEVEL%
) else (
    echo Compilation completed successfully.
)

rem Run the executable with a properly configured PATH to find all required DLLs
echo Running T1_MAST_dll_load_test.exe...
cmd /c "set PATH=%MAST_S2_DLL_PATH%;C:\opencv\x64\vc17\bin;%TBB_DLL_PATH%;%CUDA_BIN%;C:\Windows\system32 && T1_MAST_dll_load_test.exe"

echo Done.
endlocal