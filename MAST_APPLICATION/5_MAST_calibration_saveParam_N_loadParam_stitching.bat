@echo off
setlocal
echo Compiling 5_MAST_calibration_saveParam_N_loadParam_stitching.cpp...

echo Setting up Visual Studio x64 environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"


rem Set the paths to include directories and libraries
set "OPENCV_INCLUDE=C:\opencv\include"
set "OPENCV_LIB=C:\opencv\x64\vc17\lib"
set "CUDA_LIB_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64"
set "MAST_S2_DLL_PATH=..\MAST_S2_BUILD\MAST_S2_DLL_WIN_CV481_CUDA12.8"
set "TBB_DLL_PATH=C:\Program Files (x86)\Intel\oneAPI\tbb\latest\redist\intel64\vc14"
set "CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"

rem Compile the program using /Fe for the output executable
cl 5_MAST_calibration_saveParam_N_loadParam_stitching.cpp /Fe5_MAST_calibration_saveParam_N_loadParam_stitching.exe ^
   /I "%OPENCV_INCLUDE%" ^
   /I "%MAST_S2_DLL_PATH%" ^
   /EHsc ^
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
)

echo Compilation complete.

rem Run the executable in a new CMD session with a minimal PATH.
rem The minimal PATH includes:
rem   - MAST_S2_DLL_PATH for custom DLLs,
rem   - OpenCV binaries,
rem   - Intel TBB DLLs,
rem   - CUDA binaries (for cudart64, etc.),
rem   - Windows system DLLs.
cmd /c "set PATH=%MAST_S2_DLL_PATH%;C:\opencv\x64\vc17\bin;%TBB_DLL_PATH%;%CUDA_BIN%;C:\Windows\system32 && 5_MAST_calibration_saveParam_N_loadParam_stitching.exe"

echo Done.
endlocal
