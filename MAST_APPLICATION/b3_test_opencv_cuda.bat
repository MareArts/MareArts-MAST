@echo off
setlocal

echo Compiling b3_test_opencv_cuda.cpp...

rem Check if source file exists
if not exist b3_test_opencv_cuda.cpp (
    echo ERROR: Source file b3_test_opencv_cuda.cpp not found!
    exit /b 1
)

rem Set up paths - keep these as short as possible to avoid command line length limitations
set OPENCV_INC=C:\opencv\include
set OPENCV_LIB=C:\opencv\x64\vc17\lib
set CUDA_INC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include
set CUDA_LIB=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64

rem Compile the program
cl b3_test_opencv_cuda.cpp /Fe:b3_test_opencv_cuda.exe /I "%OPENCV_INC%" /I "%CUDA_INC%" /EHsc /MD /link /LIBPATH:"%OPENCV_LIB%" /LIBPATH:"%CUDA_LIB%" opencv_core481.lib opencv_highgui481.lib opencv_imgcodecs481.lib opencv_imgproc481.lib opencv_cudaarithm481.lib opencv_cudawarping481.lib opencv_cudaimgproc481.lib cudart.lib

if %ERRORLEVEL% NEQ 0 (
    echo Compilation failed!
    exit /b %ERRORLEVEL%
) else (
    echo Compilation completed successfully.
)

endlocal

echo.
echo Running b3_test_opencv_cuda.exe...
b3_test_opencv_cuda.exe

echo.
echo Done.