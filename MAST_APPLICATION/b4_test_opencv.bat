@echo off
setlocal

echo Compiling b4_test_opencv.cpp...

rem Check if source file exists
if not exist b4_test_opencv.cpp (
    echo ERROR: Source file b4_test_opencv.cpp not found!
    exit /b 1
)

rem Set up paths - keep these as short as possible to avoid command line length limitations
set OPENCV_INC=C:\opencv\include
set OPENCV_LIB=C:\opencv\x64\vc17\lib
set TBB_DLL=C:\Program Files (x86)\Intel\oneAPI\tbb\latest\redist\intel64\vc14

rem Compile the program - added opencv_videoio481.lib for VideoCapture support
cl b4_test_opencv.cpp /Fe:b4_test_opencv.exe /I "%OPENCV_INC%" /EHsc /MD /link /LIBPATH:"%OPENCV_LIB%" ^
   opencv_core481.lib ^
   opencv_highgui481.lib ^
   opencv_imgcodecs481.lib ^
   opencv_imgproc481.lib ^
   opencv_cudaarithm481.lib ^
   opencv_cudawarping481.lib ^
   opencv_videoio481.lib

if %ERRORLEVEL% NEQ 0 (
    echo Compilation failed!
    exit /b %ERRORLEVEL%
) else (
    echo Compilation completed successfully.
)

rem Copy the TBB DLL to the current directory
if exist "%TBB_DLL%\tbb12.dll" (
    echo Copying TBB DLL...
    copy "%TBB_DLL%\tbb12.dll" . >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo TBB DLL copied successfully.
    ) else (
        echo WARNING: Failed to copy TBB DLL.
    )
)

echo.
echo Running b4_test_opencv.exe...
b4_test_opencv.exe

echo.
echo Done.