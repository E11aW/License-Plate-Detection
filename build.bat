@echo off
echo Building License Plate Detection project...

if not exist build mkdir build

cd build

cmake .. -DOpenCV_DIR=C:\opencv\build

if errorlevel 1 (
    echo CMake configuration failed.
    pause
    exit /b 1
)

cmake --build . --config Debug

if errorlevel 1 (
    echo Build failed.
    pause
    exit /b 1
)

echo Build completed successfully.
pause