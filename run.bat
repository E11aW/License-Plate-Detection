@echo off
echo Running License Plate Detection project...

cd /d "%~dp0"

if exist build\Debug\FinalProject.exe (
    build\Debug\FinalProject.exe
) else (
    echo Could not find build\Debug\FinalProject.exe
    echo Run build.bat first.
)

pause