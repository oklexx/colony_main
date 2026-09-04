@echo off
setlocal
set "ROOT=C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main"
set "PY=C:\Python314\python.exe"

cd /d "%ROOT%\build\py"

echo === CMake Configure ===
%PY% -m cmake "%ROOT%" -G "Visual Studio 18 2026" -A x64
if errorlevel 1 goto :fail

echo === CMake Build ===
%PY% -m cmake --build . --config Release
if errorlevel 1 goto :fail

echo === Copy pyd ===
copy /y "%ROOT%\python\Release\colony_cpp.pyd" "%ROOT%\python\colony_cpp.pyd"
if errorlevel 1 goto :fail

echo BUILD OK
exit /b 0

:fail
echo BUILD FAILED
exit /b 1
