@echo off
setlocal

set "ROOT=C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main"
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
set "OBJDIR=%ROOT%\build\gui_obj"
set "RLDIR=%ROOT%\raylib\raylib-6.0_win64_msvc16"

call "%VCVARS%" x64
if errorlevel 1 (
    echo ERROR: vcvarsall failed
    exit /b 1
)

echo Working dir: %CD%
if not exist "%ROOT%\src\gui.cpp" (
    echo ERROR: gui.cpp not found
    exit /b 1
)
if not exist "%OBJDIR%" mkdir "%OBJDIR%"

echo === Compiling game sources ===
cl.exe /O2 /EHsc /std:c++17 /utf-8 /nologo /c ^
    /I"%ROOT%\include" /I"%ROOT%\include\third_party" ^
    "%ROOT%\src\rng.cpp" ^
    /Fo"%OBJDIR%\rng.obj"
if errorlevel 1 goto :fail

cl.exe /O2 /EHsc /std:c++17 /utf-8 /nologo /c ^
    /I"%ROOT%\include" /I"%ROOT%\include\third_party" ^
    "%ROOT%\src\resources.cpp" ^
    /Fo"%OBJDIR%\resources.obj"
if errorlevel 1 goto :fail

cl.exe /O2 /EHsc /std:c++17 /utf-8 /nologo /c ^
    /I"%ROOT%\include" /I"%ROOT%\include\third_party" ^
    "%ROOT%\src\data.cpp" ^
    /Fo"%OBJDIR%\data.obj"
if errorlevel 1 goto :fail

cl.exe /O2 /EHsc /std:c++17 /utf-8 /nologo /c ^
    /I"%ROOT%\include" /I"%ROOT%\include\third_party" ^
    "%ROOT%\src\earth.cpp" ^
    /Fo"%OBJDIR%\earth.obj"
if errorlevel 1 goto :fail

cl.exe /O2 /EHsc /std:c++17 /utf-8 /nologo /c ^
    /I"%ROOT%\include" /I"%ROOT%\include\third_party" ^
    "%ROOT%\src\game.cpp" ^
    /Fo"%OBJDIR%\game.obj"
if errorlevel 1 goto :fail

cl.exe /O2 /EHsc /std:c++17 /utf-8 /nologo /c ^
    /I"%ROOT%\include" /I"%ROOT%\include\third_party" ^
    "%ROOT%\src\env.cpp" ^
    /Fo"%OBJDIR%\env.obj"
if errorlevel 1 goto :fail

echo === Compiling GUI ===
cl.exe /O2 /EHsc /std:c++17 /utf-8 /nologo /c ^
    /I"%ROOT%\include" /I"%ROOT%\include\third_party" /I"%RLDIR%\include" ^
    "%ROOT%\src\gui.cpp" ^
    /Fo"%OBJDIR%\gui.obj"
if errorlevel 1 goto :fail

echo === Linking ===
cl.exe /nologo ^
    "%OBJDIR%\gui.obj" ^
    "%OBJDIR%\rng.obj" ^
    "%OBJDIR%\resources.obj" ^
    "%OBJDIR%\data.obj" ^
    "%OBJDIR%\earth.obj" ^
    "%OBJDIR%\game.obj" ^
    "%OBJDIR%\env.obj" ^
    /Fe"%ROOT%\sakhalin_colony_gui.exe" ^
    /link /SUBSYSTEM:CONSOLE /NODEFAULTLIB:libcmt.lib ^
    "%RLDIR%\lib\raylib.lib" opengl32.lib winmm.lib gdi32.lib user32.lib shell32.lib ^
    ucrt.lib vcruntime.lib msvcrt.lib advapi32.lib ole32.lib oleaut32.lib
if errorlevel 1 goto :fail

echo Build OK: %ROOT%\sakhalin_colony_gui.exe
echo Copying raylib.dll...
copy "%RLDIR%\lib\raylib.dll" "%ROOT%\raylib.dll" >nul
echo Copying original icons (ui/assets -> assets)...
if not exist "%ROOT%\assets" mkdir "%ROOT%\assets"
xcopy /Y /I "%ROOT%\ui\assets\*.png" "%ROOT%\assets" >nul
exit /b 0

:fail
echo BUILD FAILED
exit /b 1
