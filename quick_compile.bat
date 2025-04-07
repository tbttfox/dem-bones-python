SETLOCAL

SET PYTHON_VERSION=3.10
REM "vs" "ninja"
REM use VS for the debugger, otherwise use NINJA
REM Until I figure out how to debug using nvim
SET BACKEND=ninja
REM "debug" "debugoptimized" "release"
SET BUILDTYPE=release
SET BUILDDIR=pybuild_%BUILDTYPE%_Py%PYTHON_VERSION:.=%_%BACKEND%

SET MESON="C:\Program Files\Python%PYTHON_VERSION:.=%\Scripts\meson.exe"

IF NOT EXIST %BUILDDIR%\ (
    %MESON% setup %BUILDDIR% -Dpyversion=%PYTHON_VERSION% --buildtype %BUILDTYPE% --vsenv --backend %BACKEND%
)

IF EXIST %BUILDDIR%\ (
    %MESON% compile -C %BUILDDIR%
    REM Make sure to skip subprojects with eigen
    REM otherwise it'll try to install its headers
    %MESON% install --skip-subprojects -C %BUILDDIR%
)

PAUSE
