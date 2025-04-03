setlocal

SET MAYA_VERSION=2024
REM "vs" "ninja"
REM use VS for the debugger, otherwise use NINJA
REM Until I figure out how to debug using nvim
SET BACKEND=ninja
REM "debug" "debugoptimized" "release"
SET BUILDTYPE=release
SET BUILDDIR=mayabuild_%BUILDTYPE%_%MAYA_VERSION%_%BACKEND%

if not exist %BUILDDIR%\ (
    meson setup %BUILDDIR% -Dmaya:maya_version=%MAYA_VERSION% --buildtype %BUILDTYPE% --vsenv --backend %BACKEND%
)

if exist %BUILDDIR%\ (
    meson compile -j 32 -C %BUILDDIR%
    REM Make sure to skip subprojects with eigen
    REM otherwise it'll try to install its headers
    meson install --skip-subprojects -C %BUILDDIR%
)

pause
