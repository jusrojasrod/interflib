@ECHO OFF
REM Build the documentation: make.bat html (notebooks executed) or make.bat quick (not executed).
pushd %~dp0
if "%SPHINXBUILD%" == "" set SPHINXBUILD=sphinx-build

if "%1" == "quick" (
	set NB_EXECUTION_MODE=off
	%SPHINXBUILD% -b html . _build\html
	goto end
)
if "%1" == "clean" (
	rmdir /s /q _build api\generated notebooks\_generated 2>NUL
	goto end
)
%SPHINXBUILD% -W --keep-going -b html . _build\html

:end
popd
