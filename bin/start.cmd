@echo off
rem OceanEmbed: start backend + frontend dev servers.
rem Optional args pass through, e.g.:  bin\start.cmd -ApiPort 8017 -WebPort 5199
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
