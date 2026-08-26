@echo off
rem OceanEmbed: stop backend + frontend dev servers.
rem Optional args pass through, e.g.:  bin\stop.cmd -ApiPort 8017 -WebPort 5199
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop.ps1" %*
