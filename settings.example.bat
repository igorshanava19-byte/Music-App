@echo off
rem Optional settings for Music App. setup.bat copies this file to settings.bat;
rem edit settings.bat (remove "rem " in front of a line to turn it on).

rem If YouTube says "Sign in to confirm you're not a bot", let the app use the
rem YouTube cookies from a browser you're signed in with (chrome, edge, firefox, brave):
rem set MUSICAPP_COOKIES_FROM_BROWSER=chrome

rem Password for the live link (required when the app is reachable from the internet):
rem set MUSICAPP_PASSWORD=choose-a-long-password

rem Use a different port (default 8000):
rem set MUSICAPP_PORT=8000

rem Store songs somewhere else (default: the "data" folder next to this file):
rem set MUSICAPP_DATA_DIR=E:\Music App Data

rem Force CPU processing even if you have a GPU:
rem set MUSICAPP_DEVICE=cpu
