@echo off
setlocal
.venv\Scripts\python.exe transcription_client.py --host localhost --port 9100 --language foobar
pause

