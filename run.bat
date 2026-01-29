taskkill /IM python.exe /F >nul 2>&1
watchmedo auto-restart --directory="." --pattern="*.py" --recursive -- .\.venv\Scripts\python.exe bot.py