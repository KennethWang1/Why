taskkill /IM python.exe /F >nul 2>&1
pip install -r requirements.txt
.\.venv\Scripts\python.exe bot.py