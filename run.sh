taskkill /IM python.exe /F >nul 2>&1
source .venv/bin/activate 
pip install -r requirements.txt
python bot.py