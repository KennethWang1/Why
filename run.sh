#!/bin/bash

pkill -f python > /dev/null 2>&1

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate

pip install -r requirements.txt

nohup python bot.py > bot.log 2>&1 &