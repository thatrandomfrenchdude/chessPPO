#!/usr/bin/env bash

if [ -z "$VIRTUAL_ENV" ]; then
  echo "Virtual environment not detected."
  if [ -d "chess-ppo-venv" ]; then
    echo "Found chess-ppo-venv. Activating..."
    source chess-ppo-venv/bin/activate
else
    echo "chess-ppo-venv does not exist. Create it? (y/n)"
    read answer
    if [ "$answer" != "y" ]; then
        echo "Exiting..."
        exit 1
    fi
    echo "Creating virtual environment..."
    python3.12 -m venv chess-ppo-venv
    source chess-ppo-venv/bin/activate
    echo "Installing requirements..."
    pip install -r requirements.txt
  fi
fi

# run the code
echo "Running chessPPO.py..."
python main.py

# run the cleanup script
echo "Running cleanup.sh..."
bash scripts/cleanup_v1.sh