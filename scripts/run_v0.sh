#!/usr/bin/env bash

if [ -z "$VIRTUAL_ENV" ]; then
  echo "No virtual environment detected."
  if [ -d "chess-ppo-venv" ]; then
    echo "Found chess-ppo-venv. Activating..."
    source chess-ppo-venv/bin/activate
  else
    echo "chess-ppo-venv does not exist. Creating it..."
    python3.12 -m venv chess-ppo-venv
    source chess-ppo-venv/bin/activate
    echo "Installing requirements..."
    pip install -r requirements.txt
  fi
else
  echo "Virtual environment is already active."
fi

echo "Running chessPPO.py..."
python chessPPO.py
