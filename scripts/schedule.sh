#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

uv run src/train.py trainer.max_epochs=5 logger=csv

uv run src/train.py trainer.max_epochs=10 logger=csv
