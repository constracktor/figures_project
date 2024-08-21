#!/bin/bash
# create python enviroment
python3 -m venv plot_env
# activate enviroment
source plot_env/bin/activate
# install requirements
pip install --no-cache-dir -r docker/requirements.txt
# launch game
python3 plot.py
python3 plot_tiles.py
