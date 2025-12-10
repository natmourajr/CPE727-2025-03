#!/bin/bash
#
#

# Training and evaluating CAE with # of epochs = 4, 6, 8, 10, 20, 30, 40, and 50.
# Results will be saved in out/ directory.
#
python3 CAE.py --epochs=4
python3 CAE.py --epochs=6
python3 CAE.py --epochs=8
python3 CAE.py --epochs=10
python3 CAE.py --epochs=20
python3 CAE.py --epochs=30
python3 CAE.py --epochs=40
python3 CAE.py --epochs=50

# Create all the figures.
# Figures will be saved in plots/ directory.
python3 CAE_result.py

