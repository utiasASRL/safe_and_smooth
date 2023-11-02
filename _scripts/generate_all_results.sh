#!/bin/bash
#python3 _scripts/simulate_time.py --resultdir="_results"
#python3 _scripts/simulate_noise.py --resultdir="_results"
python3 _scripts/evaluate_real.py --resultdir="_results"
python3 _scripts/plot_results.py --resultdir="_results" --plotdir="_plots"
