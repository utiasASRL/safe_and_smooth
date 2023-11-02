#!/bin/bash

# run with fewer instances to make sure everything is working properly.
#python3 _scripts/simulate_time.py --test --resultdir="_results"
#python3 _scripts/simulate_noise.py --test --resultdir="_results"
#python3 _scripts/evaluate_real.py --test --resultdir="_results"
python3 _scripts/plot_results.py --resultdir="_results_server" --plotdir="_plots_test/"

# generate final results
#python3 _scripts/simulate_time.py --resultdir="_results_final/"
#python3 _scripts/simulate_noise.py --resultdir="_results_final/"
#python3 _scripts/evaluate_real.py --resultdir="_results_final/"
#python3 _scripts/plot_results.py --resultdir="_results_final/" --plotdir="_plots/"
