#!/bin/bash

# run with fewer instances to make sure everything is working properly.
python3 _scripts/simulate_time.py --test --resultdir="_results_test"
python3 _scripts/simulate_noise.py --test --resultdir="_results_test"
python3 _scripts/evaluate_real.py --test --resultdir="_results_test"
python3 _scripts/plot_results.py --resultdir="_results_test" --plotdir="_plots_test"
