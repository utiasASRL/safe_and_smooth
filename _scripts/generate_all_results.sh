#!/bin/bash

# run with fewer instances to make sure everything is working properly.
python3 _scripts/simulate_time.py --test
python3 _scripts/simulate_noise.py --test
python3 _scripts/evaluate_real.py --test

# generate final results
#python3 _scripts/simulate_time.py 
#python3 _scripts/simulate_noise.py 
#python3 _scripts/evaluate_real.py
