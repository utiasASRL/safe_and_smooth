#!/bin/bash
python3 _scripts/simulate_time.py --logging
python3 _scripts/simulate_noise.py --logging
python3 _scripts/evaluate_real.py --logging
