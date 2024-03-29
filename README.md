# Safe and Smooth: Certified Continuous-Time Range-Only Localization

[![Python Package using Conda](https://github.com/utiasASRL/safe_and_smooth/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/utiasASRL/safe_and_smooth/actions/workflows/python-package-conda.yml)

This repository contains the code to reproduce all results of the paper:

````
F. Dümbgen, C. Holmes and T. D. Barfoot, "Safe and Smooth: Certified Continuous-Time 
Range-Only Localization," in IEEE Robotics and Automation Letters, vol. 8, no. 2, 
pp. 1117-1124, Feb. 2023, doi: 10.1109/LRA.2022.3233232.
````

A pre-print is available at [https://arxiv.org/abs/2209.04266](https://arxiv.org/abs/2209.04266).

## Installation

This code was last tested with Ubuntu 20.04.1, using Python 3.10.3.

### Local install

Make sure to do a recursive clone:
```
git clone --recursive git@github.com:utiasASRL/safe_and_smooth
```

All requirements can be installed by running
```
conda env create -f environment.yml
```

To check that the installation was successful, run
```
conda activate safeandsmooth
pip install pytest
pytest .
```
You can also check that you can generate some toy example results by running
```
_scripts/generate_test_results.sh
```
and then checking the output created in `_plots_test`. 

### Docker install

You can also use docker to run the code in this repository. To create the docker image, you can run
```
make safe-build
```
and to test that installation was successful, you can run
```
make safe-test
```
which runs `pytest` and generates test data. 



## Generate results

There are three types of results reported in the paper:

- Noise study: Run `_scripts/simulate_noise.py` to generate the simulation study (Figures 4 and 7 (appendix)). 
- Timing study:  Run `_scripts/simulate_time.py` to generate the runtime comparison (Figure 5)
- Real data: Run `_scripts/evaluate_real.py` to evaluate the real dataset (Figures 1, 5 and 6). 

You can generate all results by running (this will take a while)
```
_scripts/generate_all_results.sh
```
After generating, all data can be evaluated, and new figures created, by running `python _scripts/plot_results.py`. For more evaluations of the real dataset, refer to the notebook `_notebooks/DatasetEvaluation.ipynb` (you may need to run `pip install -r requirements.txt` for additional plotting libraries). 

### Docker instructions

If you are using docker, you can use this one-liner to run the above command in a docker container:
```
make safe-run
```

## Code references

The code refers to the following papers:

- [1] F. Dümbgen, C. Holmes and T. D. Barfoot, "Safe and Smooth: Certified Continuous-Time Range-Only Localization," in IEEE Robotics and Automation Letters, vol. 8, no. 2, pp. 1117-1124, Feb. 2023. https://doi.org/10.1109/LRA.2022.3233232.
- [2] Barfoot, Tim, Chi Hay Tong, and Simo Sarkka. “Batch Continuous-Time Trajectory Estimation as Exactly Sparse Gaussian Process Regression,” 2014. https://doi.org/10.15607/RSS.2014.X.001.
- [3] Barfoot, Timothy D. State Estimation for Robotics. Cambridge University Press, 2017. https://doi.org/10.1017/9781316671528.
