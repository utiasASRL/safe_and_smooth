# Safe and Smooth: Certified Continuous-Time Range-Only Localization

This repository contains the code to reproduce all results of the paper:

```
Dümbgen, Frederike, Connor Holmes, and Timothy D. Barfoot. “Safe and Smooth: Certified Continuous-Time Range-Only Localization.”, arXiv:2209.04266 [cs.RO], Nov. 2022
```

A pre-print is available at [https://arxiv.org/abs/2209.04266](https://arxiv.org/abs/2209.04266).

## Installation

This code was written for Ubuntu 20.04.5, using Python 3.8.10.

### Local install
All requirements can be installed by running
```
pip install -r requirements.txt
```
To check that the installation is successful, run
```
pytest .
```

### Docker install
Alternatively, the provided Dockerfile can be used to avoid locally installing dependencies. To build the container, run
```
sudo docker build -t safe .
```

To check that the installation is successful, run
```
sudo docker run -it --volume $(pwd):/safe safe pytest .
```

Please report any installation issues. 

## Generate results

There are three types of results reported in the paper:

- Noise study: Run `simulate_noise.py` to generate the simulation study (Figures 4 and 7 (appendix)). 
- Timing study:  Run `simulate_time.py` to generate the runtime comparison (Figure 5)
- Real data: Run `evaluate_data.py` to evaluate the real dataset (Figures 1, 5 and 6). 

If you are using Docker, you can generate all results by running
```
_scripts/generate_all_results.sh
```
After generating, all data can be evaluated, and new figures created, using the jupyter notebook `SafeAndSmooth.ipynb`. For more evaluations of the real dataset, refer to the notebook `DatasetEvaluation.ipynb`. 

## Code references

The code refers to the following papers:
```
[1] Dümbgen, Frederike, Connor Holmes, and Timothy D. Barfoot. “Safe and Smooth: Certified Continuous-Time Range-Only Localization.”, arXiv:2209.04266 [cs.RO], Nov. 2022
[2] Barfoot, Tim, Chi Hay Tong, and Simo Sarkka. “Batch Continuous-Time Trajectory Estimation as Exactly Sparse Gaussian Process Regression,” 2014. https://doi.org/10.15607/RSS.2014.X.001.
[3] Barfoot, Timothy D. State Estimation for Robotics. Cambridge University Press, 2017. https://doi.org/10.1017/9781316671528.

```
