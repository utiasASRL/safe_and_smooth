#!/bin/bash
sudo docker build -t safe .
sudo docker rm -f timing; sudo docker run -itd --name "timing" --volume $(pwd):/safe safe python3 simulate_time.py --logging
sudo docker rm -f simulation; sudo docker run -itd --name "simulation" --volume $(pwd):/safe safe python3 simulate_noise.py --logging
sudo docker rm -f real; sudo docker run -itd --name "real" --volume $(pwd):/safe safe python3 evaluate_real.py --logging
