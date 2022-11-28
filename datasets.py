import yaml

import numpy as np
import pandas as pd

def read_anchors(fname, use_anchors=None):
    """
    :param use_anchors: list of indices of anchors to use
    """
    with open(fname) as f:
        anchor_dict = yaml.load(f, yaml.FullLoader)
    anchor_names = np.array([a_name for a_name in anchor_dict.keys() if a_name.startswith("anchor_")])
    if use_anchors is not None:
        anchor_names = anchor_names[use_anchors]
    anchors = np.array([anchor_dict[a_name] for a_name in anchor_names], dtype=float)
    return anchors, anchor_names

def read_dataset(fname="trial0"):
    data_uwb = pd.read_csv(f"_data/{fname}/uwb.txt", 
                       names=["timestamp", "tag", "anchor", "distance", "std"])
    data_gt = pd.read_csv(f"_data/{fname}/vicon.txt", 
                          names=["timestamp", "x", "y", "z", "roll", "pitch", "yaw"])
    data_uwb["times"] = (data_uwb.timestamp - data_uwb.iloc[0].timestamp)
    data_gt["times"] = (data_gt.timestamp - data_gt.iloc[0].timestamp)
    return data_gt, data_uwb
