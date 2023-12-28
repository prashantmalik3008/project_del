import yaml
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def data_load_split(yaml_path):
    with open(yaml_path) as file:
        param_yaml = yaml.load(file)

    load_data = param_yaml["data_source"]["local_path"]

    data_df = pd.read_csv(load_data)
    random_state = param_yaml["base"]['random_state']
    split_ratio = param_yaml["split"]["split_ratio"]

    train, test = train_test_split(
        data_df,
        test_size=split_ratio,
        random_state=random_state
    )

    # Create the output directory if it doesn't exist
    os.makedirs(param_yaml["split"]["dir"], exist_ok=True)  

    train_data_path = os.path.join(param_yaml["split"]["dir"], param_yaml["split"]["train_file"])
    test_data_path = os.path.join(param_yaml["split"]["dir"], param_yaml["split"]["test_file"])

    train.to_csv(train_data_path, index=False)
    test.to_csv(test_data_path, index=False)

if __name__=="__main__":
    data_load_split("params.yaml")