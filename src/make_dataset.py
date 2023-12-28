import yaml
import os
import argparse
import pandas as pd
import numpy as np
import sklearn.model_selection import train_test_split

def data_load_split(yaml_path):
    with open(yaml_path) as file:
        param_yaml= yaml.load(file)
    load_data = param_yaml["data_source"]["local_path"]

    data_df =pd.read_csv(load_data)
    random_state =param_yaml["base"]['random_state']
    split_ratio = param_yaml["split"]["split_ratio"]

    train,test = train_test_split(
        data_df,
        test_size=split_ratio,
        random_state = random_state
        )
    
    os.makedirs(param_yaml["split"]["dir"], exist_ok=True)
    train_data_path = os.path.join(param_yaml["split"]["dir"], param_yaml["split"]["train_file"])

    train.to_csv(train_data_path,index=False)

    test_data_path = os.path.join(param_yaml["split"]["dir"], param_yaml["split"]["test_file"]) 
    test.to_csv(test_data_path,index=False)