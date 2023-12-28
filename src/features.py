import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import os
from fancyimpute import IterativeImputer

def filling_na(data_path):
    data = pd.read_csv(data_path)
    imputer = IterativeImputer(max_iter=10, random_state=0)
    # Fit and transform the DataFrame to impute missing values
    imputed_data = imputer.fit_transform(data)
    # Convert the imputed data back to a DataFrame
    df_imputed_using_iterativeimputer = pd.DataFrame(imputed_data, columns=data.columns)
    return df_imputed_using_iterativeimputer


if __name__ =="__main__":
    param_yaml_path = "params.yaml"
    with open(param_yaml_path) as yaml_file:
        param_yaml = yaml.safe_load(yaml_file)
    
    train_data_path = os.path.join(param_yaml["split"]["dir"], param_yaml["split"]["train_file"])
    final_train_data = filling_na(data_path = train_data_path)

    os.makedirs(param_yaml["process"]["dir"], exist_ok=True)
    final_train_data_path = os.path.join(param_yaml["process"]["dir"], param_yaml["process"]["train_file"])
    final_train_data.to_csv(final_train_data_path, index=False)

    test_data_path = os.path.join(param_yaml["split"]["dir"], param_yaml["split"]["test_file"])
    final_test_data = filling_na(data_path = test_data_path)
    final_test_data_path = os.path.join(param_yaml["process"]["dir"], param_yaml["process"]["test_file"])
    final_test_data.to_csv(final_test_data_path, index=False)