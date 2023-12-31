import os
import pandas as pd
import numpy as np
import pickle
import sys
import yaml
from sklearn.ensemble import RandomForestClassifier

def training(yaml_path):
    with open(yaml_path) as file:
        param_yaml= yaml.safe_load(file)
    final_train_data_path = os.path.join(param_yaml["process"]["dir"],param_yaml["process"]["train_file"])
    final_test_data_path = os.path.join(param_yaml["process"]["dir"], param_yaml["process"]["test_file"])

    random_state = param_yaml["base"]["random_state"]
    target = [param_yaml["base"]["target_col"]]

    train = pd.read_csv(final_train_data_path)
    test = pd.read_csv(final_test_data_path)

    y_train = train[target]
    y_test = test[target]

    X_train = train.drop(target, axis=1)
    X_test = test.drop(target, axis=1)

    random_state = param_yaml["base"]["random_state"]
    n_est = param_yaml["train"]["n_est"]

    rfc =RandomForestClassifier(random_state=random_state,n_estimators=n_est)
    rfc.fit(X_train, y_train.values.ravel())

    model_dir= param_yaml["model"]["dir"]
    os.makedirs(model_dir,exist_ok=True)
    with open (model_dir+"/model.pkl","wb") as f:
        pickle.dump(rfc,f)


if __name__== "__main__":
    training(yaml_path = "params.yaml")






#####original Code################
# params = yaml.safe_load(open("params.yaml"))["train"]

# if len(sys.argv) != 3:
#     sys.stderr.write("Arguments error. Usage:\n")
#     sys.stderr.write("\tpython train.py features model\n")
#     sys.exit(1)

# input = sys.argv[1]
# output = sys.argv[2]
# seed = params["seed"]
# n_est = params["n_est"]
# min_split = params["min_split"]

# with open(os.path.join(input, "train.pkl"), "rb") as fd:
#     matrix, _ = pickle.load(fd)

# labels = np.squeeze(matrix[:, 1].toarray())
# x = matrix[:, 2:]

# sys.stderr.write("Input matrix size {}\n".format(matrix.shape))
# sys.stderr.write("X matrix size {}\n".format(x.shape))
# sys.stderr.write("Y matrix size {}\n".format(labels.shape))

# clf = RandomForestClassifier(
#     n_estimators=n_est, min_samples_split=min_split, n_jobs=2, random_state=seed
# )

# clf.fit(x, labels)

# with open(output, "wb") as fd:
#     pickle.dump(clf, fd)
