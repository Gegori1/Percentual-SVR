# %%
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

from svr_qcqp_multi_kernel_l2 import svr_qcqp_multi_kernel_l2
from optimizer_crossvalidation import CrossValidationOptimizer
import argparse

# %% arguments



# %% load data

diabetes = pd.read_csv("diabetes.csv", delimiter="\t")
X_train, X_test, y_train, y_test = train_test_split(diabetes.drop(columns="Y"), diabetes.Y, test_size=0.225, random_state=42)
X_train, y_train = X_train.values, y_train.values
# %% parameters

logs_name = args.logs_file
load_previous = False
log_file = f"{logs_name}.jsonl"

params = {
    'C': (args.C_min, args.C_max, "float"),
    'epsilon': (args.epsilon_min, args.epsilon_max, "float"),
    'tau': (args.tau_min, args.tau_max, "float"),
    'kernel': ("rbf", 
        None, 
        "stationary"
    )
}

n_iter = args.n_iter

# %% Optimization

optimizer = CrossValidationOptimizer(
    X_train, y_train, 
    svr_qcqp_multi_kernel_l2, 
    mean_absolute_percentage_error,
    save_path=log_file,
    cloud_name=args.cloud_name,
    cloud_bucket=args.cloud_bucket,
    cloud_key=args.cloud_key,
    param_distributions=params,
    n_folds=args.n_splits,
    standardize=True,
    upload_cloud_rate=args.cloud_uploadrate
)

# Create a study object and optimize the objective function
study = optuna.create_study(direction='minimize')
study.optimize(optimizer, n_trials=n_iter, n_jobs=-1)

# Print the best hyperparameters and the best accuracy
print("Best hyperparameters: ", study.best_params)
print("Best accuracy: ", study.best_value)