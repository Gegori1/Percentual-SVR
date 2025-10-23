# %%
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.model_selection import train_test_split

from gcp_library.optimization_utils.optimizer_crossvalidation import CrossValidationOptimizer
from percentual_svr_package.svr_ls_rmspe import svr_ls_rmspe
# from percentual_svr_package.svr_mape import svr_original_mape



# %% load data

diabetes = pd.read_csv("../data/diabetes.csv", delimiter="\t")
X_train, X_test, y_train, y_test = train_test_split(diabetes.drop(columns="Y"), diabetes.Y, test_size=0.225, random_state=42)
X_train, y_train = X_train.values, y_train.values
# %% parameters

logs_name = "svr_qcqp"
load_previous = False
log_file = f"{logs_name}.jsonl"

params = {
    'C': (1e-3, 1e6, "float"),
    'gamma': (1e-3, 1e6, "float"),
    'epsilon': (1e-3, 1e2, "float"),
    'kernel': ("rbf", 
        None, 
        "stationary"
    )
}

n_iter = 200

# %% Optimization

optimizer = CrossValidationOptimizer(
    X_train, y_train, 
    svr_ls_rmspe, 
    mape,
    save_path=log_file,
    cloud_name="",
    cloud_bucket="",
    cloud_key="",
    param_distributions=params,
    n_folds=5,
    standardize=True,
    n_jobs=-1,
    upload_to_cloud=False
)

# Create a study object and optimize the objective function
study = optimizer.optimize(n_trials=n_iter)

