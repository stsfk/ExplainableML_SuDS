import shap

import numpy as np
import xgboost as xgb

import numpy as np


def read_xgboost_data(iter):
    """
    This function reads training and test data for computing SHAP value
    """
    dtrain = xgb.DMatrix(f"./data/WS/model_fits/iter={iter}opt=1dtrain.data")
    dtest = xgb.DMatrix(f"./data/WS/model_fits/iter={iter}opt=1dtest.data")

    return dtrain, dtest


def read_xgboost_model(iter):
    """
    This function reads trained xgboost models;
    The solution from the following link is used:
    https://github.com/slundberg/shap/issues/1215#issue-619973736
    https://github.com/slundberg/shap/issues/1215#issuecomment-641102855
    """
    model_fname = f"./data/WS/model_fits/gof_iter={iter}opt=1.model"

    booster = xgb.Booster()
    booster.load_model(model_fname)

    model_bytearray = booster.save_raw()[4:]
    booster.save_raw = lambda: model_bytearray

    return booster


n_train = [29061, 31229, 31064, 32135, 31851]

for iter in range(5):
    iter = iter + 1
    all_data = np.genfromtxt(
        f"./data/WS/model_fits/iter={iter}opt=1_train_test.csv",
        delimiter=",",
        skip_header=1,
    )
    all_data = all_data[:, 1:]
    booster = read_xgboost_model(iter)

    backgroud_data = all_data[0 : n_train[iter - 1], :]

    explainer = shap.TreeExplainer(
        booster, feature_perturbation="interventional", data=backgroud_data
    )

    all_shap_values = explainer.shap_values(all_data)
    np.savetxt(
        f"./data/WS/model_fits/int_SHAP_iter{iter}.csv",
        all_shap_values,
        delimiter=",",
    )
