import shap

import numpy as np
import xgboost as xgb

import numpy as np


def read_xgboost_data(option, model_id):
    """
    This function reads training and test data for computing SHAP value
    """
    train_fname = (
        "./data/SHC/model_fits/xgb_opt_"
        + option.__str__()
        + "/train_"
        + model_id.__str__()
        + ".csv"
    )

    test_fname = (
        "./data/SHC/model_fits/xgb_opt_"
        + option.__str__()
        + "/test_"
        + model_id.__str__()
        + ".csv"
    )
    dtrain = np.genfromtxt(train_fname, delimiter=",", skip_header=1)
    dtest = np.genfromtxt(test_fname, delimiter=",", skip_header=1)

    return dtrain, dtest


def read_xgboost_model(option, model_id):
    """
    This function reads trained xgboost models;
    The solution from the following link is used:
    https://github.com/slundberg/shap/issues/1215#issue-619973736
    https://github.com/slundberg/shap/issues/1215#issuecomment-641102855
    """
    model_fname = (
        "./data/SHC/model_fits/xgb_opt_"
        + option.__str__()
        + "/model_"
        + model_id.__str__()
        + ".model"
    )

    booster = xgb.Booster()
    booster.load_model(model_fname)

    model_bytearray = booster.save_raw()[4:]
    booster.save_raw = lambda: model_bytearray

    return booster


def save_shap(option, model_id, backgroud_portion):
    """
    This function computes SHAP value and save it
    """

    shap_train_fname = (
        "./data/SHC/model_fits/xgb_opt_"
        + option.__str__()
        + "/shap_train_"
        + model_id.__str__()
        + ".csv"
    )

    shap_test_fname = (
        "./data/SHC/model_fits/xgb_opt_"
        + option.__str__()
        + "/shap_test_"
        + model_id.__str__()
        + ".csv"
    )

    # read dataset
    dtrain, dtest = read_xgboost_data(option, model_id)

    # read model
    booster = read_xgboost_model(option, model_id)

    # set up TreeExplainer
    np.random.seed(option * model_id)
    random_indices = np.random.choice(
        dtrain.shape[0], size=round(dtrain.shape[0] * backgroud_portion), replace=False
    )
    backgroud_data = dtrain[random_indices, :]

    if backgroud_portion == 1:
        backgroud_data = dtrain

    explainer = shap.TreeExplainer(
        booster, feature_perturbation="interventional", data=backgroud_data
    )

    train_shap_values = explainer.shap_values(dtrain, check_additivity=False)
    np.savetxt(shap_train_fname, train_shap_values, delimiter=",")

    test_shap_values = explainer.shap_values(dtest, check_additivity=False)
    np.savetxt(shap_test_fname, test_shap_values, delimiter=",")

    return 0


eval_grid = np.genfromtxt(
    "./data/SHC/model_fits/optimal_candidate_id.csv",
    delimiter=",",
    skip_header=1,
    dtype="int",
)

for i in eval_grid:
    option, model_id = i
    save_shap(option, model_id, backgroud_portion=1)
