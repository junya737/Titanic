import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import cross_validate


def add_one_hot(df, column_name):
    """ Add one hot vector

    Args:
        df (pd dataframe):

    Returns:
        _type_: df
    """
    if (column_name in df.columns):
        one_hot_sex = pd.get_dummies(df[column_name], prefix=column_name)
        df = pd.concat([df, one_hot_sex], axis=1)
        df.pop(column_name)
        if (column_name == "Sex"):
            df.pop("Sex_male")
    return df


def nan_to_mean(df):
    df = df.fillna(df.mean())
    return df


def save_kaggle_prediction(prediction, Id, target_col_name="target",
                           Id_col_name="Id"):
    # my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む
    my_solution = pd.DataFrame(prediction, Id, columns=[target_col_name])
    dt_now = datetime.datetime.now()
    # my_tree_one.csvとして書き出し
    my_solution.to_csv("../data/result/prediction_" +
                       dt_now.strftime('%Y%m%d_%H%M%S') + ".csv",
                       index_label=[Id_col_name])


def get_score_StratifiedKFold_cv(clf, x, y, n_splits, scoring, shuffle=True):
    """層化抽出KfoldCVのスコアを取得

    Args:
        clf (class): model
        x (np_array): 
        y (np_array): 
        n_splits (int): 
        scoring (str): 
        shuffle (bool): Defaults to True.

    Returns:
        float: score
    """
    kf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=shuffle, random_state=42)
    result = cross_validate(clf, x, y, cv=kf, scoring=scoring)

    return np.mean(result['test_score'])
