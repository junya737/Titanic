import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
import lightgbm as lgb


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
        x (np_array): fitting
        y (np_array): answer
        n_splits (int): num of split
        scoring (str): the wey of scoring
        shuffle (bool): Defaults to True.

    Returns:
        float: score
    """
    kf = StratifiedKFold(
        n_splits=n_splits, shuffle=shuffle, random_state=42)
    result = cross_validate(clf, x, y, cv=kf, scoring=scoring)

    return np.mean(result['test_score'])


def cross_val_score_lgbm_earlystopping(clf, x, y, cv, stopping_rounds=50,
                                       scoring="accuracy",
                                       eval_metric="logloss"):
    """Get cross validation score using LightGBM with early stopping

    Args:
        clf (class): model
        x (np_array): features
        y (np_array): labels
        cv (class): cross val
        stopping_rounds (int, optional): _description_. Defaults to 50.
        scoring (str, optional): score metric. Defaults to "accuracy".
        eval_metric (str, optional): metric for early stopping.
            Defaults to "logloss".

    Returns:
        float: score
    """

    # クロスバリデーションのデータ分割
    scores = []
    for _, (train_idx, val_idx) in enumerate(cv.split(x, y)):
        x_train = x[train_idx]
        x_val = x[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        verbose_eval = 0  # この数字を1にすると学習時のスコア推移がコマンドライン表示される
        clf.fit(x_train, y_train,
                # early_stoppingの評価指標(学習用の'metric'パラメータにも同じ指標が自動入力される)
                eval_metric=eval_metric,
                eval_set=[(x_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=stopping_rounds,
                                              verbose=False),
                           # early_stopping用コールバック関数
                           lgb.log_evaluation(verbose_eval)]
                # コマンドライン出力用コールバック関数
                )

        y_pred = clf.predict(x_val)
        score = accuracy_score(y_true=y_val, y_pred=y_pred)
        scores.append(score)

    return np.mean(scores)
