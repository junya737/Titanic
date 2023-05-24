import pandas as pd
import datetime


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
