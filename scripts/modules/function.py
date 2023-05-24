import pandas as pd


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
