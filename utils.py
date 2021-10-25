import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def label_encoder(c):
    lc = LabelEncoder()
    return lc.fit_transform(c)


def feature_engineering(df, target_name=None):
    # engineering features
    cat_cols = [i for i in df.columns if df.dtypes[i] == 'object']
    num_cols = list(set(df.columns) - set(cat_cols))

    ## fill na value
    for col in num_cols:
        mean_value = df[col].mean()
        df[col].fillna(mean_value)
    for col in cat_cols:
        df[col].fillna("NA")

    ## encoding features
    label_encoded_df = df[cat_cols].apply(label_encoder)
    numerical_df = df[num_cols]

    X = pd.concat([label_encoded_df, numerical_df], axis=1)
    if target_name:
        y = X.pop(target_name)
        return X, y
    return X