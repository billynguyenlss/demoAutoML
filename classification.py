import streamlit as st
import numpy as np
import pandas as pd
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDClassifier, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import utils


def auto_lightgbm(X,y):
    params = {
        'metric': 'binary_logloss',
        'n_estimators': 10000,
        'objective': 'binary',
        'random_state': 2021,
        'learning_rate': 0.02,
        'min_child_samples': 150,
        'reg_alpha': 3e-5,
        'reg_lambda': 9e-2,
        'num_leaves': 20,
        'max_depth': 16,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'subsample_freq': 2,
        'max_bin': 240
    }

    oof = np.zeros(X.shape[0])
    # preds = 0

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2021)
    model = None

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        # st.progress(fold)
        print(f"===== FOLD {fold} =====")

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            early_stopping_rounds=100,
            verbose=500
        )

        oof[valid_idx] = model.predict(X_valid)
        # preds += model.predict(X_test, num_iteration=model.best_iteration_) / skf.n_splits

        acc_score = accuracy_score(y_valid, np.where(oof[valid_idx] > 0.5, 1, 0))
        print(f"===== ACCURACY SCORE {acc_score} =====\n")

    acc_score = accuracy_score(y, np.where(oof > 0.5, 1, 0))
    print(f"===== ACCURACY SCORE {acc_score} =====")

    return model, acc_score

class ClassifierLite:
    df = None
    features = None
    target_name = None
    selected_features = None
    select_all = False


class Classifier:
    def __init__(self):
        self.df = None
        self.features = None
        self.target_name = None
        self.selected_features = None
        self.checked = False
        self.submitted = False

    def load_data(self, file_path):
        self.df = pd.read_csv(file_path)
        self.features = list(self.df.columns)

    def select_features(self):
        self.target_name = st.selectbox("Target", self.features)
        st.write(self.target_name)
        # with st.form("Machine Learning task"):
        #     select_all = st.checkbox("Select all features")
        #     if select_all:
        #         self.selected_features = list(set(self.features) - {self.target_name})
        #     else:
        #         self.selected_features = st.multiselect("Select features", list(
        #             set(self.features) - {self.target_name}))
        #     self.checked = st.checkbox("Display data profiling")
        #     self.submitted = st.form_submit_button("Xây dựng mô hình máy học")

    def display_pandas_profiling(self):
        if self.selected_features and self.submitted:
            # visualize targets
            df = self.df.loc[:, [self.target_name] + self.selected_features]
            if self.checked:
                st.header("Display profile của dữ liệu")
                pr = df.profile_report()
                st_profile_report(pr)

    def training(self):
        # feature engineering
        X, y = utils.feature_engineering(self.df, target_name=st.session_state['target_name'])

        # lightgbm
        model, acc_score = auto_lightgbm(X, y)
        return model, acc_score

    def performance(self, model, acc_score):
        # show model performance
        st.header("Model performance")
        st.write("Độ chính xác (accuracy) của mô hình học máy là: ")
        st.write(acc_score)

        # show feature importance
        st.write("Các yếu tố có ảnh hưởng nhiều nhất đến kết quả của mô hình học máy")
        feature_imp = pd.DataFrame(
            sorted(zip(model.feature_importances_, model.feature_name_)),
            columns=['Value', 'Feature'])
        fig, ax = plt.subplots()
        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False), ax=ax)
        ax.set_title("Feature importance")
        st.pyplot(fig)

    def download_model(self):
        pass

    def run(self):
        self.selected_features()
        self.display_pandas_profiling()
        model, acc_score = self.training()
        self.performance(model, acc_score)
        self.download_model()
