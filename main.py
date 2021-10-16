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

def label_encoder(c):
    lc = LabelEncoder()
    return lc.fit_transform(c)

if 'df' not in st.session_state:
    st.session_state['df'] = None

st.header("Upload your data")
file = st.file_uploader("upload file", type={"csv", "txt"})

if file is not None:
    st.session_state['df'] = pd.read_csv(file)
    st.write("Lựa chọn biến mong đợi (label, target) để xây dựng mô hình")
    features = list(st.session_state['df'].columns)
    target_name = st.selectbox("Target", features)
    # X = st.session_state['df'].loc[:,list(set(features) - {target_name})]
    # Y = st.session_state['df'].loc[:,[target_name]]

    st.header("Select features")
    st.write("Lựa chọn yếu tố ảnh hưởng (features) để xây dựng mô hình máy học")
    with st.form("Machine Learning task"):
        selected_features = st.multiselect("Select features", list(set(features) - {target_name}))
        checked = st.checkbox("Display data profiling")
        submitted = st.form_submit_button("Xây dựng mô hình máy học")

    if selected_features and submitted:
        # visualize targets
        df = st.session_state['df'].loc[:, [target_name] + selected_features]
        if checked:
            st.header("Display profile của dữ liệu")
            st.session_state['pr'] = df.profile_report()
            st_profile_report(st.session_state['pr'])
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
        y = X.pop(target_name)

        # select ML model [random forest, logistic regression, lightgbm]
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

        for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
            st.progress(fold)
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
            #preds += model.predict(X_test, num_iteration=model.best_iteration_) / skf.n_splits

            acc_score = accuracy_score(y_valid, np.where(oof[valid_idx] > 0.5, 1, 0))
            print(f"===== ACCURACY SCORE {acc_score} =====\n")

        acc_score = accuracy_score(y, np.where(oof > 0.5, 1, 0))
        print(f"===== ACCURACY SCORE {acc_score} =====")
        # show model performance
        st.header("Model performance")
        st.write("Độ chính xác (accuracy) của mô hình học máy là: ")
        st.write(acc_score)

        # show feature importance
        st.write("Các yếu tố có ảnh hưởng nhiều nhất đến kết quả của mô hình học máy")
        feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_, model.feature_name_)), columns=['Value', 'Feature'])
        fig, ax = plt.subplots()
        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False), ax=ax)
        ax.set_title("Feature importance")
        st.pyplot(fig)
