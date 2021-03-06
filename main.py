import streamlit as st
import numpy as np
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import matplotlib.pyplot as plt
import seaborn as sns
import utils
import classification

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


if 'C' not in st.session_state:
    st.session_state['C'] = classification.ClassifierLite()
classifier = st.session_state['C']

st.header("Upload your data")
file = st.file_uploader("upload file", type={"csv", "txt"})

# if upload successful,
if file is not None:
    classifier.df = pd.read_csv(file)
    st.write("Lựa chọn biến mong đợi (label, target) để xây dựng mô hình")
    classifier.features = list(classifier.df.columns)
    classifier.target_name = st.selectbox("Target", classifier.features)

    st.header("Select features")
    st.write("Lựa chọn yếu tố ảnh hưởng (features) để xây dựng mô hình máy học")
    with st.form("Machine Learning task"):
        classifier.select_all = st.checkbox("Select all features")
        if classifier.select_all:
            classifier.selected_features = list(
                set(classifier.features) - {classifier.target_name})
        else:
            classifier.selected_features = st.multiselect("Select features", list(
                set(classifier.features) - {classifier.target_name}))
        checked = st.checkbox("Display data profiling")
        submitted = st.form_submit_button("Xây dựng mô hình máy học")

    if classifier.selected_features and submitted:
        # visualize targets
        df = classifier.df.loc[:, [classifier.target_name] + classifier.selected_features]
        if checked:
            st.header("Display profile của dữ liệu")
            pr = df.profile_report()
            st_profile_report(pr)

        # feature engineering
        X, y = utils.feature_engineering(df, target_name=classifier.target_name)

        # lightgbm
        model, acc_score = classification.auto_lightgbm(X,y)
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


