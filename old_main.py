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


if 'df' not in st.session_state:
    st.session_state['df'] = None

if 'test' not in st.session_state:
    st.session_state['test'] = None

st.session_state['selected_features'] = None

st.header("Upload your data")

# upload data
file = st.file_uploader("upload file", type={"csv", "txt"})

# if upload successful,
if file is not None:
    # classifier = Classifier(file)
    # classifier.select_features()
    st.session_state['df'] = pd.read_csv(file)
    st.write("Lựa chọn biến mong đợi (label, target) để xây dựng mô hình")
    st.session_state['features'] = list(st.session_state['df'].columns)
    st.session_state['target_name'] = st.selectbox("Target", st.session_state['features'])
    # X = st.session_state['df'].loc[:,list(set(features) - {target_name})]
    # Y = st.session_state['df'].loc[:,[target_name]]

    st.header("Select features")
    st.write("Lựa chọn yếu tố ảnh hưởng (features) để xây dựng mô hình máy học")
    with st.form("Machine Learning task"):
        select_all = st.checkbox("Select all features")
        if select_all:
            st.session_state['selected_features'] = list(
                set(st.session_state['features']) - {st.session_state['target_name']})
        else:
            st.session_state['selected_features'] = st.multiselect("Select features", list(
                set(st.session_state['features']) - {st.session_state['target_name']}))
        checked = st.checkbox("Display data profiling")
        submitted = st.form_submit_button("Xây dựng mô hình máy học")

    if st.session_state['selected_features'] and submitted:
        # visualize targets
        df = st.session_state['df'].loc[:, [st.session_state['target_name']] + st.session_state['selected_features']]
        if checked:
            st.header("Display profile của dữ liệu")
            st.session_state['pr'] = df.profile_report()
            st_profile_report(st.session_state['pr'])

        # feature engineering
        X, y = utils.feature_engineering(df, target_name=st.session_state['target_name'])

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


