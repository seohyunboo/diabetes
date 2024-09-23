import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import numpy as np
from pages.__pycache__.function import split_and_convert_data, get_clf_eval, plot_roc_curve, plot_learning_curve_accuracy, confusion

# 모델 불러오기
model_total = joblib.load('data/pred.pkl')
model_nonsampling = joblib.load("data/pred_nonsampling.pkl")

# CSV 파일에서 데이터 로드
X = pd.read_csv('data/X.csv')
y = pd.read_csv('data/y.csv')
X_nonsampling = pd.read_csv("data/X_nonsampling.csv")
y_nonsampling = pd.read_csv("data/y_nonsampling.csv")


# 페이지 설정
st.set_page_config(layout="wide")
st.header(':pill:  당뇨병 예측 모델')
con1, con2 = st.columns([0.9,0.1])
with con1:
    with st.container():
        tab3, tab1, tab2= st.tabs(["개발 환경", "최종 모델 소개","샘플링 유무 성능 비교"])

    with tab3:
        st.subheader(':small_blue_diamond: 사용 기술 및 개발 환경')
        st.image('data/자료.png')

    with tab1:
        st.subheader(':small_blue_diamond: 이상치 제거 : 3-sigma')
        with st.expander("Details",expanded=True):
            st.image('data/이상치.png', width=600)
            st.write('''
                     - 3시그마(3-sigma) 방법은 데이터에서 이상치를 탐지하고 제거하는 데 사용되는 간단한 통계적 방법입니다
                     - 이 방법은 데이터가 정규분포를 따른다고 가정할 때, **평균과 표준편차를 기준**으로 이상치를 식별합니다
                        ''')
        st.subheader(':small_blue_diamond: 샘플링 : UnderSambling')
        with st.expander("Details",expanded=True):
            st.image('data/언더샘플링.png', width=600)
            st.write('''
                     - Under-sampling은 데이터의 불균형 문제를 해결하기 위한 기법으로, 주로 클래스 불균형이 있는 데이터셋에서 사용됩니다
                     - **정상 샘플 수가 당뇨 환자의 샘플 수에 비해 월등히 많아** 다수 클래스의 샘플을 줄여서 **데이터의 균형**을 맞추고자 했습니다
                     - 저희는 **재현율을 중요한 지표로 삼아 당뇨인 사람을 더 잘 예측하고자 Under-sampling을 사용**했습니다
                        ''')

    with tab2:
        # 페이지 상태 관리
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'intro'


        # 소개 페이지
        if st.session_state.current_page == 'intro':

            st.session_state.model_choice = st.selectbox("모델 선택", ["최종 모델", "샘플링 안한 xgboost"])

            if st.button("성능 보기"):
                st.session_state.current_page = 'next'


        # 모델 성능 페이지
        elif st.session_state.current_page == 'next':
            if st.session_state.model_choice == "최종 모델":
                model = model_total
                st.header("최종 모델 성능")
                X_train, X_test, y_train, y_test = split_and_convert_data(X, y, test_size=0.2, random_state=42)

                # 예측
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]

                col1, col2 = st.columns(2)

                # 성능, 혼동행렬
                with col1:
                    get_clf_eval(y_test, y_pred, y_proba)

                with col2:
                    confusion(y_test, y_pred)

                col3, col4 = st.columns(2)

                # 좌측 컬럼에 ROC Curve 배치
                with col3:
                    plot_roc_curve(y_test, y_proba)

                # 우측 컬럼에 Learning Curve 배치
                with col4:
                    with st.spinner("최종 모델의 성능을 계산하는 중..."):
                        plot_learning_curve_accuracy(model, X_train, y_train)
                
            

            if st.session_state.model_choice == "샘플링 안한 xgboost":
                model = model_nonsampling
                st.header("샘플링 안한 XGBoost 모델 성능")
                X_train, X_test, y_train, y_test = split_and_convert_data(X_nonsampling, y_nonsampling, test_size=0.2, random_state=42)

                # 예측
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]

                col1, col2 = st.columns(2)

                # 성능, 혼동행렬
                with col1:
                    get_clf_eval(y_test, y_pred, y_proba)

                with col2:
                    confusion(y_test, y_pred)

                # 화면을 2개의 컬럼으로 나누기
                col3, col4 = st.columns(2)

                # 좌측 컬럼에 ROC Curve 배치
                with col3:
                    plot_roc_curve(y_test, y_proba)

                # 우측 컬럼에 Learning Curve 배치
                with col4:
                    with st.spinner("샘플링 안한 xgboost 모델의 성능을 계산하는 중..."):
                        plot_learning_curve_accuracy(model, X_train, y_train)


            #다시 소개 페이지로 돌아가는 버튼
            if st.button("돌아가기"):
                st.session_state.current_page = 'intro'
