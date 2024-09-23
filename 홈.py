import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
# from function import go_to_page, split_and_convert_data, get_clf_eval
# from function import remove_outliers, plot_roc_curve, plot_learning_curve_accuracy

# 페이지 설정
st.set_page_config(layout="wide")


con1, empty1, empty2 = st.columns([0.7, 0.2, 0.1])

with con1:
    st.header("국건영 데이터를 이용한 당뇨병 예측 모델 :hospital::pill:")

with empty1:
    with st.expander(":black_medium_square:  **당뇨팀**",expanded=True):
        st.write('''
            고은비, 박서현, 정영성    
                ''', unsafe_allow_html=True)

with empty2:
    st.write('')

st.write(' ')

con3, empty3, con4 = st.columns([0.25, 0.05, 0.7])

with con3:   
    st.write('''
            :white_check_mark:  **개요**<br>
             당뇨의 정의 | 원인 | 증상 | 진단
             
            :white_check_mark:  **데이터 소개**<br>
             국건영 데이터 소개 | 프로젝트 목표
             
            :white_check_mark:  **모델링**<br>
             최종 모델 소개 | 샘플링 유무 성능 비교
             
            :white_check_mark:  **예측**<br>
             사용자가 건강 검진 데이터를 이용해 당뇨 발병 확률 예측
                ''', unsafe_allow_html=True)

with empty3:
    st.write('')

with con4:
    st.image('data/다운로드.png', caption='출처 : GC녹십자아이메드')#, use_column_width=True)

