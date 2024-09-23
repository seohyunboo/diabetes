import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from data.function import remove_outliers
import matplotlib.font_manager as fm


# 페이지 설정
st.set_page_config(layout="wide")

# 데이터 로드
data = joblib.load('data/10년치데이터.pkl')
# 시스템에서 사용할 수 있는 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'

## main ##
with st.container():
    tab1, tab2, tab3= st.tabs(['사용 데이터', "당뇨 현황", "당뇨 진단 기준"])

with tab1: # 국건영 소개
    con1, con2= st.columns([0.4, 0.6])

    st.subheader('이 조사는 보통 보건복지부와 질병관리청, 그리고 관련 연구 기관들이 협력하여 진행합니다.')

    with con1:
        st.image('data/국건영.png', caption='출처: 질병관리청')
    with con2:
        st.header(':pill: 국민 건강 영양 조사란?')
        with st.expander(" ",expanded=True):
            st.write('''
                    ##### :small_blue_diamond: 한국인의 건강 상태와 식습관을 분석하여, 국민 건강 증진을 위한 정책 및 프로그램을 개발하는 데 필요한 기초 자료를 제공

                    :small_blue_diamond: 조사 내용
                    - 건강 상태: 개인의 건강 이력, 질병 유병률, 신체 지표(체중, 신장, 혈압 등) 등을 조사
                    - 영양 섭취: 식단 분석을 통해 영양 불균형 또는 과잉 섭취 문제를 파악
                    - 생활 습관: 운동, 흡연, 음주 등 생활 습관과 관련된 정보를 수집
                    - 사회경제적 정보: 교육 수준, 직업, 소득 등 사회경제적 배경도 조사하여 건강과 영양에 미치는 영향을 분석

                    :small_blue_diamond: 조사 방법:   -설문조사 -신체 측정 -혈액 검사
                        ''')




with tab2:
    st.header(":pill:  당뇨 현황")
    st.write("2011~2021년 총 10년치의 국민건강영양조사 데이터에 당뇨 기준을 적용해 정상과 당뇨를 나눈 비율입니다.")
    st.markdown("___")

    con1, con2= st.columns([0.4, 0.6])
    with con1:
        # 당뇨여부별 카운트 계산
        count_us = data['당뇨여부'].value_counts().sort_index()

        # 라벨 및 값 설정
        labels = ['정상(0)', '당뇨(1)']
        values = count_us

        # Plotly 파이 차트 생성
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.5,  # 도넛 차트로 변경하고 싶다면 이 값을 설정합니다.
            textinfo='label+percent',  # 라벨과 퍼센트 정보 표시
            insidetextorientation='horizontal',  # 텍스트 방향 설정
            marker=dict(colors=['#66c2a5', '#fc8d62']),  # 색상 설정
            showlegend=True  # 범례 표시
        )])

        # 그래프 레이아웃 설정
        fig.update_layout(
            title_text='당뇨 비율',
            font=dict(family="Malgun Gothic", size=14),  # 한글 폰트 설정
            annotations=[dict(text='당뇨 비율', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )

        st.plotly_chart(fig)  # Plotly 그래프 출력

    with con2:
        count_by_year = data[data['당뇨여부'] == 1.0].groupby('year').size().reset_index(name='Count')

        # 그래프 그리기
        fig = go.Figure()

        fig = px.bar(count_by_year, x='year', y='Count',
                title='연도별 당뇨 발병 추이',
                labels={'year': '년도', 'Count': '당뇨 발병 수'},
                color='Count',
                color_continuous_scale='sunset')

        # 그래프 레이아웃 설정
        fig.update_layout(
            title='연도별 당뇨 발병 추이',
            xaxis_title='년도',
            yaxis_title='당뇨 발병 여부',
            xaxis=dict(tickmode='linear'),
            yaxis=dict(title='발병 환자 수(명)'),
            plot_bgcolor='white'
        )

        st.plotly_chart(fig)  # Plotly 그래프 출력



with tab3: # 당뇨 현황 및 예측 프로그램 필요성 시각화
    con3, con4 = st.columns([0.4, 0.3])

    with con3:
        st.header(":pill: 당뇨 진단 기준")
        df = data[['HE_glu','HE_HbA1c']]
        df_transformed = remove_outliers(df)

        dang = st.selectbox("당뇨 진단 기준에 따른 데이터 시각화", ["공복혈당", "당화혈색소"])
        if dang == '공복혈당':
            sns.histplot(x='HE_glu', data=df_transformed, kde=False, stat='density', bins=50)
            # 수직선 추가
            plt.axvline(100, label='당뇨 전단계', linestyle='-', color='red', alpha=0.5)
            plt.axvline(126, label='당뇨', linestyle='--', color='red', alpha=1)
            # 범례 추가
            plt.legend()
            plt.title('공복 혈당')
            # 그래프 출력
            st.pyplot(plt)

        elif dang == '당화혈색소':
            sns.histplot(x='HE_HbA1c', data=df_transformed, kde=False, stat='density', bins=48)
            # 수직선 추가
            plt.axvline(5.7, label='당뇨 전단계', linestyle='-', color='red', alpha=0.5)
            plt.axvline(6.5, label='당뇨', linestyle='--', color='red', alpha=1)
            # 범례 추가
            plt.legend()
            plt.title('당화혈색소')
            # 그래프 출력
            st.pyplot(plt)

        plt.clf()

        st.markdown('___')

        ## 상관관계 히트맵
        st.write('상관계수 히트맵')
        corr_= data.drop(columns = ['Unnamed: 0', 'year', 'HE_glu', 'HE_HbA1c'])
        corr = corr_.corr()

        # 히트맵 그리기
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix Heatmap')
        st.pyplot(plt)

    with con4:
        st.write(' ')
        st.header('사용한 칼럼 및 설명')
        st.write(' \
                 ')
        with st.expander(" ",expanded=True):
            st.write('''
                     #### 총 8개의 컬럼을 사용했습니다.
                    - **나이 'age'**: 나이가 증가함에 따른 변화를 확인하려고 칼럼에 넣었습니다.
                    - **성별 'sex'**: 남,녀 성별에 따른 변화를 확인하려고 칼럼에 넣었습니다.
                    - **허리 둘레 'HE_wc'**: 허리 둘레는 복부 비만을 나타내는 지표로, 내장지방의 양을 간접적으로 나타냅니다. 복부 비만은 인슐린 저항성과 밀접한 관계가 있습니다.
                    - **요단백 'HE_Upro'**: 요단백은 소변에서 단백질의 양을 측정한 것으로, 신장 기능의 지표로 사용됩니다.
                    - **혈중 크레아티닌 'HE_crea'**: 혈중 크레아티닌 수치는 신장 기능을 평가하는 데 사용됩니다. 신장 기능이 저하되면 혈중 크레아티닌 수치가 증가합니다.
                    - **간장질환 ALT 'HE_alt'**: ALT(알라닌 아미노전이효소)는 간의 기능을 평가하는 지표입니다. 간의 손상이나 염증이 있으면 ALT 수치가 상승합니다.
                    - **중성지방 'HE_TG'**: 중성지방은 혈액 내의 지방을 나타내며, 혈중 중성지방 농도는 대사 증후군의 일부로 당뇨병과 연관될 수 있습니다.
                    - **총 콜레스테롤  'HE_chol'**: 총 콜레스테롤은 혈액 내에서 운반되는 콜레스테롤의 총량을 측정한 것입니다. 이는 LDL 콜레스테롤, HDL 콜레스테롤, 그리고 중성지방을 포함합니다.

                        저밀도 지단백질 콜레스테롤 (LDL-C): 종종 "나쁜" 콜레스테롤이라고 불리며, 동맥 벽에 콜레스테롤을 축적시키는 역할을 합니다.

                        고밀도 지단백질 콜레스테롤 (HDL-C): "좋은" 콜레스테롤이라고 불리며, 혈중 콜레스테롤을 간으로 운반하여 배출하도록 돕습니다.

                        중성지방 (Triglycerides): 혈중에서 콜레스테롤과 함께 존재하며, 체내 에너지원으로 사용됩니다.
                        ''')
