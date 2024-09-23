import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import io
import re
import pandas as pd
import streamlit as st
from google.cloud import vision
import os
import plotly.graph_objects as go
from pages.__pycache__.function import classify_risk

# 모델 로드
pred_model = joblib.load('data/pred.pkl')

# 페이지 설정
st.set_page_config(layout="wide")

def convert_to_float(value):
    if value is None or value == '':
        return None
    try:
        return float(value)
    except ValueError:
        return None  # 변환할 수 없는 경우 None 반환

def update_prediction(data):
    con1, con2= st.columns([0.7, 0.2])
    with con1:
        df = pd.DataFrame(data)
        pred_proba = pred_model.predict_proba(df)[:, 1]
        pred_pro = np.round(pred_proba * 100, 2)

        st.markdown('___')
        st.subheader(f':small_blue_diamond:업데이트된 당뇨확률: {pred_pro}%')

        # 당뇨 확률 데이터
        other_percentage = 100 - pred_pro  # 나머지 비율
        # 도넛 그래프를 위한 데이터
        labels = ['당뇨 확률', ' ']
        values = [pred_pro.mean(), other_percentage.mean()]
        colors = ['#ff9999', '#e6e6e6']

        # 도넛 그래프 생성
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.5,  # 도넛의 중심 구멍 크기
            marker=dict(colors=colors),
            textinfo='label+percent',  # 레이블과 퍼센트 표시
            textfont_size=16,
            title="당신의 당뇨 확률"
        )])

        # 레이아웃 업데이트
        fig.update_layout(
            title_text="당뇨 확률 그래프",
            title_x=0.5
        )

        # 그래프 출력
        st.plotly_chart(fig)


def home_page():
    st.title(':pill: 당뇨 예측')
    st.markdown('___')
    st.subheader(":small_blue_diamond: 건강 검진 결과 사진을 업로드 해주세요.")

    # 이미지 파일 업로드 (여러 파일 선택 가능)
    uploaded_files = st.file_uploader("이미지 최대 2장까지 업로드 가능", type=["jpg", "jpeg", "png", "tif", "tiff"], accept_multiple_files=True)

    st.markdown(' ')
    if uploaded_files:
        all_parsed_data = []

        for uploaded_file in uploaded_files:
            image_bytes = uploaded_file.read() # 파일을 바이트 형식으로 변환

            # 텍스트 추출
            parsed_data = detect_text(image_bytes)

            if parsed_data:  # 텍스트가 추출된 경우
                # 숫자값을 float로 변환
                for key in parsed_data:
                    parsed_data[key] = convert_to_float(parsed_data.get(key, ''))

                parsed_data['파일명'] = uploaded_file.name  # 파일명을 추가
                all_parsed_data.append(parsed_data)

        st.markdown(' ')
        if all_parsed_data:
            st.subheader("텍스트 추출이 완료되었습니다.")
            df = pd.DataFrame(all_parsed_data)
            st.write(df)

            if len(df) > 1:
                    # 두 행을 결합
                    combined_row = df.iloc[0].combine_first(df.iloc[1])

                    # 새로운 데이터 프레임을 생성
                    combined_df = pd.DataFrame([combined_row])
                    st.write(combined_df)
            else:
                # 데이터가 하나만 있을 경우
                combined_df = df

               # DataFrame을 세션 상태에 저장
            st.session_state.df = combined_df

             # 검토 및 수정 버튼
            if st.button("데이터 검토 및 수정"):
                st.subheader("추출된 데이터 확인")
                st.write('추출된 데이터가 사진과 일치하지 않을 경우 수정해주세요')
                valid_data = True
                with st.form('data_verification_form'):
                    for column in combined_df.columns:
                        if column in ['파일명']:  # 파일명은 수정하지 않음
                            continue
                        value = combined_df[column].values[0]
                        corrected_value = st.text_input(f"{column}:", value=value, key=column)
                        if value != corrected_value:
                            valid_data = False
                            combined_df[column] = corrected_value

                    if not valid_data:
                        st.warning("데이터가 수정되었습니다. 수정된 데이터를 확인 후 제출해주세요.")
                    else:
                        st.success("데이터가 올바릅니다.")

                    # 데이터 수정 완료 버튼
                    submit_button = st.form_submit_button("수정된 데이터 제출")
                    if submit_button:
                        st.session_state.df = combined_df
                        st.session_state.page = 'survey'
            else:
                # 데이터가 맞다고 판단되면 설문조사 페이지로 넘어가기
                if st.button('설문조사 시작하기'):
                    st.session_state.page = 'survey'

    else:
        # 파일이 업로드되지 않은 경우, 빈 데이터프레임으로 기본값 설정
        columns = ['요단백', '허리둘레','총 콜레스테롤', '중성지방', '혈중크레아티닌', '간장질환 ALT(SGPT)']
        default_data = {col: np.nan for col in columns}
        combined_df = pd.DataFrame([default_data])
        st.session_state.df = combined_df  # DataFrame을 세션 상태에 저장
        st.write(combined_df)

        if st.button('설문조사 시작하기'):
            st.session_state.page = 'survey'


from google.oauth2 import service_account
from googleapiclient.discovery import build

# Streamlit secrets에서 서비스 계정 정보를 가져옴
google_cloud = st.secrets["google_cloud"]

# 서비스 계정 인증 생성
credentials = service_account.Credentials.from_service_account_info(google_cloud)

# Google API 클라이언트를 생성하여 API 사용
service = build('your_api_service', 'v1', credentials=credentials)


def detect_text(image_bytes):
    try:
        # API키 값 위치 설정
        client = vision.ImageAnnotatorClient(credentials=credentials)

        # 이미지 객체 생성
        image = vision.Image(content=image_bytes)

        # 텍스트 감지 요청
        response = client.text_detection(image=image)
        texts = response.text_annotations

        # 추출된 텍스트를 하나의 문자열로 합침
        full_text = ' '.join([text.description for text in texts])
        st.write(full_text)

        # 텍스트 파싱
        parsed_data = parse_medical_report(full_text)

        return parsed_data

    except Exception as e:
        st.error(f'An error occurred: {str(e)}')
        return None

def parse_medical_report(text):
    result = {}
    patterns = [
        (r"요검사\s*신장질환\s*요단백\s*(\w+)", "요단백"),
        (r"허리둘레\s*(\w+)", "허리둘레"),
        (r"총 콜레스테롤\s*(\w+)", "총 콜레스테롤"),
        (r"중성지방\s*(\w+)", "중성지방"),
        (r"혈청\s크레아티닌\s*([\d.]+)", "혈중크레아티닌"),
        (r"ALT\(SGPT\)\s*(\d+)\s*U/L", "간장질환 ALT(SGPT)")
    ]

    for pattern, key in patterns:
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip()
            if key == "요단백":
                result[key] = 0 if value == '음성' else 1
            else:
                result[key] = value
        else:
            result[key] = "정보 없음"

    return result

def survey_page():
    # 파싱된 데이터
    extracted_data = st.session_state.df
    data = extracted_data.iloc[0].to_dict()
    #st.write(data)

    # 설문조사 표시
    display_survey(data)


def display_survey(existing_data):
    st.title(':pill: 건강 상태 조사')
    st.write('추가적인 자료를 필요로 하여 설문지에 응해 주시면 감사하겠습니다.')


    with st.form('survey_form'):
        form_elements = []

        # 성별 선택
        성별_default = 0 if existing_data.get('성별', '남성') == '남성' else 1
        성별 = st.radio("성별을 선택하세요", ('남성', '여성'), index=성별_default)
        성별_value = 1 if 성별 == '남성' else 2
        form_elements.append(('sex', 성별_value))

        # 나이
        나이 = existing_data.get('나이', 30)
        나이 = int(나이) if 나이 is not None and not pd.isna(나이) else 30
        나이 = st.number_input('나이를 입력해주세요 (80세 이상이면 80입력해주세요) ', min_value=20, max_value=80, value=나이, key='input_age')
        form_elements.append(('age', 나이))

        # 허리둘레
        허리둘레 = existing_data.get('허리둘레')
        if pd.isna(허리둘레):
            허리둘레 = float(허리둘레) if 허리둘레 is not None and not pd.isna(허리둘레) else 75
            허리둘레_value = st.number_input('허리둘레를 입력해주세요 (단위:cm)', min_value=50, max_value=150, value=허리둘레, key='input_HE_wc')
            form_elements.append(('HE_wc', 허리둘레_value))
        else:
            form_elements.append(('HE_wc', 허리둘레))

        # 단백뇨 검사 결과
        요단백 = existing_data.get('요단백')
        if pd.isna(요단백):
            요단백_default = 0 if existing_data.get('요단백', '음성') == '음성' else 1
            요단백 = st.radio("단백뇨 검사 결과를 선택해주세요", ('음성', '양성'), index=요단백_default)
            요단백_value = 0 if 요단백 == '음성' else 1
            form_elements.append(('HE_Upro', 요단백_value))
        else:
            form_elements.append(('HE_Upro', 요단백))

        # 총 콜레스테롤
        총_콜레스테롤 = existing_data.get('총 콜레스테롤')
        if pd.isna(총_콜레스테롤):
            총_콜레스테롤 = float(총_콜레스테롤) if 총_콜레스테롤 is not None and not pd.isna(총_콜레스테롤) else 50
            총_콜레스테롤_value = st.number_input('총 콜레스테롤 수치를 입력해주세요 (단위: mg/dL)', min_value=50, max_value=500, value=총_콜레스테롤, key='input_chol')
            form_elements.append(('HE_chol', 총_콜레스테롤_value))
        else:
            form_elements.append(('HE_chol', 총_콜레스테롤))

        # 중성지방
        중성지방 = existing_data.get('중성지방')
        if pd.isna(중성지방):
            중성지방 = float(중성지방) if 중성지방 is not None and not pd.isna(중성지방) else 50
            중성지방_value = st.number_input('중성지방 수치를 입력해주세요 (단위: mg/dL)', min_value=50, max_value=1000, value=중성지방, key='input_HE_TG')
            form_elements.append(('HE_TG', 중성지방_value))
        else:
            form_elements.append(('HE_TG', 중성지방))

        # 혈중 크레아티닌
        혈중_크레아티닌 = existing_data.get('혈중크레아티닌', np.nan)
        if pd.isna(혈중_크레아티닌):
            혈중_크레아티닌_value = st.number_input('혈중 크레아티닌 수치를 입력해주세요 (단위: mg/dL)', min_value=0.2, max_value=1.6, value=1.0, key='input_HE_crea', step=0.1)
            form_elements.append(('HE_crea', 혈중_크레아티닌_value))
        else:
            form_elements.append(('HE_crea', float(혈중_크레아티닌)))

        # ALT 수치
        간장질환_ALT = existing_data.get('간장질환 ALT(SGPT)', np.nan)
        if pd.isna(간장질환_ALT):
            ALT_value = st.number_input('간장질환 ALT(SGPT) 수치를 입력해주세요 (단위: U/L)', min_value=5, max_value=90, value=20, key='input_HE_alt')
            form_elements.append(('HE_alt', ALT_value))
        else:
            form_elements.append(('HE_alt', float(간장질환_ALT)))


        submitted = st.form_submit_button("제출")
        if submitted:
            # form_elements를 데이터프레임으로 변환하고 피처 순서 정렬
            data_dict = dict(form_elements)
            df = pd.DataFrame([data_dict])  # DataFrame을 올바르게 생성

            # 모델이 기대하는 피처 순서
            feature_order = ['HE_chol', 'HE_wc', 'HE_crea', 'HE_alt', 'HE_TG', 'age','sex','HE_Upro']

            # 피처 순서 정렬
            df = df[feature_order]

            st.session_state.data = df    #.to_dict(orient='list')
            st.session_state.view_data = data_dict
            st.session_state.page = 'results'

def results_page():
    st.header(":pill: 예측 결과")
    df = pd.DataFrame(st.session_state.data)
    view_df = pd.DataFrame([st.session_state.view_data])
    st.write(view_df)

    pred = pred_model.predict(df)
    pred_proba = pred_model.predict_proba(df)[:, 1]
    pred_pro = np.round(pred_proba * 100, 2)

    risk_labels = classify_risk(pred_proba)

    st.markdown('___')
    st.header(f':small_blue_diamond: 당신의 당뇨 확률 :{pred_pro}%')
    st.write(f'당신의 당뇨 현 상황은 {risk_labels}')

    if st.button("홈으로 돌아가기"):
        st.session_state.page = 'home'

    data = st.session_state.data

    # 사이드바에서 변수를 조정하여 실시간으로 예측값을 업데이트
    st.sidebar.header('변수를 조정해보세요')

    st.sidebar.slider('총 콜레스테롤 수치', min_value=50, max_value=500, value=int(data['HE_chol'][0]), key='slider_chol')
    st.sidebar.slider('중성지방 수치', min_value=50, max_value=300, value=int(data['HE_TG'][0]), key='slider_TG')
    st.sidebar.slider('혈중 크레아티닌 수치', min_value=0.2, max_value=1.6, value=float(data['HE_crea'][0]), step=0.1, key='slider_crea')
    st.sidebar.slider('ALT 수치', min_value=5, max_value=60, value=int(data['HE_alt'][0]), key='slider_alt')
    st.sidebar.slider('허리둘레', min_value=50, max_value=120, value=int(data['HE_wc'][0]), key='slider_wc')

       # 사용자가 사이드바에서 값을 변경할 때마다 업데이트
    updated_data = {
        'HE_chol': [int(st.session_state.slider_chol)],
        'HE_wc': [int(st.session_state.slider_wc)],
        'HE_crea': [float(st.session_state.slider_crea)],
        'HE_alt': [int(st.session_state.slider_alt)],
        'HE_TG': [int(st.session_state.slider_TG)],
        'age': [df['age'][0]],
        'sex': [df['sex'][0]],
        'HE_Upro': [df['HE_Upro'][0]]
    }

    update_prediction(updated_data)


    if st.button("다시하기"):
        # 모든 세션 상태 초기화
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        # 세션 상태 초기화 후 설문 페이지로 이동
        st.session_state.page = 'survey'


if 'page' not in st.session_state:
    st.session_state.page = 'home'

if st.session_state.page == 'home':
    home_page()

elif st.session_state.page == 'survey':
    survey_page()

elif st.session_state.page == 'results':
    results_page()
