import streamlit as st
import pandas as pd
import joblib

# 저장된 모델/스케일러 불러오기
knn_model_eng = joblib.load("diabetes_model.pkl")
scaler = joblib.load("diabetes_scaler.pkl")

st.title("🩺 당뇨병 예측 시스템")
st.write("사용자의 건강 정보를 입력하면 당뇨 여부를 예측한다냐.")

# 입력값 받기
preg = st.number_input("임신 횟수", min_value=0, step=1)
glucose = st.number_input("포도당", min_value=0.0)
bp = st.number_input("혈압", min_value=0.0)
skin = st.number_input("피부 두께", min_value=0.0)
insulin = st.number_input("인슐린", min_value=0.0)
bmi = st.number_input("체질량지수(BMI)", min_value=0.0)
dpf = st.number_input("당뇨 유전 지수", min_value=0.0)
age = st.number_input("나이", min_value=0, step=1)

# 버튼 클릭 시 실행
if st.button("예측하기"):

    # DataFrame 생성
    input_data = pd.DataFrame(
        [[preg, glucose, bp, skin, insulin, bmi, dpf, age]],
        columns=[
            '임신 횟수',
            '포도당',
            '혈압',
            '피부 두께',
            '인슐린',
            '체질량지수',
            '당뇨 유전 지수',
            '나이'
        ]
    )

    # 파생 변수 생성
    input_data['혈당BMI'] = input_data['포도당'] * input_data['체질량지수']
    input_data['고혈당'] = (input_data['포도당'] >= 126).astype(int)
    input_data['고령'] = (input_data['나이'] >= 50).astype(int)
    input_data['인슐린저항'] = input_data['포도당'] + input_data['인슐린']

    # 스케일링
    input_scaled = scaler.transform(input_data)

    # 예측
    predicted = knn_model_eng.predict(input_scaled)
    prob = knn_model_eng.predict_proba(input_scaled)

    diabetes_prob = prob[0][1] * 100

    # 결과 출력
    st.subheader("📊 예측 결과")

    if predicted[0] == 1:
        st.error(f"당뇨병일 가능성이 높다냐! ({diabetes_prob:.1f}%)")
    else:
        st.success(f"정상일 가능성이 높다냐아~ ({100-diabetes_prob:.1f}%)")

    st.write(f"당뇨 확률: {diabetes_prob:.1f}%")

    # 진행바
    st.progress(int(diabetes_prob))