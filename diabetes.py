import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# 페이지 설정
# -----------------------------
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# CSS 스타일
# -----------------------------
st.markdown("""
<style>

.main {
    background: linear-gradient(to bottom right, #f8fbff, #eef4ff);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* 제목 */
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #1e293b;
    margin-bottom: 5px;
}

.sub-title {
    font-size: 17px;
    color: #64748b;
    margin-bottom: 30px;
}

/* 카드 */
.card {
    background: white;
    padding: 25px;
    border-radius: 22px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.06);
    border: 1px solid #e2e8f0;
}

/* 결과 카드 */
.result-card {
    padding: 30px;
    border-radius: 24px;
    text-align: center;
    margin-top: 20px;
}

/* 정상 */
.good {
    background: linear-gradient(135deg, #dcfce7, #bbf7d0);
    color: #166534;
}

/* 위험 */
.bad {
    background: linear-gradient(135deg, #fee2e2, #fecaca);
    color: #991b1b;
}

/* 버튼 */
.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 15px;
    border: none;
    background: linear-gradient(135deg, #2563eb, #3b82f6);
    color: white;
    font-size: 18px;
    font-weight: 700;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 20px rgba(37,99,235,0.25);
}

/* 입력칸 */
.stNumberInput input {
    border-radius: 12px !important;
}

/* metric */
[data-testid="metric-container"] {
    background-color: white;
    border: 1px solid #e5e7eb;
    padding: 15px;
    border-radius: 18px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.04);
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# 모델 불러오기
# -----------------------------
knn_model_eng = joblib.load("diabetes_model.pkl")
scaler = joblib.load("diabetes_scaler.pkl")

# -----------------------------
# 헤더
# -----------------------------
st.markdown('<div class="main-title">🩺 당뇨병 예측 시스템</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">건강 정보를 기반으로 당뇨병 가능성을 분석합니다.</div>',
    unsafe_allow_html=True
)

# -----------------------------
# 입력 영역
# -----------------------------
col1, col2 = st.columns([1.2, 1])

with col1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📋 건강 정보 입력")

    c1, c2 = st.columns(2)

    with c1:
        preg = st.number_input("임신 횟수", min_value=0, step=1)
        glucose = st.number_input("포도당", min_value=0.0)
        bp = st.number_input("혈압", min_value=0.0)
        skin = st.number_input("피부 두께", min_value=0.0)

    with c2:
        insulin = st.number_input("인슐린", min_value=0.0)
        bmi = st.number_input("BMI", min_value=0.0)
        dpf = st.number_input("당뇨 유전 지수", min_value=0.0)
        age = st.number_input("나이", min_value=0, step=1)

    predict_btn = st.button("🔍 당뇨병 예측하기")

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# 오른쪽 정보 패널
# -----------------------------
with col2:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📈 건강 상태 요약")

    st.metric("포도당", f"{glucose}")
    st.metric("BMI", f"{bmi}")
    st.metric("나이", f"{age}")

    if glucose >= 126:
        st.warning("고혈당 수치가 감지되었습니다.")

    if bmi >= 30:
        st.warning("비만 범위 BMI입니다.")

    if age >= 50:
        st.info("연령 위험 요소가 포함됩니다.")

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# 예측 실행
# -----------------------------
if predict_btn:

    # 데이터프레임 생성
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
    normal_prob = 100 - diabetes_prob

    st.markdown("## 📊 분석 결과")

    # 진행바
    st.progress(int(diabetes_prob))

    # 결과 카드
    if predicted[0] == 1:

        st.markdown(f"""
        <div class="result-card bad">
            <h1>⚠️ 당뇨병 위험</h1>
            <h2>{diabetes_prob:.1f}%</h2>
            <p>당뇨병 가능성이 높게 예측되었습니다.</p>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="result-card good">
            <h1>✅ 정상 가능성 높음</h1>
            <h2>{normal_prob:.1f}%</h2>
            <p>현재 건강 상태는 비교적 안정적으로 보입니다.</p>
        </div>
        """, unsafe_allow_html=True)

    # 상세 정보
    st.markdown("### 📌 상세 분석")

    d1, d2, d3 = st.columns(3)

    with d1:
        st.metric("당뇨 확률", f"{diabetes_prob:.1f}%")

    with d2:
        st.metric("정상 확률", f"{normal_prob:.1f}%")

    with d3:
        risk_level = (
            "높음"
            if diabetes_prob >= 70
            else "보통"
            if diabetes_prob >= 40
            else "낮음"
        )

        st.metric("위험 수준", risk_level)
