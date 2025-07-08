import streamlit as st
import datetime
import json

def load_region_data():
    def load_json(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            return json.load(f)
    return {**load_json("region_data.json"), **load_json("region_data_nam.json")}

def select_region(merged_data):
    dos = list(merged_data.keys())
    col1, col2, col3 = st.columns(3)

    selected_do = selected_si = selected_gu = None

    with col1:
        selected_do = st.selectbox("1️⃣ 도 선택", dos)

    if selected_do:
        sis = list(merged_data[selected_do].keys())
        with col2:
            selected_si = st.selectbox("2️⃣ 시/군/구 선택", sis)

        if selected_si:
            gus = list(merged_data[selected_do][selected_si].keys())
            with col3:
                selected_gu = st.selectbox("3️⃣ 읍/면/동 선택", gus)

    return selected_do, selected_si, selected_gu

def input_page():
    # 큰 글씨를 위한 스타일 커스텀
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;700&display=swap');
            html, body, [class*="css"] {
                font-family: 'Pretendard', sans-serif;
                font-size: 20px !important;
            }

            .stTextInput > div > input,
            .stTextArea textarea,
            .stSelectbox > div,
            .stDateInput input {
                font-size: 22px !important;
                padding: 14px !important;
            }

            button[kind="primary"] {
                background-color: #4CAF50 !important;
                font-size: 24px !important;
                font-weight: 700 !important;
                padding: 1rem 2.5rem !important;
                border-radius: 14px !important;
                color: white !important;
                width: 100%;
            }

            button[kind="primary"]:hover {
                background-color: #3e9442 !important;
            }

            h1 {
                font-size: 50px !important;
                text-align: center;
                margin-bottom: 0.5em;
            }

            h3 {
                font-size: 26px !important;
                margin-top: 1.5em;
                margin-bottom: 0.5em;
            }

            p {
                font-size: 20px !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # 제목 및 설명
    st.markdown("<h1>🌾 맞춤 작물 추천</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>귀농을 준비하시나요? 아래 정보를 입력하시면<br><strong>당신에게 맞는 작물</strong>을 추천해드립니다!</p>", unsafe_allow_html=True)

    st.markdown("---")

    # 📅 귀농 예정일
    st.markdown("### 📅 귀농 예정일을 선택해 주세요")
    default_date = datetime.date.today()
    expected_date = st.date_input(
        "예상 날짜", value=default_date,
        min_value=default_date, max_value=datetime.date(2030, 12, 31),
        key="expected_date"
    )

    st.markdown("---")

    # 📍 지역 선택
    st.markdown("### 📍 원하시는 지역을 선택해 주세요")
    st.markdown("도 → 시/군/구 → 읍/면/동 순으로 선택해 주세요")
    region_data = load_region_data()
    province, city, town = select_region(region_data)

    st.markdown("---")

    # ❗ 알러지 정보
    st.markdown("### ❗ 알러지가 있다면 알려주세요")
    allergy_info = st.text_area(
        "알러지 작물 또는 성분",
        placeholder="예: 복숭아, 밀가루, 땅콩 등"
    )

    st.markdown("---")

    # 💪 노동 가능 수준
    st.markdown("### 💪 노동 가능 수준을 선택해 주세요")
    labor_level = st.selectbox(
        "투입 가능한 노동 강도",
        ["하 (가벼운 작업)", "중 (보통 수준)", "상 (활동 많아도 괜찮아요)"]
    )

    st.markdown("---")

    # 제출 버튼
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 작물 추천 받기"):
            st.session_state['province'] = province
            st.session_state['city'] = city
            st.session_state['town'] = town
            st.session_state['allergy_info'] = allergy_info
            st.session_state['labor_level'] = labor_level

            if city:
                st.session_state['region'] = city
            elif province:
                st.session_state['region'] = province
            else:
                st.session_state['region'] = None

            st.session_state['page'] = 'asd'
