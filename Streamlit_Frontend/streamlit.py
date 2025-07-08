
import streamlit as st

st.title("귀농 정보 입력 폼")

# 1. 귀농 예상 시기 선택
expected_time = st.selectbox(
    "귀농을 예상하는 시기를 선택하세요",
    ["6개월 이내", "1년 이내", "2년 이내", "미정"]
)

# 2. 선호 지역 선택 (복수 선택 가능하게 multiselect 도 가능)
preferred_region = st.selectbox(
    "선호하는 귀농 지역을 선택하세요",
    ["강원도", "경상북도", "경상남도", "전라북도", "전라남도", "충청북도", "충청남도", "기타"]
)

# 3. 알러지 정보 입력
allergy_info = st.text_area(
    "알러지가 있는 작물이나 성분이 있다면 적어주세요",
    placeholder="예: 복숭아, 밀가루, 땅콩 등"
)

# 제출 버튼
if st.button("제출하기"):
    st.subheader("입력한 정보")
    st.write(f"🗓 귀농 예상 시기: **{expected_time}**")
    st.write(f"📍 선호 지역: **{preferred_region}**")
    st.write(f"⚠️ 알러지 정보: {allergy_info if allergy_info else '없음'}")

    # 여기에 이 값들을 서버로 보내거나 API 요청 등을 넣으면 돼!
