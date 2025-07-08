from ml_test import predict_top3_product
import streamlit as st

def crop_page():
    # 입력값 세션에서 꺼내기
    if 'expected_date' not in st.session_state or 'province' not in st.session_state:
        st.warning("입력 페이지에서 값을 먼저 입력해주세요.")
        return

    # 날짜 정보
    expected_date = st.session_state['expected_date']
    year = expected_date.year
    month = expected_date.month
    day = expected_date.day

# 딕셔너리: 도 이름 줄이기
    province_name_map = {
        '전북특별자치도': '전북',
        '전라남도': '전남',
    }

# 함수: 도 이름 치환 + city 결합
    def convert_province_city(province, city, town):
        province_short = province_name_map.get(province, province)  # 기본값은 province
        return f"{province_short} {city} {town}"


    # 사용 예시 (페이지 코드에서)
    province = st.session_state['province']
    city = st.session_state['city']
    town = st.session_state['town']
    sanji_nm = convert_province_city(province, city, town)

    # 예측 실행
    top3_results = predict_top3_product(sanji_nm, year, month, day)

    # 출력
    st.success("✅ 추천 작물 예측 완료!")
    for i, (label, score) in enumerate(top3_results, 1):
        st.markdown(f"**{i}. {label}** – 예측 확률: {score:.2%}")

if __name__ == "__main__":
    crop_page()