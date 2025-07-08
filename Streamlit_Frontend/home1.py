import streamlit as st

def home_page():
    st.title("🤖 FarmPilot")
    st.markdown("""
        <h1 style='text-align: center; font-size: 36px;'>
            당신의 귀농 여정,<br> 이제 우리가 조종합니다 ✈️
        </h1>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🌾 작물 추천", use_container_width=True):
            st.session_state['page'] = 'input'  # 👈 연결 포인트!

    with col2:
        if st.button("📜 귀농 가이드", use_container_width=True):
            st.session_state['page'] = 'policy'

    with col3:
        if st.button("🏠 주거 지원", use_container_width=True):
            st.session_state['page'] = 'housing'  # 나중에 연결
