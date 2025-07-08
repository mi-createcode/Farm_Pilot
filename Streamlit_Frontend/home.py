import streamlit as st
import base64

def image_to_base64(image_path):
    """
    이미지 파일을 base64 문자열로 인코딩합니다.
    파일을 찾을 수 없는 경우 오류 메시지를 출력하고 빈 문자열을 반환합니다.
    """#
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        st.error(f"Error: {image_path} 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
        return "" # 파일을 찾지 못했을 때 빈 문자열 반환

def home_page():
    # ✅ 이미지 base64 인코딩
    # 'logo.png'와 'icon.png' 파일이 스크립트와 같은 디렉토리에 있는지 확인하세요.
    logo_base64 = image_to_base64("logo1.png")
    icon_base64 = image_to_base64("icon.png")

    # 세션 상태 초기화
    if 'show_sidebar' not in st.session_state:
        st.session_state.show_sidebar = False
    if 'page' not in st.session_state:
        st.session_state.page = 'home'

    # ✅ 페이지 설정
    st.set_page_config(page_title="🤖FarmPilot", layout="wide")

    # ✅ CSS 스타일 삽입
    st.markdown("""
        <style>
            /* Pretendard 폰트 임포트 */
            @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;700&display=swap');

            /* 전체 페이지 기본 스타일 및 배경색 설정 */
            html, body, [class*="css"] {
                font-family: 'Pretendard', sans-serif;
                background-color: #E0F2F7; /* 전체 배경을 부드러운 연한 하늘색으로 설정 */
                color: #333; /* 기본 글자색 (배경에 맞춰 조정) */
            }

            /* 로고 래퍼 스타일 */
            .logo-wrapper {
                text-align: center;
                margin-top: 40px;
                margin-bottom: 20px;
            }

            /* 히어로 섹션 래퍼 스타일 */
            .hero-wrapper {
                position: relative;
                border-radius: 24px;
                overflow: hidden;
                margin: 30px auto 40px auto;
                box-shadow: 0 8px 20px rgba(0,0,0,0.05);
            }

            /* 히어로 배경 이미지 스타일 */
            .hero-background {
                position: absolute;
                top: 0; left: 0;
                width: 100%; height: 100%;
                background-image: url('https://watermark.lovepik.com/photo/20211123/large/lovepik-green-rice-picture_500824006.jpg');
                background-size: cover;
                background-position: center;
                opacity: 0.2; /* 투명도 조절 */
                z-index: 0;
            }

            /* 히어로 콘텐츠 스타일 */
            .hero-content {
                position: relative;
                z-index: 1;
                display: flex;
                justify-content: flex-start;
                align-items: center;
                padding: 60px;

                max-width: 1000px;
                margin: 0 auto;
                gap: 60px;
            }

            /* 히어로 섹션 왼쪽 텍스트 스타일 */
            .hero-left h1 {
                font-size: 42px;
                color: #101C3D;
                margin-bottom: 20px;
            }

            .hero-left p {
                font-size: 20px;
                color: #191970;
                line-height: 1.6;
            }

            /* 히어로 섹션 오른쪽 이미지 스타일 */
            .hero-right img {
                max-width: 100%;
                border-radius: 12px;
            }

            /* --- 상단 메뉴 버튼(☰)의 기본 Streamlit 버튼 스타일 --- */
            /* 이 부분은 최소한으로 유지하여 메인 버튼 스타일과 충돌을 피합니다. */
            ..stButton > button, 
            .stButton > button > div, 
            .stButton > button > span {
                border-radius: 32px;
                border: none;
                transition: all 0.3s ease;
                height: auto; /* 내용에 따라 높이 자동 조절 */
                width: auto; /* 내용에 따라 너비 자동 조절 */
                font-size: 94px; /* 기본 폰트 크기 */
                font-weight: 800; /* 기본 폰트 두께 */
                background: transparent; /* 투명한 배경 */
                color: #333; /* 기본 글자색 */
                padding: 0.25rem 0.5rem; /* 기본 패딩 */
                cursor: pointer; /* 클릭 가능한 요소임을 표시 */
            }

            .stButton > button:hover {
                background: rgba(0,0,0,0.05); /* 기본 호버 효과 */
                color: #ADD8E6;
            }

            div[class^="stButton"] button {
                font-size: 76px !important;
                padding: 20px 32px !important;
                font-weight: 1500 !important;
                line-height: 1.2 !important;
                border-radius: 16px !important;
            }

            /* Streamlit 컬럼 사이의 구분선 스타일 */
            section[data-testid="column"] {
                border-left: 1px solid #d0e0d0;
                padding-left: 15px; /* 왼쪽 패딩 */
                padding-right: 15px; /* 오른쪽 패딩 */
            }

            /* 첫 번째 컬럼에는 왼쪽 구분선 제거 */
            section[data-testid="column"]:first-child {
                border-left: none;
            }
        </style>
    """, unsafe_allow_html=True)

    # 상단 메뉴 버튼 컨테이너 (실제 기능은 없지만 HTML 구조상 존재 가능)
    with st.container():
        st.markdown("""
            <div class="menu-button-container">
            </div>
            """, unsafe_allow_html=True)

    # 상단 메뉴 버튼 (☰)
    col_menu, col_empty = st.columns([1, 9])
    with col_menu:
        if st.button("☰"):
            st.session_state.show_sidebar = not st.session_state.show_sidebar
    with col_empty:
        pass # 빈 컬럼으로 공간 확보

    # 사이드바 메뉴
    if st.session_state.show_sidebar:
        with st.sidebar:
            st.markdown("### 메뉴")
            if st.button("🏠 홈"):
                st.session_state.page = 'home'
                st.session_state.show_sidebar = False
            if st.button("🌾 맞춤작물"):
                st.session_state.page = 'input'
                st.session_state.show_sidebar = False
            if st.button("📜 귀농정책 및 보조금"):
                st.session_state.page = 'policy'
                st.session_state.show_sidebar = False
            if st.button("🏠 빈집찾기"):
                st.session_state.page = 'housing'
                st.session_state.show_sidebar = False

    # 🚜 로고 삽입
    # base64 인코딩된 로고 이미지가 있을 때만 렌더링
    if logo_base64:
        st.markdown(f"""
            <div class="logo-wrapper">
                <img src="data:image/png;base64,{logo_base64}" width="400" alt="FarmPilot Logo">
            </div>
        """, unsafe_allow_html=True)

    # ✈️ 히어로 섹션
    # base64 인코딩된 아이콘 이미지가 있을 때만 렌더링
    if icon_base64:
        st.markdown(f"""
            <div class="hero-wrapper">
                <div class="hero-background"></div>
                <div class="hero-content">
                    <div class="hero-left">
                        <h1>당신의 귀농 여정,<br>어렵지 않게 안내해드려요  </h1>
                        <p>복잡하고 막막한 귀농, <strong>FarmPilot</strong>은<br>정확하고 따뜻한 길잡이입니다.</p>
                    </div>
                    <div class="hero-right">
                        <img src="data:image/png;base64,{icon_base64}" width="300">
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # 🌾 주요 3개 버튼 섹션 (핵심: 이 div로 버튼들을 감싸서 CSS 선택자 분리)
    left_spacer, center_col, right_spacer = st.columns([1, 2, 1])

    with center_col:
        # 내부에서 버튼을 3개로 나누기
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🌾 작물 추천"):
                st.session_state['page'] = 'input'
        with col2:
            if st.button("📜 귀농 정책"):
                st.session_state['page'] = 'policy'
        with col3:
            if st.button("🏠 매물 찾기"):
                st.session_state['page'] = 'housing'


# 앱 실행 엔트리 포인트
if __name__ == "__main__":
    home_page()