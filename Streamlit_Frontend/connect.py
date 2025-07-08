import streamlit as st
from date_region import input_page
from weather import output_page
from home import home_page 
from emptyhouse import housing_page
from policy import policy_page
from asd import crop_weather_page

def main():
    if 'page' not in st.session_state:
        st.session_state['page'] = 'home'  # 홈화면 기본 설정

    if st.session_state['page'] == 'input':
        input_page()
    elif st.session_state['page'] == 'output':
        output_page()
    elif st.session_state['page'] == 'housing':
        housing_page()
    elif st.session_state['page'] == 'policy':
        policy_page()
    elif st.session_state['page'] == 'asd':
        crop_weather_page()
    else:
        home_page()

if __name__ == "__main__":
    main()