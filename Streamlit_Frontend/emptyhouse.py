import os
import certifi
import streamlit as st
from openai import AzureOpenAI
from openai.types.chat import ChatCompletion

AZURE_OPENAI_ENDPOINT = "https://farm-sesacteam1-openai.openai.azure.com/"
AZURE_OPENAI_KEY = "secret" 
DEPLOYMENT_NAME = "gpt-4o"
SEARCH_ENDPOINT = "https://farm-sesacteam1-aisearch.search.windows.net"
SEARCH_INDEX = "house-toji-rag"
SEARCH_KEY = "secret"

# 👉 Azure OpenAI 클라이언트 생성
def create_openai_client(api_key: str, endpoint: str) -> AzureOpenAI:
    os.environ["SSL_CERT_FILE"] = certifi.where()
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version="2025-01-01-preview"
    )

# 👉 채팅 프롬프트 구성
def build_chat_prompt(user_text: str) -> list[dict]:
    system_prompt = '''사용자가 입력한 지역을 기준으로, '농지_전처리완료.csv' 파일 내 데이터를 분석하여 인근의 매물 정보를 최대 5개까지 추천합니다. 가장 가까운 순서대로 정리하여 아래 양식으로 출력합니다.

# 출력 형식
  용지종류: [용지종류]
  공부지목: [공부지목]
  실지목: [실지목]
  판매구분: [판매구분]
  매매/임대가: [매매/임대가]
  면적: [면적]
  매물주소: [매물주소]
  판매자 이름: [판매자 이름]
  판매자 연락처: [판매자 연락처]
  담당자 이름(부서): [담당자 이름(부서)]
  담당자 연락처: [담당자 연락처]
  등록일: [등록일]
  특이사항: [특이사항]

(최대 5개 매물까지 출력)
'''
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text}
    ]

# 👉 Azure OpenAI 호출 함수
def get_completion(user_text, client, deployment, search_endpoint, search_key, search_index) -> ChatCompletion:
    messages = build_chat_prompt(user_text)

    return client.chat.completions.create(
        model=deployment,
        messages=messages,
        max_tokens=1500,
        temperature=0.7,
        top_p=0.4,
        frequency_penalty=1,
        presence_penalty=1,
        stream=False,
        extra_body={
            "data_sources": [{
                "type": "azure_search",
                "parameters": {
                    "endpoint": search_endpoint,
                    "index_name": search_index,
                    "semantic_configuration": "house-toji-rag-semantic-configuration",
                    "query_type": "semantic",
                    "fields_mapping": {},
                    "in_scope": True,
                    "filter": None,
                    "strictness": 5,
                    "top_n_documents": 2,
                    "authentication": {
                        "type": "api_key",
                        "key": search_key
                    }
                }
            }]
        }
    )

# 👉 Streamlit 메인 UI

def clean_listing_text(raw_text: str) -> str:
    keywords = [
        "용지종류", "공부지목", "실지목", "판매구분", "매매/임대가", "면적", "대지면적",
        "건축면적", "연면적", "매물주소", "판매자 이름", "판매자 연락처", "담당자 이름",
        "담당자 연락처", "등록일", "특이사항"
    ]
    
    for key in keywords:
        raw_text = raw_text.replace(f"{key}:", f"\n**{key}**: ")

    return raw_text.strip()



def housing_page():
    st.set_page_config(page_title="빈 집/ 매물 추천기", layout="centered")
    st.title("🏠 빈 집/매물 추천기")
    st.markdown("전라남도 또는 전라북도 내 지역명을 입력하면 해당 지역의 빈 집을 추천해드려요.")

    # session_state에서 도시/동 정보 읽기
    city = st.session_state.get('city', '')
    town = st.session_state.get('town', '')
    region_text = f"{city} {town}".strip()

    user_input = st.text_input("📍 지역을 입력하세요", value=region_text, placeholder="예: 전라북도 진안군")

    if st.button("🔎 빈 집/ 매물 조회"):
        prompt = user_input.strip()
        if not prompt:
            st.warning("⚠️ 지역명을 입력해 주세요.")
            return

        with st.spinner("⏳ 추천 매물을 찾고 있어요..."):
            try:
                client = create_openai_client(AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT)
                response = get_completion(
                    prompt,
                    client,
                    DEPLOYMENT_NAME,
                    SEARCH_ENDPOINT,
                    SEARCH_KEY,
                    SEARCH_INDEX
                )
                full_result = response.choices[0].message.content

                if not full_result or "입력하신 지역의 이름을 확인" in full_result:
                    st.error("❌ 해당 지역의 매물이 없습니다.")
                    return
                
                listings = full_result.split("- 추천 매물 ")
                listings = [l.strip() for l in listings if l.strip()]

                st.success("✅ 추천 매물 조회 완료!")
                
                for i, item in enumerate(listings):
                    formatted_text = clean_listing_text(item)
                    st.markdown(f"""
                        <div style="
                            background-color: #f8f9fa;
                            border-left: 5px solid #4CAF50;
                            padding: 1rem 1.2rem;
                            margin-bottom: 1.2rem;
                            border-radius: 10px;
                            box-shadow: 1px 1px 5px rgba(0,0,0,0.05);
                        ">
                            <h3 style="margin-top:0;">🏘️ 추천 매물 </h3>
                            <pre style="white-space: pre-wrap; word-wrap: break-word; font-size: 15px; line-height: 1.5;">{formatted_text}</pre>
                        </div>  
                    """, unsafe_allow_html=True)


            except Exception as e:
                st.error(f"❌ 오류 발생: {e}")

def get_house_text():
    if 'city' in st.session_state and 'town' in st.session_state:
        prompt = f"{st.session_state['city']} {st.session_state['town']} 빈 집 / 매물 정보"
        st.subheader(f"📍 {st.session_state['city']} {st.session_state['town']} 빈 집 / 매물 정보")
        with st.spinner("매물 정보를 불러오는 중... ⏳"):
            try:
                client = create_openai_client(AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT)
                response = get_completion(
                    prompt,
                    client,
                    DEPLOYMENT_NAME,
                    SEARCH_ENDPOINT,
                    SEARCH_KEY,
                    SEARCH_INDEX
                )
                answer = response.choices[0].message.content

                if not answer or "입력하신 지역의 이름을 확인" in answer:
                    st.error("❌ 해당 지역의 매물이 없습니다.")
                    return
                
                listings1 = answer.split("추천 매물")
                listings1 = [l.strip() for l in listings1 if l.strip()]

                st.success("✅ 추천 매물 조회 완료!")
                
                for i, item in enumerate(listings1):
                    formatted_text = clean_listing_text(item)
                    st.markdown(f"""
                            <div style="
                                background-color: #f8f9fa;
                                border-left: 5px solid #4CAF50;
                                padding: 1rem 1.2rem;
                                margin-bottom: 1.2rem;
                                border-radius: 10px; 
                                box-shadow: 1px 1px 5px rgba(0,0,0,0.05);
                            ">
                                <h3 style="margin-top:0;">🏘️ 추천 매물 </h3>
                                <pre style="white-space: pre-wrap; word-wrap: break-word; font-size: 15px; line-height: 1.5;">{formatted_text}</pre>
                            </div>  
                        """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"오류 발생: {e}")
    else:
        st.info("지역 정보가 없습니다. 먼저 지역명을 선택해 주세요.")

# 👉 앱 실행
if __name__ == "__main__":
    housing_page()
