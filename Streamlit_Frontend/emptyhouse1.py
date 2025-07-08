import os
import certifi
import streamlit as st
from openai import AzureOpenAI
from openai.types.chat import ChatCompletion

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
- 추천 매물 1:
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
def housing_page():
    st.title("🏠 빈 집 추천기")
    st.markdown("전라남도 또는 전라북도 내 지역명을 입력하면 해당 지역의 빈 집을 추천해드려요.")

    user_input = st.text_input("📍 지역을 입력하세요", placeholder="예: 전라북도 진안군")
    show_all = st.checkbox("📌 모든 추천 매물 보기")

    if st.button("🔎 빈 집 조회"):
        if not user_input.strip():
            st.warning("⚠️ 지역명을 입력해 주세요.")
            return

        with st.spinner("⏳ 추천 매물을 찾고 있어요..."):
            try:
                deployment = "gpt-4o"
                search_endpoint = "https://farm-sesacteam1-aisearch.search.windows.net"
                search_index = "house-toji-rag"
                search_key = "..."  # 실제 키
                subscription_key = "..."  # 실제 키
                endpoint = "https://farm-sesacteam1-openai.openai.azure.com/"

                client = create_openai_client(subscription_key, endpoint)
                response = get_completion(user_input, client, deployment, search_endpoint, search_key, search_index)

                full_result = response.choices[0].message.content
                if not full_result or "입력하신 지역의 이름을 확인" in full_result:
                    st.error("❌ 해당 지역의 매물이 없습니다.")
                    return

                listings = [l.strip() for l in full_result.split("- 추천 매물 ") if l.strip()]
                st.success("✅ 추천 매물 조회 완료!")

                display_count = len(listings) if show_all else min(3, len(listings))
                for i in range(display_count):
                    st.markdown(f"### 🏘️ 추천 매물 {i+1}")
                    st.markdown("- " + listings[i])

                if not show_all and len(listings) > 3:
                    st.info(f"총 {len(listings)}개의 매물이 있습니다. 체크박스를 클릭하여 모두 확인하세요.")
            except Exception as e:
                st.error(f"❌ 오류 발생: {str(e)}")