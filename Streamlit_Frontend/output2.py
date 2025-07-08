import os
import certifi
import streamlit as st
from openai import AzureOpenAI

def openai_compliance(user_text):
    endpoint = "https://farm-sesacteam1-openai.openai.azure.com/"
    deployment = "gpt-4o"
    search_endpoint = "https://farm-sesacteam1-aisearch.search.windows.net"
    search_key =  "secret"
    search_index = "farm-compliance"
    subscription_key = "secret"

    os.environ["SSL_CERT_FILE"] = certifi.where()
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version="2025-01-01-preview",
    )

    chat_prompt = [
        {
            "role": "system",
            "content": "전라도(전라남도 및 전라북도) 내 특정 지역과 귀농 지원 정책에 대해 '귀농귀촌' 관련 PDF 파일을 참조하여 상세 사항을 제공하십시오.\n(중략...생략)"
        },
        {
            "role": "user",
            "content": user_text
        }
    ]

    completion = client.chat.completions.create(
        model=deployment,
        messages=chat_prompt,
        max_tokens=1500,
        temperature=0.7,
        top_p=0.05,
        frequency_penalty=1,
        presence_penalty=1,
        stream=False,
        extra_body={
            "data_sources": [{
                "type": "azure_search",
                "parameters": {
                    "endpoint": search_endpoint,
                    "index_name": search_index,
                    "semantic_configuration": "farm-compliance-semantic-configuration",
                    "query_type": "vector_semantic_hybrid",
                    "fields_mapping": {},
                    "in_scope": True,
                    "filter": None,
                    "strictness": 5,
                    "top_n_documents": 5,
                    "authentication": {
                        "type": "api_key",
                        "key": search_key
                    },
                    "embedding_dependency": {
                        "type": "deployment_name",
                        "deployment_name": "text-embedding-3-large"
                    }
                }
            }]
        }
    )

    # dict 형태 반환
    return completion

def policy_page():
    st.title("📜정책/보조금 찾기")
    st.markdown("전라남도 또는 전라북도 내 지역명을 입력하면 해당 지역의 귀농 정책 및 보조금을 알려드려요.")

    user_input = st.text_input("🔍 지역과 함께 궁금한 정보를 입력하세요", placeholder="예: 전라북도 진안군 귀농 정책 알려줘")

    if st.button("정책/보조금 조회"):
        if user_input.strip() == "":
            st.warning("지역명을 입력해 주세요.")
        else:
            with st.spinner("정보를 찾고 있어요... ⏳"):
                try:
                    response = openai_compliance(user_input)
                    with st.expander("상세 정책 정보 보기", expanded=True):
                        answer = response.choices[0].message.content
                        st.markdown(answer)
                except Exception as e:
                    st.error(f"오류 발생: {e}")


if __name__ == "__main__":
    policy_page()

