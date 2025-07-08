import os
import certifi
from openai import AzureOpenAI


def openai_field(user_text):
    endpoint = "https://farm-sesacteam1-openai.openai.azure.com/"
    deployment = "gpt-4o"
    search_endpoint = "https://farm-sesacteam1-aisearch.search.windows.net"
    search_key = "put your Azure AI Search admin key here"
    search_index = "house-toji-rag"
    subscription_key = "REPLACE_WITH_YOUR_KEY_VALUE_HERE"

    # 키 기반 인증을 사용하여 Azure OpenAI 클라이언트 초기화
    os.environ["SSL_CERT_FILE"] = certifi.where()
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version="2025-01-01-preview",
    )

    # 채팅 프롬프트 준비
    chat_prompt = [
        # 시스템 프롬프트
        {
            "role": "system",
            "content": "사용자가 입력한 지역을 기준으로, '농지_전처리완료.csv' 파일 내 데이터를 분석하여 인근의 매물 정보를 세 군데 추천합니다. 가장 가까운 순서대로 데이터를 정리해 아래 정해진 양식으로 출력합니다. \n\n# 입력 처리\n\n1. 사용자가 입력한 지역(주소 또는 지명)을 수신합니다.\n2. `농지_전처리완료.csv` 파일에서 주소 데이터를 사용하여 입력 지역과 가장 가까운 곳을 식별합니다.\n3. 가까운 순서로 정렬하고, 상위 5개 매물 정보를 추출합니다.\n\n# 데이터 요구사항 및 파일 정보\n- 파일명: `농지_전처리완료.csv`\n- 컬럼:\n  1. 용지종류\n  2. 공부지목\n  3. 실지목\n  4. 판매구분\n  5. 매매/임대가\n  6. 면적\n  7. 매물주소\n  8. 판매자 이름\n  9. 판매자 연락처\n  10. 담당자 이름(부서)\n  11. 담당자 연락처\n  12. 등록일\n  13. 특이사항\n\n# 작업 요약\n\n1. 사용자 입력 지역과 `매물주소` 데이터를 비교하여 가까운 순서로 매물을 정렬합니다.\n2. 매물의 주요 정보를 5개만 추출하여 아래 출력 형식에 맞게 정리합니다.\n\n# Output Format\n\n아래 출력 형식을 따르십시오:\n\n```\n- 추천 매물 1:\n  용지종류: [용지종류]\n  공부지목: [공부지목]\n  실지목: [실지목]\n  판매구분: [판매구분]\n  매매/임대가: [매매/임대가]\n  면적: [면적]\n  매물주소: [매물주소]\n  판매자 이름: [판매자 이름]\n  판매자 연락처: [판매자 연락처]\n  담당자 이름(부서): [담당자 이름(부서)]\n  담당자 연락처: [담당자 연락처]\n  등록일: [등록일]\n  특이사항: [특이사항]\n\n- 추천 매물 2:\n  ...\n(5개 매물 정보까지 반복)\n```\n\n# Notes\n- 비슷한 거리일 경우 등록일이 더 최근인 매물을 우선으로 추천하십시오.\n- 등록일 포맷이 표준 날짜 형식이 아닌 경우 적절히 변환하여 정렬에 활용합니다. \n- 특정 컬럼의 데이터가 결측치일 경우 빈칸으로 남겨놓습니다.\n- 특이사항이 없는 경우 \"특이사항 없음\"으로 출력합니다. \n- 사용자 입력 지역이 데이터와 크게 불일치하거나 매물이 없을 경우 “입력하신 지역의 이름을 확인하고 다시 입력해주세요.”라고 출력하십시오. \n\n# Examples\n\n**사용자 입력**: 전라남도 함평군 함평읍\n\n**출력**: \n```\n- 추천 매물 1:\n  용지종류: 주택/시설\n  공부지목: 대\n  실지목: \n  판매구분: 임대\n  매매/임대가: 협의\n  면적: 33㎡\n  매물주소: 전라남도 함평군 함평읍 남일길 59\n  판매자 이름: 주혜경\n  판매자 연락처: 010-2670-4962\n  담당자 이름(부서): 정난숙\n  담당자 연락처: 061-320-2131\n  등록일: 2025-06-23\n  특이사항: 사무실 가능, 권리금 없음\n\n- 추천 매물 2:\n  ...\n```\n### 잘못된 입력이 입력되었을 경우\n다음과 같이 \"입력하신 지역의 이름을 확인하고 다시 입력해주세요.\"를 출력하세요.\n\n```\n**사용자 입력**: 제주도 제주시 애월읍  \n**출력**:\n입력하신 지역의 이름을 확인하고 다시 입력해주세요.\n```"
        },
        # 예시
        ## 사용자
        {
            "role": "user",
            "content": "전라남도 함평군 함평읍"
        },
        ## 도우미
        {
            "role": "assistant",
            "content": "사용자가 입력한 지역 \"전라남도 함평군 함평읍\"과 가장 가까운 매물을 분석하여 상위 3개의 매물을 추천드립니다. 아래는 정리된 결과입니다:\n\n| 순위 | 용지종류 | 공부지목 | 실지목 | 판매구분 | 매매/임대가 | 면적(㎡) | 매물주소 | 판매자 이름 | 판매자 연락처 | 등록일 | 특이사항 |\n|------|----------|----------|--------|----------|------------|----------|----------|------------|--------------|--------|----------|\n| 1 | 주택 | 대 | 대 | 매매 | 협의가 | 0 | 전라남도 함평읍 중앙길 | 비공개 | 비공개 | 2020-03-27 | 함평읍 군지부 농협 사거리 3층 건물, 1층 분식집 및 소형 상가 가능, 2층 주방 겸 방, 3층 화장실 및 방 |\n| 2 | 주택/농지 | 대 | 대 | 매매 | 협의가 | 1494 | 전라남도 강진군 칠량면 반계길 32-1 | 주인남 | 010-9115-2913 | 2020-03-20 | 강진군 칠량면 장계리 149-1, 150-1, 151-1, 155답 452평 |\n| 3 | 주택 | 대 | 대 | 매매 | 5 | 486 | 전라남도 강진군 칠량면 구로길 77-8 | 전순애 | 010-3857-8463 | 2020-03-18 | 대지 147평, 건평 20평, 인근밭 351평 |\n\n위 매물 정보는 입력 지역과의 거리 및 관련 데이터를 기반으로 정렬된 결과입니다. 추가 정보가 필요하시면 알려주세요!"
        },
        # 사용자 텍스트
        {
            "role": "user",
            "content": user_text
        }
    ]

    # 음성이 사용되는 경우 음성 결과 포함
    messages = chat_prompt

    # 완료 생성
    completion = client.chat.completions.create(
        model=deployment,
        messages=messages,
        max_tokens=1500,
        temperature=0.7,
        top_p=0.2,
        frequency_penalty=1,
        presence_penalty=1,
        stop=None,
        stream=False,
        extra_body={
            "data_sources": [{
                "type": "azure_search",
                "parameters": {
                    "endpoint": f"{search_endpoint}",
                    "index_name": search_index,
                    "semantic_configuration": "house-toji-rag-semantic-configuration",
                    "query_type": "vector_semantic_hybrid",
                    "fields_mapping": {},
                    "in_scope": True,
                    "filter": None,
                    "strictness": 2,
                    "top_n_documents": 5,
                    "authentication": {
                        "type": "api_key",
                        "key": f"{search_key}"
                    },
                    "embedding_dependency": {
                        "type": "deployment_name",
                        "deployment_name": "text-embedding-3-large"
                    }
                }
            }]
        }
    )

    return completion.to_json()


if __name__ == "__main__":
    test_text = "전라남도 완도군 청산면"
    print(openai_field(test_text))
