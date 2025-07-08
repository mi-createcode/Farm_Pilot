import os
import certifi
from openai import AzureOpenAI


def openai_compliance(user_text):
    endpoint = "https://farm-sesacteam1-openai.openai.azure.com/"
    deployment = "gpt-4o"
    search_endpoint = "https://farm-sesacteam1-aisearch.search.windows.net"
    search_key = "put your Azure AI Search admin key here"
    search_index = "farm-compliance"
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
            "content":  "전라도(전라남도 및 전라북도) 내 특정 지역과 귀농 지원 정책에 대해 '귀농귀촌' 관련 PDF 파일을 참조하여 상세 사항을 제공하십시오.\n\n## 제한 사항\n\n- 답안을 생성할 때 반드시 Reference Architecture for Generative AI (RAG)에 적재된, 제목에 '귀농귀촌'이 포함된 PDF 파일만 참조해야 합니다.\n- 요청된 지역과 관련된 정책에 한정해서 답변을 작성합니다.\n- 정책 내용은 요청자가 지정한 출력 양식에 맞춰 작성합니다.\n\n# Steps\n\n1. 입력된 요청에서 사용자가 명시한 전라도의 특정 지역(예: 전라북도 고창, 전라남도 여수 등)을 식별하십시오. 단, 해당 지역이 전라도에 속하지 않는 경우, 절대로 답안을 생성하지 말고 다음의 문구를 출력해 사용자가 올바른 지역명을 입력하도록 유도하시오: \"전라도에 속하는 지역명인지 확인한 후, 다시 질문해주세요.\"\n2. 해당 지정 지역과 관련된 귀농 지원 정책을 찾기 위해, RAG에 적재된 '귀농귀촌'이라는 이름이 포함된 PDF 데이터베이스를 검색하고 관련 내용을 추출하십시오.\n3. 요청된 양식에 따라 정보를 요약하고 구조화하십시오. 필요한 경우 정책의 주요 기준, 지원 자격, 지원 혜택 등을 포함하십시오.\n4. 참조한 내용을 정확히 바탕으로 정보를 확실히 전달하고, 확인 불가능한 정보는 제공하지 마십시오.\n\n# Output Format  \n\n다음의 JSON 양식으로 답변을 작성합니다:\n\n```json\n{\n  \"지역\": \"[요청된 특정 지역]\",\n  \"정책명\": \"[귀농 지원 정책 이름]\",\n  \"지원자격\": \"[어떤 자격이 필요한지 상세히 기술]\",\n  \"지원내용\": {\n    \"금전지원\": \"[금전적 혜택 관련 상세 설명]\",\n    \"기술지원\": \"[필요 시 기술 관련 지원 설명]\",\n    \"교육지원\": \"[교육 프로그램 및 교육 지원에 관한 상세 설명]\"\n  },\n  \"추가정보\": \"[기타 참고할 사항을 간략히 요약]\",\n  \"출처\": \"[참조한 PDF 파일 이름 및 관련 페이지]\"\n}\n```\n\n# Examples  \n\n**Input:**  \n\"전라남도 순천의 귀농 정책에 대해 상세히 알려주세요.\"\n\n**Output:**  \n```json\n{\n  \"지역\": \"전라남도 순천\",\n  \"정책명\": \"귀농 정착 지원 사업\",\n  \"지원자격\": \"만 19세에서 65세 이하 귀농 예정자 또는 귀농 후 5년 이내인 사람\",\n  \"지원내용\": {\n    \"금전지원\": \"최대 5천만 원의 창업 자본 대출 지원\",\n    \"기술지원\": \"비닐하우스 설치 및 농업 기계 구입 관련 기술 지원\",\n    \"교육지원\": \"농업 전문 교육과정 및 멘토링 프로그램 제공\"\n  },\n  \"추가정보\": \"귀농 자격 요건을 충족해야 하며, 해당 지역 이주 후 최소 1년 이상 거주해야 함.\",\n  \"출처\": \"귀농귀촌.pdf, 페이지 15\"\n}\n```\n\n**주의:**  \n- 실제 답변 시, 참조한 데이터와 정책이 반드시 정확히 일치해야 하며, 가상의 정보 생성은 금지됩니다.\n- 오류 가능성이 있는 경우, 답변에서 이를 명확히 언급하십시오. \n\n## Notes\n\n- 요청한 지역 내에서 관련 정책이 없을 경우, \"해당 지역과 관련된 귀농 지원 정책 정보가 없습니다.\"라는 메시지와 함께 JSON을 제공하십시오.\n- 지역명이 \"전라도\"로만 입력될 경우, \"전라남도\"와 \"전라북도\"를 포함한 전국귀농귀촌 정책을 요약하여 제공합니다.\n- 사용자가 입력한 요청의 구체성이 부족하면 가급적 정책 탐색을 시도하되, 불가능할 경우 요청의 구체화를 유도하십시오.\n- 전화번호를 출력해야 할 경우, 해당 정책이 있는 페이지의 다른 문단의 전화번호를 가져와서는 절대 안 됩니다. 반드시 해당 정책이 서술된 문단의 해당 정책 담당 기관의 전화번호를 출력하시오."
        },
        # 예시
        ## 사용자
        {
            "role": "user",
            "content": "전라남도 순천의 귀농 정책 알려줘."
        },
        ## 도우미
        {
            "role": "assistant",
            "content": "### 전라남도 순천시 귀농 지원 정책\n\n1. **귀농 지원 센터 연락처**  \n   - 순천시 농업기술센터 농업정책과:  \n     - 전화: 061-749-8703, 061-749-8705 [doc1].\n\n2. **귀농인 정착 지원**  \n   - 귀농인 정착자금 지원:  \n     - 지원 대상: 귀농 후 3년 이내의 귀농인.  \n     - 지원 금액: 가구당 월 30만 원, 최대 12개월 지원 .  \n   - 귀농인 주택 수리비 지원:  \n     - 지원 대상: 귀농 후 5년 이내의 귀농인.  \n     - 지원 금액: 가구당 최대 1,000만 원 .\n\n3. **귀농 교육 및 정착 프로그램**  \n   - 귀농·영농 교육:  \n     - 교육 시간: 40시간 이상 이수 필요.  \n     - 대상: 귀농 후 3년 이내의 귀농인 .  \n   - 영농 실습비 지원:  \n     - 지원 대상: 18~49세 귀농인.  \n     - 지원 금액: 월 50만 원, 매월 10일 이상, 50시간 이상 실습 시 지원 .\n\n4. **농업 창업 지원**  \n   - 농업 창업자금 융자 지원:  \n     - 지원 금액: 연 200만 원, 최대 2년 .  \n   - 소규모 농업용 기자재 구입비 지원:  \n     - 지원 금액: 가구당 600만 원, 자부담 10% .\n\n위 정책들은 순천시에서 귀농을 고려하는 분들에게 실질적인 도움을 제공하기 위해 마련된 지원책입니다. 추가 문의는 순천시 농업기술센터로 연락하시기 바랍니다."
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
        top_p=0.05,
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
                    "semantic_configuration": "farm-compliance-semantic-configuration",
                    "query_type": "vector_semantic_hybrid",
                    "fields_mapping": {},
                    "in_scope": True,
                    "filter": None,
                    "strictness": 5,
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
    test_text = "진안군 귀농 농업창업 및 주택구입 지원사업 알려줘."
    print(openai_compliance(test_text))
