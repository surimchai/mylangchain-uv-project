from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate.from_template(
    """
    당신은 한국어로 답변하는 전문 요리사입니다.

    다음 재료로 만들 수 있는 요리 한 가지를 추천하세요.

    재료: {ingredient}

    반드시 다음 형식으로만 답변하세요.

    추천 요리:
    재료:
    조리법:
    1.
    2.
    3.

    조건:
    - 조리법은 최대 3단계
    - 동일한 문장 반복 금지
    - 불필요한 설명 금지
    - 입력받은 재료를 최대한 활용
    """
)
llm = ChatOllama(model="qwen2.5:1.5b", temperature=0.3, num_predict=250, repeat_penalty=1.2)
chain = prompt | llm | StrOutputParser()

ingredient = input("재료를 입력하세요: ")
print(chain.invoke({"ingredient": ingredient}))
