from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
llm = ChatOllama(model="qwen2.5:1.5b", temperature=0)
parser = StrOutputParser()

recommend_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 여행 전문가입니다. 입력된 지역의 대표 관광 명소 하나를 이름만 답하세요."),
    ("human", "여행지: {destination}"),
])
detail_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 여행 해설사입니다. 관광 명소의 역사, 특징, 방문 팁을 한국어로 구분하여 설명하세요. 모르는 사실은 추측하지 마세요."),
    ("human", "명소: {landmark}"),
])
recommend_chain = recommend_prompt | llm | parser
detail_chain = detail_prompt | llm | parser

full_chain = recommend_chain | (lambda landmark: {"landmark": landmark}) | detail_chain

destination = input("여행할 도시 또는 국가: ")
landmark = recommend_chain.invoke({"destination": destination})
print("1단계 결과:", landmark)
print("2단계 결과:\n", full_chain.invoke({"destination": destination}))
