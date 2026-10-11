from dotenv import load_dotenv
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
examples = [
    {"news": "삼성전자가 내년 초 자체 개발한 인공지능 가속기를 출시하여 엔비디아에 도전할 예정이다.", "keywords": "삼성전자, 인공지능, 엔비디아"},
    {"news": "세계보건기구가 새로운 건강 위기에 대응하기 위해 국제 협력의 중요성을 강조했다.", "keywords": "세계보건기구, 건강위기, 국제협력"},
    {"news": "한국은행은 물가 안정과 경기 상황을 고려하여 기준금리를 동결했다.", "keywords": "한국은행, 기준금리, 물가안정"},
]
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{news}"),
    ("ai", "키워드: {keywords}"),
])
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt, examples=examples
)
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "뉴스 키워드 추출 전문가입니다. 핵심 키워드를 정확히 3개 추출하고 '키워드: A, B, C' 형식만 출력하세요."),
    few_shot_prompt,
    ("human", "{input}"),
])
chain = final_prompt | ChatOllama(model="qwen2.5:1.5b", temperature=0) | StrOutputParser()
news_list = [
    "제미나이 2.0 플래시는 구글 AI 스튜디오와 버텍스 AI에서 API를 통해 개발자에게 제공되며 멀티모달 입력을 지원합니다.",
    "국내 프로야구 구단들이 포스트시즌 진출을 위해 치열한 순위 경쟁을 펼치고 있다.",
    "정부는 친환경 에너지 확대를 위한 태양광 발전 지원 정책을 발표했다.",
]
for news in news_list:
    print("뉴스:", news)
    print(chain.invoke({"input": news}), "\n")
