from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import streamlit as st
import requests

#Stream 받아 줄 Handler 만들기
from langchain.callbacks.base import BaseCallbackHandler
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


# 사이드바에 버튼 추가 (페이지 전환용)
st.sidebar.title("안녕하세요!")
st.sidebar.markdown("방문해주셔서 감사합니다:)")

if st.sidebar.button("🤖:rainbow[서현 챗봇]"):
    st.session_state.page = "서현봇"
if st.sidebar.button("📃이력서"):
    st.session_state.page = "이력서"
if st.sidebar.button("📄추천서"):
    st.session_state.page = "추천서"
if st.sidebar.button("📚학습 활동"):
    st.session_state.page = "학습 활동"

# 기본 페이지를 서현봇으로 설정
page = st.session_state.get('page', '서현봇')

if page == "서현봇":
    st.title(":blue[지서현]을 소개합니다! :sunglasses:")
    st.write("---")
    
    # 텍스트 파일 URL 설정 (고정된 경로)
    url = "https://raw.githubusercontent.com/jade-pond/seohyunbot/main/seohyun.txt"

    def load_txt_from_url(url):
        response = requests.get(url)
        response.raise_for_status()  # 요청이 실패하면 예외 발생
        text = response.text
        return [Document(page_content=text)]  # Document 객체로 반환

    # 텍스트 파일을 URL에서 로드 및 처리
    pages = load_txt_from_url(url)

    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    texts = text_splitter.split_documents(pages)

    # Embedding
    embeddings_model = OpenAIEmbeddings()

    # Load it into Chroma
    db = Chroma.from_documents(texts, embeddings_model)

    # Question
    st.header("궁금한 점을 말씀해주세요 :)")
    st.markdown("🚀예시) 카카오 지원 동기가 무엇입니까?")

    # 사용자가 입력한 질문을 세션 상태에 저장
    if 'question' not in st.session_state:
        st.session_state.question = ""
    
    question = st.text_input(label="질문을 입력하세요:", value=st.session_state.question)
    
    if question:
        # 세션 상태에 질문 저장
        st.session_state.question = question
    
        with st.spinner('서현봇 로딩 중...'):
            chat_box = st.empty()
            stream_handler = StreamHandler(chat_box)
            
            # 온도 값을 높여서 유연한 답변 생성
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo", 
                temperature=0.9, 
                streaming=True,
                callbacks=[stream_handler],
            )
            
            # 시스템 프롬프트 추가
            system_prompt = (
                "당신은 구직자인 지서현의 대변인입니다. 카카오 인사 담당자에게 지서현의 이력서와 지원 동기를 전달하고, "
                "지서현의 경험과 능력이 카카오의 커머스 조직에 어떻게 기여할 수 있는지를 명확하고 설득력 있게 설명하세요. "
                "친근하면서도 전문적인 톤을 유지하며, 간결하고 명확하게 소통하세요. "
                "지서현의 경험, 역량, 그리고 카카오에 대한 열정을 강조하세요. "
                "카카오의 조직 문화와 가치에 부합하는 점을 언급하고, 지서현이 팀에 얼마나 잘 어울릴지에 대한 확신을 심어주세요."
            )
            
            # 프롬프트에 추가 지침을 포함
            custom_prompt = f"{system_prompt} {question}"
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm, 
                chain_type="stuff", 
                retriever=db.as_retriever()
            )
            qa_chain({"query": custom_prompt})


elif page == "추천서":
    st.title("📄 추천서")
    st.image("https://github.com/jade-pond/seohyunbot/raw/main/referenceletter.jpg", caption="추천서 이미지", use_column_width=True)

elif page == "이력서":
    st.title("📃 이력서")
    st.image("https://github.com/jade-pond/seohyunbot/raw/main/CV.jpg", caption="이력서 이미지", use_column_width=True)

elif page == "학습 활동":
    st.title("📚 학습 활동")
    
    # 링크 1 (노션)
    st.markdown("""
    <div style='border: 1px solid #ddd; padding: 10px; border-radius: 10px; margin-bottom: 20px;'>
        <a href="https://jeeseohyun.notion.site/Jee-Seo-Hyun-6822c9993db843d8aff3db76ec48d34f" target="_blank" style="text-decoration: none; color: black;">
            <div style='padding: 10px;'>
                <h3>노션 - 지서현</h3>
                <p>노션에서 지서현님의 포트폴리오를 확인하세요.</p>
            </div>
        </a>
    </div>
    """, unsafe_allow_html=True)

    # 링크 2 (네이버 블로그)
    st.markdown("""
    <div style='border: 1px solid #ddd; padding: 10px; border-radius: 10px;'>
        <a href="https://blog.naver.com/jadesea0816" target="_blank" style="text-decoration: none; color: black;">
            <div style='padding: 10px;'>
                <h3>네이버 블로그</h3>
                <p>네이버 블로그에서 지서현님의 활동을 확인하세요.</p>
            </div>
        </a>
    </div>
    """, unsafe_allow_html=True)
