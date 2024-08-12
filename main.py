__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
import requests

st.title(":blue[지서현]을 소개합니다!:sunglasses:")
st.write("---")

# 텍스트 파일 URL 설정
url = "https://raw.githubusercontent.com/jade-pond/seohyunbot/main/seohyun.txt"

def load_txt_from_url(url):
    response = requests.get(url)
    response.raise_for_status() 
    text = response.text
    return [Document(page_content=text)]

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

# Stream 받아 줄 Handler
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# Question
st.header("궁금한 점을 말씀해주세요 :)")
question = st.text_input('질문을 입력하세요.')

if st.button('질문하기'):
    with st.spinner('서현봇 로딩 중...'):
        chat_box = st.empty()
        stream_handler = StreamHandler(chat_box)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", 
                         temperature=0,
                         streaming=True,
                         callbacks=[stream_handler])
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())
        qa_chain({"query": question})
