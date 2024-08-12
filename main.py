__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')



# 라이브러리 로드
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
import streamlit as st
import requests
import os

st.title(":blue[지서현]을 소개합니다!:sunglasses:")
st.write("---")

# PDF 파일 경로 설정 (고정된 경로)
url = "https://raw.githubusercontent.com/your-username/your-repo/main/seohyun.pdf"
pdf_filepath = "seohyun.pdf"

# 파일 다운로드
response = requests.get(url)
with open(pdf_filepath, "wb") as file:
    file.write(response.content)

# PDF 로드
loader = PyPDFLoader(pdf_filepath)
pages = loader.load_and_split()

def pdf_to_documents(pdf_filepath):
    loader = PyPDFLoader(pdf_filepath)
    pages = loader.load_and_split()
    return pages

# PDF 파일 로드 및 처리
pages = pdf_to_documents(pdf_filepath)

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
st.header("질문해주세요ㅎㅎ")
question = st.text_input('질문을 입력하세요.')

if st.button('질문하기'):
    with st.spinner('지서현에 대해 조사하고 있습니다...'):
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())
        result = qa_chain({"query": question})
        st.write(result['result'])  # 필요한 결과만 추출하여 표시
