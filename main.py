from flask import Flask, request, jsonify, render_template, session
import random

#### chatGPT RAG 활용
import os
import openai
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain, LLMChain, StuffDocumentsChain
# from langchain.retrievers import EmbeddingRetriever
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.document_loaders import PyPDFDirectoryLoader
import numpy as np
####
from dotenv import load_dotenv


#### api 키 설정 
# .env 파일에서 환경 변수 로드
load_dotenv()
api_key = os.getenv('API_KEY')
os.environ['OPENAI_API_KEY'] = api_key
####

#### PDF 위치 저장
loader = PyPDFDirectoryLoader('./pdf/')
raw_text = loader.load_and_split()


#### gpt 모델 설정
model = ChatOpenAI(model="gpt-4o") # gpt-3.5-turbo, gpt-4

qa_chain = load_qa_chain(model, chain_type="stuff")
qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)
#### 


def split_text(raw_text):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
        encoding_name='cl100k_base'
    )
    # 문서를 청크로 나누기
    texts = text_splitter.split_documents(raw_text)
    print('text start')
    print(texts)
    return texts


# # 임베딩 및 벡터 데이터베이스 생성
embeddings_model = OpenAIEmbeddings()

texts = split_text(raw_text)

print('vector store start')
vectorstore = FAISS.from_documents(texts,
    embedding = embeddings_model,
    distance_strategy = DistanceStrategy.COSINE  
    )

# GPT 모델을 이용한 응답 생성 함수
def get_response(prompt):
    k = 5
    fetch_k = 15
    retriever = vectorstore.as_retriever(search_type = 'mmr', search_kwags={'k': k, 'fetch_k' : fetch_k})
    print('retriever is')
    print(retriever)


    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever = retriever,
        return_source_documents=True)
    
    response = chain(prompt)
    full_response = f"{response['answer']}" 
    
    return full_response



app = Flask(__name__)
app.secret_key = 'PM6_chatbot2'  # 세션을 사용하기 위해 secret_key 설정


# Initialize session state
session_state = {
    'messages': []
}


@app.route('/')
def page1():
    return render_template('chatbot.html')

@app.route('/two')
def chatbot():
    return render_template('chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    if 'messages' not in session:
        session['messages'] = []
    session_messages = session['messages']
    
    user_input = request.json.get('message')

    initial_content = '''
    너의 이름은 에듀로봇이야. 
    올해는 2024년이야.
    '개정', '변경'이라는 질문에는 연도별 변경사항과 이전 년도의 내용도 포함해서 마크다운 형식으로 답해줘. 
    다음 질문에 대해 한국어로 대답해. 

    '''
    full_prompt = initial_content
    
    # 사용자 입력만 저장
    session_messages.append(("User", user_input))
    full_prompt += f"User: {user_input}" 

    response = get_response(full_prompt)
    
    # 이전 GPT 응답만 저장
    if len(session_messages) >= 2:
        session_messages = session_messages[-2:]
    session_messages.append(("GPT", response))
    session['messages'] = session_messages

    return jsonify({'message': response})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)