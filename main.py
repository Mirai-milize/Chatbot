from flask import Flask, request, jsonify, render_template
import random

#### chatGPT RAG 활용
import os
from PyPDF2 import PdfReader
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
####
from dotenv import load_dotenv


#### api 키 설정 
# .env 파일에서 환경 변수 로드
load_dotenv()
api_key = os.getenv('API_KEY')
os.environ['OPENAI_API_KEY'] = api_key
####

#### PDF 위치 저장
folder_dir = './pdf'
pdfs = os.listdir(folder_dir)
raw_text = ""
for i in range(len(pdfs)):
    reader = PdfReader("./pdf/"+pdfs[i])
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text    
####



#### gpt 모델 설정
model = ChatOpenAI(model="gpt-3.5-turbo") # gpt-3.5-turbo, gpt-4

qa_chain = load_qa_chain(model, chain_type="map_reduce")
qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)
#### 


def get_response(prompt):

    response = qa_document_chain.run(
    input_document=raw_text,
    question=prompt)

    return response

app = Flask(__name__)


# Initialize session state
session_state = {
    'messages': []
}




@app.route('/')
def index():
    
    return render_template('page1.html')

@app.route('/two')
def index2():
    return render_template('chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():



    user_input = request.json.get('message')
    initial_content = '다음 내용에 대해 초등학생에게 이야기하듯이 대답해줘 '
    full_prompt = initial_content + user_input
    session_state['messages'].append(("User", user_input))
    response = get_response(full_prompt)
    session_state['messages'].append(("GPT", response))



   


    return jsonify({'message': response})
    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)