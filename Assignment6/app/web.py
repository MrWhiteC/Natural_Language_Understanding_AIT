from flask import Flask, request, jsonify, render_template
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from langchain import PromptTemplate
from langchain.document_loaders import PyMuPDFLoader,WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ChatMessageHistory, ConversationBufferMemory,ConversationBufferWindowMemory
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import LLMChain,ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.question_answering import load_qa_chain


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device
from langchain_core.prompts import ChatPromptTemplate

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            " I'm your clone myself voramethbot. You can ask anything according to yourself. "
            "Sometimes, it is ok to feel loss, but remember to speak to  who might knock in a sane mind again. "
            "You can ask whether who you are ? what are your studies ? what are your interest ? what expertis areas ?  "
            "No matter you feel uncomfortble, please reach out to me, ok ? {context}.",
        ),
        ("human", "{question}"),
    ]
)


cv_docs = "Vorameth'sCV.pdf"

web_loader_1 = WebBaseLoader("https://www.researchgate.net/profile/Vorameth-Reantongcome")
webs1 = web_loader_1.load()


web_loader_2 = WebBaseLoader("https://www.linkedin.com/in/vorameth-reantongcome/")
webs2 = web_loader_2.load()


pdf_loader = PyMuPDFLoader(cv_docs)
pdfs = pdf_loader.load()

documents = pdfs + webs1 + webs2

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 700,
    chunk_overlap = 100
)

doc = text_splitter.split_documents(documents)


model_name = 'hkunlp/instructor-base'

embedding_model = HuggingFaceInstructEmbeddings(
    model_name = model_name,
    model_kwargs = {"device" : device}
)

vector_path = 'vector-store/'
if not os.path.exists(vector_path):
    os.makedirs(vector_path)
    print('create path done')



vectordb = FAISS.from_documents(
    documents = doc,
    embedding = embedding_model
)

db_file_name = 'myself_db'

vectordb.save_local(
    folder_path = os.path.join(vector_path, db_file_name),
    index_name = 'myself' #default index
)


history = ChatMessageHistory()


vectordb = FAISS.load_local(
    folder_path = os.path.join(vector_path, db_file_name),
    embeddings = embedding_model,
    index_name = 'myself', #default index
     allow_dangerous_deserialization=True
)   

retriever = vectordb.as_retriever()

history.add_user_message('hi')
history.add_ai_message('Whats up?')
history.add_user_message('How are you')
history.add_ai_message('I\'m quite good. How about you?')


memory = ConversationBufferMemory(return_messages = True)
memory.save_context({'input':'hi'}, {'output':'What\'s up?'})
memory.save_context({"input":'How are you?'},{'output': 'I\'m quite good. How about you?'})
memory.load_memory_variables({})



llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=""
)

question_generator = LLMChain(
    llm = llm,
    prompt = CONDENSE_QUESTION_PROMPT,
    verbose = True
)

doc_chain = load_qa_chain(
    llm = llm,
    chain_type = 'stuff',
    prompt = PROMPT,
    verbose = True
)


memory = ConversationBufferWindowMemory(
    k=3, 
    memory_key = "chat_history",
    return_messages = True,
    output_key = 'answer'
)

chain = ConversationalRetrievalChain(
    retriever=retriever,
    question_generator=question_generator,
    combine_docs_chain=doc_chain,
    return_source_documents=True,
    memory=memory,
    verbose=True,
    get_chat_history=lambda h : h
)

import sys

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/similarity', methods=['POST'])
def similarity():

    # try:
    data = request.json
    sentence = data.get('word')

    prompt_question = sentence
    answer = chain({"question":prompt_question})


    if not sentence:
        return jsonify({"error": "Word is required"}), 400
    
    source_all = ""
    for source in answer['source_documents']:
        source_all = source_all + ", " + str( source.metadata['source'])

    # print(answer['source_documents'][0]['source'], file=sys.stderr)

    return jsonify({
        "input_word": sentence,
        "similar_word": str(answer['answer']),
        "document":str(source_all)
    })
    # except Exception as e:
    #     return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)



