# RAG_Project

!pip install --upgrade langchain openai
!pip install sentence_transformers -q
!pip install unstructured -q
!pip install unstructured[local-inference] -q
!pip install detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.6
!apt-get install poppler-utils
!pip install -U langchain-community
! pip install pymupdf
from langchain.document_loaders import PyMuPDFLoader
loader = PyMuPDFLoader("/content/library.pdf") # replace with your file
sheets = loader.load()
print(len(sheets))
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_docs(sheets, chunk_size=1500, chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(sheets)
  return docs

docs = split_docs(sheets)
print(len(docs))
print(docs[20].page_content)
from langchain.embeddings import SentenceTransformerEmbeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
query_result = embeddings.embed_query(docs[80].page_content)
print(len(query_result))
!pip install pinecone-client  -q
import os
from pinecone import Pinecone, ServerlessSpec
# Initialize Pinecone
pinecone_instance = Pinecone(api_key='Your API', environment='us-east-1')# find at app.pinecone.io
index = pinecone_instance.Index("rag-chatbot")
def get_similiar_docs(query,k=1,score=False):
  if score:
    similar_docs = index.similarity_search_with_score(query,k=k)
  else:
    similar_docs = index.similarity_search(query,k=k)
  return similar_docs

!pip install streamlit
!pip install streamlit_chat
!pip install pyngrok
!pip install langchain
!pip install streamlit streamlit-chat pymupdf
!pip install sentence_transformers
!python -m sentence_transformers.download all-MiniLM-L6-v2
!apt-get install poppler-utils
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
from utils import *

st.subheader("Chatbot with Langchain, ChatGPT, Pinecone, and Streamlit")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context,
and if the answer is not contained within the text below, say 'I don't know'""")


human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])
st.title("RAG Chatbot")
...
response_container = st.container()
textcontainer = st.container()
...
with textcontainer:
    query = st.text_input("Query: ", key="input")
    ...
with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="Your API")
...
conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

if query:
    with st.spinner("typing..."):
        ...
        response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
    st.session_state.requests.append(query)
    st.session_state.responses.append(response)

def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string

%%writefile utils.py
def query_refiner(conversation, query):
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['text']

!pip install pyngrok
from pyngrok import ngrok
ngrok.set_auth_token("Your API")
from threading import Thread
def run_sreamlit():
    os.system("streamlit run utils.py --server.port 8080")
thread = Thread(target=run_sreamlit)
thread.start()
from pyngrok import ngrok

# Close any existing tunnels
ngrok.kill()

# Open a tunnel on the port where your Streamlit app is running
public_tunnel = ngrok.connect(addr='8080', proto='http', bind_tls=True)

# Get the public URL
public_url = public_tunnel.public_url
print('Your URL is: '+ public_url)
