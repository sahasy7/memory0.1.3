from sentence_transformers import SentenceTransformer
import openai
import streamlit as st
import qdrant_client
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant

openai.api_key = st.secrets.openai_key
QDRANT_API_KEY = st.secrets.QDRANT_API_KEY
QDRANT_HOST = st.secrets.QDRANT_HOST

embeddings = OpenAIEmbeddings()

def load_db():
    client = qdrant_client.QdrantClient(
        url=QDRANT_HOST,
        api_key=QDRANT_API_KEY,
    )
    vector_store = Qdrant(
        client = client,
        collection_name = "gsm_demo.0.0.4",
        embeddings = embeddings
    )
    print("connection established !")
    return vector_store

vectore_stor = load_db()

def find_match(input):
    input_em = embeddings.encode(input).tolist()
    result = vectore_stor.as_retriever(input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']

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

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string
