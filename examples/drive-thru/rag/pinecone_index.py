
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import ServerlessSpec
import openai
import os

def set_data_to_pinecone(pc, data):
    index_name = "drive-thru-menu"
    
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    docs = embed_text(data)
    if index_name not in existing_indexes:
        print("Creating index")
        pc.create_index(
            name=index_name,
            dimension=1536, # Replace with your model dimensions
            metric="cosine", # Replace with your model metric
            spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
            ) 
        )
    
        while not pc.describe_index(index_name).status['ready']:
            print("Waiting for index to be ready")
            time.sleep(1)
    
        index = pc.Index(index_name)
        print(f"index is {index}")
        print(f"updating index {index_name}")
        embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY")) #OPENAI API KEY
        PineconeVectorStore.from_texts(docs, embeddings, index_name=index_name)
        
    

def embed_text(text:str):
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False)
    
    split_text = text_splitter.split_text(text)
    return split_text



def generate_embeddings(text:str):
    embeddings = openai.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return embeddings
