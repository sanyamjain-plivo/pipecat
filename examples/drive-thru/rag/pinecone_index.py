
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import ServerlessSpec
import openai
import os



class PineconeIndex:
    def __init__(self, index_name, pc, file_name):
        self.index_name = index_name
        self.pc = pc
        self.file_name = file_name

    def set_data_to_pinecone(self, data):    
        existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]

        docs = self.embed_text(data)
        if self.index_name not in existing_indexes:
            print("Creating index")
            self.pc.create_index(
                name=self.index_name,
                dimension=1536, # Replace with your model dimensions
                metric="cosine", # Replace with your model metric
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                ) 
            )
    
            while not self.pc.describe_index(self.index_name).status['ready']:
                print("Waiting for index to be ready")
                time.sleep(1)
    
            embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY")) #OPENAI API KEY
            PineconeVectorStore.from_texts(docs, embeddings, index_name=self.index_name)
        
    def set_and_retrieve_pinecone_data(self, index_name):
        raw_data = open(self.file_name, "r").read()
        self.set_data_to_pinecone(raw_data)
        
        vector_store = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY")),
        )
    
        return vector_store.as_retriever()

    def embed_text(self, text:str):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False)
            
        split_text = text_splitter.split_text(text)
        return split_text