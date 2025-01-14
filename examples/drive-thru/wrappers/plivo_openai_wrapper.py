
from rag.pinecone_index import PineconeIndex
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain.agents import create_openai_tools_agent, AgentExecutor
from pinecone import Pinecone

system_prompt = """
You are Billy, salesperson at KFC. Your job is to take customer orders. \
Make yourself sound exciting and welcoming to the user. Do not use any emojis or special characters. \
Use fillers like 'Hmm,' 'Well,' 'You know,' or 'Let me think' occasionally to make the conversation more lifelike. \
Only share the entire menu along with all the items by category when asked. Mention Prices and Nutritional info only when asked. \
you need to call different tools and retrive_tool for RAG to answer KFC menu related queries from the provided content only. \
After sharing the menu, ask what the customer would like to order. After each order, confirm if they want anything else. \
If they say no, thank them, generate a random order Id between 1-1000 and always pass the information along with total price, items, item price and quantity to the create_order tool. \
provide the total price and the same order Id to the customer once the order is successfully created by the create_order tool, and end the conversation. \
Be concise, friendly, and helpful. Use only words, numbers, and essential punctuation like '?' or '!'.
Answer questions related to orders only. For all other questions, politely say that you are not sure about the answer and ask the customer to ask a question related to orders. \
Make tool calls with all the parameters required by the toolwhen needed or requested. before making tool calls, add some fillers like 'Sure, let me get the order details for you' or 'Sure, let me create the order for you' or 'Sure, let me check on that for you' or 'I'll get that right away' or 'I'll check on that for you
"""

class PlivoOpenAIWrapper:
    def __init__(self,
                 llm_model,
                 pinecone_index_name=None,
                 pinecone_data_file_name=None,
                 retriever_tool_name=None,
                 retriever_tool_description=None):
        self.llm_model = llm_model
        self.pinecone_index_name = pinecone_index_name
        self.pinecone_data_file_name = pinecone_data_file_name
        self.retriever_tool_name = retriever_tool_name
        self.retriever_tool_description = retriever_tool_description
        self.llm_tools = []
        
    def get_llm_model(self):
        return self.llm_model
        
    def set_llm_tools(self, tools):
        self.llm_tools = tools
    
    def get_llm_tools(self):
        return self.llm_tools
    
    def create_llm_service(self):
        return ChatOpenAI(model=self.llm_model)
    
    def get_llm_agent_executor(self):
        agent = create_openai_tools_agent(
            llm=self.create_llm_service(), 
            tools=self.get_llm_tools(), 
            prompt=self.create_llm_prompt()
        )
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=self.get_llm_tools(), 
            verbose=True
        )
        return agent_executor

    def create_llm_prompt(self):
        # remove the promt template or messages placeholder which are not required
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template=system_prompt)),
                MessagesPlaceholder(variable_name='chat_history', optional=True),
                HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}')),
                MessagesPlaceholder(variable_name='agent_scratchpad')
            ]
        )
    
        return answer_prompt

    def create_retriever_tool(self):
        if not self.pinecone_index_name or not self.retriever_tool_name or not self.retriever_tool_description:
            raise ValueError("Pinecone index name, retriever tool name and retriever tool description are required")

        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        pinecone_index = PineconeIndex(self.pinecone_index_name, pc, self.pinecone_data_file_name)
        retriever = pinecone_index.set_and_retrieve_pinecone_data(
            index_name=self.pinecone_index_name,
        )
        return create_retriever_tool(
            retriever, 
            name=self.retriever_tool_name,
            description=self.retriever_tool_description)