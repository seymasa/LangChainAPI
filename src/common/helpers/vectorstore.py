from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter

from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
import shutil
import glob


chat_model = ChatOpenAI(openai_api_key="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", temperature=0)
openai_ef = OpenAIEmbeddings(openai_api_key="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")

#LOAD DATA
loader_ts = PyPDFLoader("/Users/seyma.sarigil/Documents/hackathon-getir/LangChainAPI/src/common/helpers/pdfs/DATA-Customer Trust Score-040623-180555.pdf")
loader_cs = PyPDFLoader("/Users/seyma.sarigil/Documents/hackathon-getir/LangChainAPI/src/common/helpers/pdfs/CUS-Customer Squad Home-050623-162809.pdf")
loader_fb = PyPDFLoader("/Users/seyma.sarigil/Documents/hackathon-getir/LangChainAPI/src/common/helpers/pdfs/DATA-Customer Feedback Monitoring-040623-180436.pdf")
loader_onoff = PyPDFLoader("/Users/seyma.sarigil/Documents/hackathon-getir/LangChainAPI/src/common/helpers/pdfs/DATA-How it works-060623-040259.pdf")
loader_care = TextLoader("/Users/seyma.sarigil/Documents/hackathon-getir/LangChainAPI/src/common/helpers/docs/care.txt")
loader_employees = TextLoader("/Users/seyma.sarigil/Documents/hackathon-getir/LangChainAPI/src/common/helpers/docs/employees.txt")


#PROCESS DATA
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len)

docs_ts = loader_ts.load_and_split()
docs_cs = loader_cs.load_and_split()
docs_fb = loader_fb.load_and_split()
docs_onoff = loader_onoff.load_and_split()
docs_care = loader_care.load_and_split(text_splitter=text_splitter)
docs_employees = loader_employees.load_and_split(text_splitter=text_splitter)

#DELETE VSTORE DIRECTORIES IF EXISTS vstore
directories = glob.glob('vstore*')

for directory in directories:
    try:
        shutil.rmtree(directory)
        print(f"{directory} başarıyla silindi.")
    except OSError as e:
        print(f"Hata: {e.filename} - {e.strerror}.")


#CREATE VECTOR STORES
vstore_ts = Chroma.from_documents(docs_ts, openai_ef, collection_name="customer-trust-score", persist_directory="vstore_ts")
vstore_cs = Chroma.from_documents(docs_cs, openai_ef, collection_name="product-customer-squad", persist_directory="vstore_cs")
vstore_fb = Chroma.from_documents(docs_fb, openai_ef, collection_name="nlp-feedback-classification", persist_directory="vstore_fb")
vstore_onoff = Chroma.from_documents(docs_onoff, openai_ef, collection_name="onoff-project-demand-optimization", persist_directory="vstore_onoff")
vstore_care =  Chroma.from_documents(docs_care, openai_ef, collection_name="health-care-insurance", persist_directory="vstore_care")
vstore_employees =  Chroma.from_documents(docs_employees, openai_ef, collection_name="employees", persist_directory="vstore_employee")


#CREATE LLM CHAINS
chain_ts = RetrievalQA.from_chain_type(llm=chat_model, chain_type="stuff", retriever=vstore_ts.as_retriever())
chain_cs = RetrievalQA.from_chain_type(llm=chat_model, chain_type="stuff", retriever=vstore_cs.as_retriever())
chain_fb = RetrievalQA.from_chain_type(llm=chat_model, chain_type="stuff", retriever=vstore_fb.as_retriever())
chain_onoff = RetrievalQA.from_chain_type(llm=chat_model, chain_type="stuff", retriever=vstore_onoff.as_retriever())
chain_care = RetrievalQA.from_chain_type(llm=chat_model, chain_type="stuff", retriever=vstore_care.as_retriever())
chain_employee = RetrievalQA.from_chain_type(llm=chat_model, chain_type="stuff", retriever=vstore_employees.as_retriever())


#CREATE TOOLS
tools = [
    Tool(
        name = "Customer Trust Score Project QA System",
        func=chain_ts.run,
        description="useful for when you need to answer questions about the customer trust score project which is a segmentation project for customer services. Input should be a fully formed question. Output should include names of authors of document for further details."
    ),
    Tool(
        name = "Customer Squad QA System",
        func=chain_cs.run,
        description="useful for when you need to answer questions about the cross-domain customer squad under product department. Input should be a fully formed question. Output should include names of authors of document and related people mentioned in document for further details."
    ),
    Tool(
        name = "NLP, Comment and Feedback monitoring projects QA System",
        func=chain_fb.run,
        description="useful for when you need to answer questions about the natural language processing projects under Getir that involve analysis of comments and feedbacks from customers. Input should be a fully formed question. Output should include names of authors of document and team member names for further details."
    ),

    Tool(
        name = "Demand Optimization On-Off Project QA System",
        func=chain_onoff.run,
        description="useful for when you need to answer questions about the demand optimization on-off project. which is project from data department. Input should be a fully formed question. Output should direct to Ozan Can Eren for further details."
    ),

     Tool(
        name = "Getir Care for Healthcare and insurance topics",
        func=chain_care.run,
        description="useful for when you need to answer questions about the insurance and healthcare topics within getir. direct to most frequent announcement message posters."
    ),
    Tool(
        name = "Getir Employee Database",
        func=chain_employee.run,
        description="useful for when you want to inquire people in a specfic department and who manages who."
    ),
]

#CREATE AGENT
agent = initialize_agent(tools, chat_model, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
