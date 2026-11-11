## Development of a PDF-Based Question-Answering Chatbot Using LangChain

### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT:
To develop an intelligent chatbot that can read and understand the content of a PDF document and provide accurate, context-based answers to user questions. The system uses LangChain, document embeddings, FAISS vector search, and a large language model to retrieve relevant information and generate answers. Its effectiveness is evaluated using different types of questions.

### DESIGN STEPS:

#### STEP 1:
Upload the PDF document.
Extract text from the PDF using PyPDFLoader.
Split the extracted text into smaller chunks using RecursiveCharacterTextSplitter.
Set the chunk size and overlap to preserve contextual information.
#### STEP 2:
Convert the text chunks into numerical embeddings using a Hugging Face embedding model.
Store the embeddings in a FAISS vector database.
Configure a retriever to find the most relevant chunks for a user's query.
#### STEP 3:
Configure a Groq LLM using LangChain.
Create a retrieval-based question-answering chain.
Retrieve relevant document chunks and generate context-based answers.
Test the chatbot with factual, conceptual, specific-detail, and out-of-document questions.
Evaluate responses based on correctness, relevance, and groundedness.
### PROGRAM:
```
import os
import getpass
import pandas as pd

from google.colab import files
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)

# Configure Groq API key
os.environ["GROQ_API_KEY"] = getpass.getpass(
    "Enter your Groq API key: "
)

# Upload PDF
uploaded = files.upload()
pdf_path = next(iter(uploaded.keys()))

# Load PDF
loader = PyPDFLoader(pdf_path)
documents = loader.load()

print("Total pages:", len(documents))

# Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

print("Total chunks:", len(chunks))

# Generate embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS vector store
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embedding_model
)

# Configure LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

# Create prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question using only the provided context.
If the answer is unavailable, say:
"The answer is not available in the provided document."

Context:
{context}

Question:
{input}

Answer:
""")

# Create QA chain
document_chain = create_stuff_documents_chain(
    llm, prompt
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

qa_chain = create_retrieval_chain(
    retriever, document_chain
)

# Ask questions
def ask_question(question):
    response = qa_chain.invoke({"input": question})

    print("Question:", question)
    print("Answer:", response["answer"])

    print("\nSources:")
    for doc in response["context"]:
        print("Page:", doc.metadata.get("page", 0) + 1)

    return response
```
### OUTPUT:
<img width="1535" height="777" alt="image" src="https://github.com/user-attachments/assets/287f47a1-6f81-48d6-9321-0d1232513468" />
<img width="1535" height="816" alt="image" src="https://github.com/user-attachments/assets/4076f2ad-9a23-45e5-8501-39c86505ec80" />

### RESULT:
Thus, a PDF-based question-answering chatbot was successfully designed and implemented using LangChain, FAISS, Hugging Face embeddings, and Groq. The chatbot retrieves relevant information from the uploaded PDF and generates context-based answers. Its effectiveness is evaluated using diverse queries derived from the document's content.
