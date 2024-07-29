import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from langchain.prompts import load_prompt
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

# Function to read and process a JSON file
def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if "text" in data:
            content = " ".join(data["text"])
            return content
    return None

# Function to split the text into chunks
def split_text(content):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = [Document(page_content=content)]
    return splitter.split_documents(documents)

# Function to initialize and run LLaMA model
def initialize_llama_model():
    llm = Ollama(model="llama2:7b")
    return llm

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Function to run the QA chain
def ask_question(llm, context, question):
    formatted_context = format_docs(context)
    prompt = f"Context: {formatted_context}\n\nQuestion: {question}\n\nAnswer:"
    result = llm(prompt)
    return result

# Load the LLaMA model
llm = initialize_llama_model()

if __name__ == "__main__":
    while True:
        file_path = "donut_ocr/llama_all/data/combine_pdf_1_paddle+class.json"
        if file_path.lower() == "exit":
            break
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        
        content = load_json_file(file_path)
        if content is None:
            print(f"No text content found in the file: {file_path}")
            continue
        
        texts = split_text(content)
        
        question = input("Enter your question (or 'exit' to quit): ")
        if question.lower() == "exit":
            break
        
        answer = ask_question(llm, texts, question)
        print(f"\nAnswer: {answer}\n")
