import os
import json
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.prompts import load_prompt
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

os.environ["OLLAMA_USE_GPU"] = "True"

# Function to read and process a text file
def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        return content

# Function to split the text into chunks
def split_text(content):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = [Document(page_content=content)]
    return splitter.split_documents(documents)

# Function to initialize and run LLaMA model
def initialize_llama_model():
    llm = Ollama(model="llama3.1:8b")  # Set use_gpu=True if supported by Ollama
    return llm

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Function to run the QA chain
def ask_question(llm, context, question):
    formatted_context = format_docs(context)
    prompt = f"Context: {formatted_context}\n\nQuestion: {question}\n\nAnswer:"
    result = llm(prompt)
    return result

# Function to parse the answer and format it as a dictionary
def parse_answer(answer):
    # Example: You might need a custom parsing logic depending on the answer format
    # Assuming answer contains labels like "patient name:", "age:", etc.
    lines = answer.split('\n')
    answer_dict = {}
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            answer_dict[key.strip()] = value.strip()
    return answer_dict

# Function to save the answer to a JSON file
def save_answer_to_json(question, answer):
    # Parse the answer into a dictionary
    answer_dict = parse_answer(answer)
    
    output = {
        "question": question,
        "answer": answer_dict
    }
    file_name = f"page15_{uuid.uuid4()}_2.json"
    output_path = os.path.join("ocr_and_model", file_name)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    print(f"Answer saved to {output_path}")

# Function to save the answer as plain text
def save_answer_to_text(question, answer):
    file_name = f"page15_{uuid.uuid4()}.txt"
    output_path = os.path.join("ocr_and_model", file_name)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Question: {question}\n\nAnswer:\n{answer}")
    print(f"Answer saved to {output_path}")

# Load the LLaMA model
llm = initialize_llama_model()

if __name__ == "__main__":
    while True:
        file_path = "data/page15_output.txt"  # Updated to point to a text file
        if file_path.lower() == "exit":
            break
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        
        content = load_text_file(file_path)  # Updated to load a text file
        if not content:
            print(f"No content found in the file: {file_path}")
            continue
        
        texts = split_text(content)
        
        question = "get all important info in this json format {patient name:etc....,Report Paramaters:{id:some id,name:rbc,unit:rbc unit,value:,biological reference:,{id:some id,name:etc...},....}"
        if question.lower() == "exit":
            break
        
        answer = ask_question(llm, texts, question)
        print(f"\nAnswer: {answer},")
        
        save_answer_to_json(question, answer)
        save_answer_to_text(question, answer)
