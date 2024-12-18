# --- Step 1: Setup Environment ---

!pip install langchain_community langchain_huggingface langchain_cohere langchain-google-genai gradio pypdf protobuf==3.20.3 tiktoken chromadb

import os
import pickle
import asyncio
import gradio as gr
import re
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_cohere import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

# --- Step 2: API Keys ---
os.environ['COHERE_API_KEY'] = "X3fELtKQ5j15URj5YtnxW60tNukBr0kd00RrgePQ"
os.environ['LANGCHAIN_API_KEY'] = "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"
os.environ['GOOGLE_API_KEY'] = "AIzaSyAZd21CLzG23nzsMkmshBKHgMOijEl8shI"

# --- Step 3: Upload Required Files ---
from google.colab import files

print("Please upload the required PDF files.")
uploaded_files = files.upload()

pdf_files = list(uploaded_files.keys())
print(f"Uploaded PDFs: {pdf_files}")

# --- Step 4: Preprocess the PDFs ---
# Load and split the content
fetch_text = []
for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()
    fetch_text.extend(documents)

# Split the content into chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=50
)
splits = text_splitter.split_documents(fetch_text)

# Create a vectorstore
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=HuggingFaceEmbeddings()
)

def vector_store():
    return vectorstore

# --- Step 5: Define the Model Chain ---
# Prompt template
template = """You are an intelligent and professional healthcare assistant, highly skilled in answering questions about diseases, symptoms, and treatment methods in a clear and concise manner. You can retrieve relevant medical information from a vector store to provide accurate answers, much like a knowledgeable doctor.

If a user provides a query describing symptoms, you can analyze the symptoms and identify possible diseases that could be associated with them. You then provide some possible results, explain each briefly, and always recommend that the user consult a healthcare professional for an accurate diagnosis and treatment plan.

When you don't know the answer to a question or can't find sufficient information, you first search other reliable online resources to try to find the answer. If you still can't find the answer, you admit it and apologize. You then provide the user with reliable links where they can learn more about the disease or symptoms.

Answer the question based on your knowledge:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

# Define the chain
chain = prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))

# --- Step 6: Define the Preprocessing Function ---
def clean_text(text_list):
    # Filter out empty lines
    cleaned_lines = [line for line in text_list if line.strip()]
    # Remove markdown formatting (**bold**)
    cleaned_lines = [re.sub(r'\*\*(.*?)\*\*', r'\1', line) for line in cleaned_lines]
    # Format the list items for better readability
    cleaned_lines = [line.replace("1. ", "\n1. ").replace("2. ", "\n2. ").replace("3. ", "\n3. ").replace("4. ", "\n4. ") for line in cleaned_lines]
    return cleaned_lines

# --- Step 7: Define the Gradio Interface ---
retriever = vector_store()

async def healthcare(text):
    if not text.strip():
        return None, gr.Warning("Please enter a text to convert.")

    docs = retriever.similarity_search(text)
    ans = chain.invoke({"context": docs, "question": text})
    cleaned_text = clean_text(ans)
    final_output = "\n".join(cleaned_text)
    return final_output

def gradio_interface(text):
    output = asyncio.run(healthcare(text))
    return output

def gradio_app():
    app = gr.Interface(
        fn=gradio_interface,
        inputs="textbox",
        outputs="textbox",
        title="Health Care Assistant Using RAG",
        description="AI Healthcare Assistant offering instant, accurate insights on diseases, symptoms, and treatments for personalized care.",
        analytics_enabled=False,
        flagging_mode="never"
    )
    return app

# --- Step 8: Launch the Gradio App ---
if __name__ == "__main__":
    demo = gradio_app()
    demo.launch(debug=True)
