import os
import PyPDF2
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# Set OpenAI API key
OPENAI_API_KEY = ' '
OPENAI_ORGANIZATION = ''
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_ORGANIZATION"] = OPENAI_ORGANIZATION

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

# Function to chunk text into smaller pieces
def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(' '.join(current_chunk)) > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# Path to your PDF file
pdf_path = 'C:/Users/Ibrahim/Desktop/rag/Linux-Tutorial.pdf'

# Extract text from the PDF
pdf_text = extract_text_from_pdf(pdf_path)

# Chunk the text
chunks = chunk_text(pdf_text)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Create a FAISS vector store from the chunks
vector_store = FAISS.from_texts(chunks, embeddings)

# Initialize the ChatOpenAI LLM with GPT-3.5 Turbo
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, api_key=OPENAI_API_KEY, openai_organization=OPENAI_ORGANIZATION)

# Create a RetrievalQA chain
retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)

# Define a detailed prompt template
template = """
Based on the following context items, please answer the query.
If the query is not related to the context items, don't answer it or say 'I don't know'.
Answer in detail and with technical accuracy.

Context:
{context}

Query:
{question}

Response:
"""

# Create a PromptTemplate instance
prompt_template = PromptTemplate(input_variables=["context", "question"], template=template)

# Example query
query = "What is the main topic of the document?"

# Format the query using the prompt template
formatted_query = prompt_template.format(context='\n'.join(chunks), question=query)

# Perform RAG on the formatted query
result = retrieval_qa.run(formatted_query)

# Print the generated response
print("Question:", query)
print("Answer:", result)