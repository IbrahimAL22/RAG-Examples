import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
import PyPDF2
from transformers import logging

# Suppress transformers warnings
logging.set_verbosity_error()

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
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

# Function to truncate context if it exceeds a certain token limit
def truncate_context(context, tokenizer, max_tokens=4096 - 200):  # Adjust based on model's max tokens
    tokens = tokenizer.encode(context, truncation=False)
    if len(tokens) > max_tokens:
        tokens = tokens[-max_tokens:]  # Keep the last max_tokens
        context = tokenizer.decode(tokens, skip_special_tokens=True)
    return context

# Path to your PDF file
pdf_path = 'C:/Users/Ibrahim/Desktop/rag/Linux-Tutorial.pdf'

# Extract text from the PDF
pdf_text = extract_text_from_pdf(pdf_path)

# Chunk the text
chunks = chunk_text(pdf_text)

# Initialize Hugging Face embeddings (this replaces OpenAIEmbeddings)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create a FAISS vector store from the chunks
vector_store = FAISS.from_texts(chunks, embedding_model)

# Load the local Qwen model and tokenizer (from the path you provided)
model_path = "C:/Users/Ibrahim/Desktop/rag/Qwen"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Set up the pipeline for text generation with your local model
text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0,  # Use GPU if available; set to -1 for CPU
    max_new_tokens=200,  # Adjust as needed
    temperature=0.7,  # Adjust for creativity
    top_p=0.9,
    do_sample=True
)

# Wrap the pipeline with LangChain's HuggingFacePipeline
local_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Create a RetrievalQA chain using the local LLM
retrieval_qa = RetrievalQA.from_chain_type(
    llm=local_llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 relevant chunks
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

# Retrieve top 5 relevant chunks
retrieved_chunks = retrieval_qa.retriever.get_relevant_documents(query)[:5]
context = "\n".join([doc.page_content for doc in retrieved_chunks])

# Truncate context if necessary
context = truncate_context(context, tokenizer, max_tokens=4096 - 200)

# Format the prompt
formatted_query = prompt_template.format(context=context, question=query)

# Perform Retrieval-Augmented Generation (RAG) on the formatted query
result = retrieval_qa.run(formatted_query)


print("Answer:", result)
