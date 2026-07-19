from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
import tempfile
load_dotenv()
app=FastAPI(title="Document Q&A")
app.add_middleware(
CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)
llm=ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)
vector_store=None
class Question(BaseModel):
    question: str
@app.get("/")
def home():
    return{"message": "Welcome to the Document Q&A API. Upload a PDF and ask questions about its content."}
@app.get("/ui")
def ui():
    return FileResponse("index.html")
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global vector_store
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported."
                            )
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content=await file.read()
        tmp.write(content)
        tmp_path=tmp.name
    loader=PyPDFLoader(tmp_path)
    documents=loader.load()
    splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks=splitter.split_documents(documents)
    
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store=FAISS.from_documents(chunks,embeddings)
    os.unlink(tmp_path)
    return{
        "message": "File uploaded and processed successfully.",
        "pages": len(documents),
        "chunks": len(chunks)
    }
@app.post("/ask")
async def ask_question(q: Question):
    global vector_store
    
    if vector_store is None:
        raise HTTPException(
            status_code=400, 
            detail="Pehle PDF upload karo!"
        )
    
    # Relevant chunks dhundo
    docs = vector_store.similarity_search(query=q.question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # AI se jawab lo
    prompt = f"""Answer the question based on this PDF content only:

Context:
{context}

Question: {q.question}

Answer:"""
    
    from langchain_core.messages import HumanMessage
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "question": q.question,
        "answer": response.content
    }