from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import RAGPipeline
import os

# Set your OpenAI key
os.environ["OPENAI_API_KEY"] = "your_openai_key_here"

app = FastAPI()

rag = RAGPipeline("documents/sample.pdf")
rag.load_and_index()


class QueryRequest(BaseModel):
    question: str


@app.post("/ask")
def ask_question(request: QueryRequest):
    answer = rag.ask(request.question)
    return {"answer": answer}