from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
import os


class RAGPipeline:

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.embeddings = OpenAIEmbeddings()  # Uses OpenAI embeddings
        self.llm = OpenAI(temperature=0)
        self.vector_store = None
        self.qa_chain = None

    def load_and_index(self):
        # Load PDF
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(documents)

        # Create vector store
        self.vector_store = FAISS.from_documents(docs, self.embeddings)

        # Create retrieval chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vector_store.as_retriever()
        )

    def ask(self, query: str):
        if not self.qa_chain:
            raise Exception("Index not built. Call load_and_index() first.")

        response = self.qa_chain.run(query)
        return response