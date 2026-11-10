import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

DOCS_DIR = "./docs"
DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"

def ingest_documents():
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"Created '{DOCS_DIR}' folder. Place your PDFs, TXT, or MD files inside it and rerun.")
        return

    print("Loading documents...")
    # Load PDFs and text files
    pdf_loader = DirectoryLoader(DOCS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader)
    md_loader = DirectoryLoader(DOCS_DIR, glob="**/*.md", loader_cls=TextLoader)

    documents = []
    documents.extend(pdf_loader.load())
    documents.extend(txt_loader.load())
    documents.extend(md_loader.load())

    if not documents:
        print("No documents found in ./docs/")
        return

    print(f"Loaded {len(documents)} document(s). Chunking text...")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunk(s). Generating embeddings...")

    # Initialize embeddings
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # Persist chunks to ChromaDB
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    
    print(f"Successfully ingested {len(chunks)} chunks into vector store at '{DB_DIR}'.")

if __name__ == "__main__":
    ingest_documents()