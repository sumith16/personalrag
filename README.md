# Personal RAG

A lightweight local Retrieval-Augmented Generation (RAG) project built with Python, LangChain, Ollama, and ChromaDB. It ingests your documents into a local vector database and lets you ask questions against them using a local LLM.

This project is ideal for personal knowledge bases, internal notes, PDFs, research material, and other document collections you want to query without sending data to a cloud API.

## Features

- Ingests PDF, TXT, and Markdown files from a local `docs/` folder
- Splits documents into chunks with overlap for better retrieval
- Uses local embeddings from Ollama (`nomic-embed-text`)
- Stores embeddings in a local ChromaDB vector database
- Answers questions using a local language model (`llama3.2`)
- Runs fully on your machine, keeping data local

## Project Structure

```text
personal_rag/
├── docs/               # Put your source documents here
├── chroma_db/          # Local vector database
├── ingest.py           # Loads and embeds documents
├── query.py            # Chat interface for Q&A
├── README.md           # Project documentation
└── .venv/              # Optional local virtual environment
```

## How It Works

1. Place your documents in `docs/`.
2. Run `ingest.py` to load files, split them into chunks, and embed them.
3. Run `query.py` to ask questions against the vector store.
4. The app retrieves the most relevant chunks and sends them as context to the LLM.

## Requirements

- Python 3.10+
- Ollama installed and running locally
- Internet access for the first model download

## Installation

Clone the project and install dependencies:

```bash
pip install langchain langchain-community langchain-core langchain-text-splitters langchain-ollama chromadb pypdf
```

If you are using a virtual environment, activate it first:

```bash
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
.venv\Scripts\activate       # Windows
pip install langchain langchain-community langchain-core langchain-text-splitters langchain-ollama chromadb pypdf
```

## Ollama Setup

Download the embedding model and the chat model used by the project:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

Make sure Ollama is running in the background before using the app.

## Add Documents

Place your files inside the `docs/` directory. Supported file types in the current implementation are:

- `.pdf`
- `.txt`
- `.md`

> Note: `.docx` files are present in this project folder but are not currently ingested by the script. If you want to support Word documents, add a DOCX loader to `ingest.py`.

## Run the Ingestion

From the project root:

```bash
python ingest.py
```

This will:

- load files from `./docs`
- split them into chunks
- generate embeddings
- save them into `./chroma_db`

## Run the Q&A App

```bash
python query.py
```

Then type your question in the terminal. For example:

```text
Ask a question: What is the main topic of the documents?
```

Type `exit` or `quit` to leave the chat.

## Example Workflow

```bash
python ingest.py
python query.py
```

## Notes

- The project is designed for local, private use.
- Retrieval quality depends on your document quality and chunk size.
- This is a simple RAG starter and can be extended with:
  - better document loaders
  - hybrid search
  - metadata filtering
  - conversation memory
  - API or web UI

## Future Improvements

- Add support for `.docx` and other document formats
- Add a web interface using Streamlit or FastAPI
- Add conversation history and memory
- Improve prompt engineering for more grounded responses
- Add a requirements.txt file for easier setup

## License

This project is provided as-is for educational and personal use.
