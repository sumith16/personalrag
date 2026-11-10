from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"

def format_docs(docs):
    return "\n\n".join(f"\n{doc.page_content}" for doc in docs)

def setup_rag_chain():
    # Load existing vector store
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Custom Prompt
    prompt_template = """
    You are a helpful assistant answering questions based strictly on the provided context.
    If the context does not contain the answer, state that you don't know based on the provided documents.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatOllama(model=LLM_MODEL, temperature=0.2)

    # Construct Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def interactive_chat():
    try:
        chain = setup_rag_chain()
    except Exception as e:
        print("Error loading vector store. Did you run ingest.py first?")
        return

    print("\n--- Personal RAG Active (Type 'exit' to quit) ---\n")
    while True:
        user_query = input("Ask a question: ")
        if user_query.strip().lower() in ["exit", "quit"]:
            break
        if not user_query.strip():
            continue

        print("\nThinking...\n")
        response = chain.invoke(user_query)
        print(f"Response:\n{response}\n")
        print("-" * 50)

if __name__ == "__main__":
    interactive_chat()