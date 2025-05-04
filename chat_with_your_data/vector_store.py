from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import shutil

if __name__ == "__main__":
    current_dir = Path(__file__).parent
    loaders = [
        PyPDFLoader(current_dir / "data" / "tmp" / "0511061v1.pdf"),
        PyPDFLoader(
            current_dir
            / "data"
            / "tmp"
            / "NIPS-2017-attention-is-all-you-need-Paper.pdf"
        ),
    ]
    docs = []
    for loader in loaders:
        docs += loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    splits = text_splitter.split_documents(docs)
    print(f"Number of splits: {len(splits)}")
    embedding = OllamaEmbeddings(
        base_url="http://172.20.10.2:11434",
        model="all-minilm",
    )
    persist_directory = current_dir / "data" / "chroma"
    if persist_directory.is_dir():
        shutil.rmtree(persist_directory)
    vector_db = Chroma.from_documents(
        documents=splits, embedding=embedding, persist_directory=str(persist_directory)
    )
    question = "Where to use quantum computers?"
    results = vector_db.similarity_search(question, k=3)
    for result in results:
        print(result.page_content)
        print("-" * 80)
    vector_db.persist()
