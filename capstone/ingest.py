import os
import argparse
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

load_dotenv()

def ingest_documents(source_dir: str, persist_dir: str):
    print(f"Loading documents from {source_dir}...")
    
    # Load all txt and pdf files
    text_loader_kwargs={'autodetect_encoding': True}
    loaders = {
        ".txt": DirectoryLoader(source_dir, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs),
        ".pdf": DirectoryLoader(source_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    }
    
    docs = []
    for ext, loader in loaders.items():
        try:
            loaded = loader.load()
            print(f"Loaded {len(loaded)} documents from {ext} files.")
            docs.extend(loaded)
        except Exception as e:
            print(f"Error loading {ext} files: {e}")
            
    if not docs:
        print("No documents found to ingest.")
        return
        
    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks.")
    
    ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    embeddings = OllamaEmbeddings(model="llama3.2:3b", base_url=ollama_base_url)
    
    print(f"Ingesting into ChromaDB at {persist_dir}...")
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=persist_dir
    )
    print("Ingestion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB for RAG.")
    parser.add_argument("--source", type=str, default="data/raw_docs", help="Directory containing raw documents (.txt, .pdf)")
    parser.add_argument("--persist", type=str, default="data/chroma_db", help="Directory to persist ChromaDB")
    args = parser.parse_args()
    
    os.makedirs(args.source, exist_ok=True)
    os.makedirs(args.persist, exist_ok=True)
    
    ingest_documents(args.source, args.persist)
