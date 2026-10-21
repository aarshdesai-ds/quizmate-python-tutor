from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

def parse_front_matter(text):
    metadata = {}
    if not text.startswith("---"):
        return ({},text)
    splits = text.split("---", maxsplit = 2)
    header = splits[1]
    body = splits[2]

    for line in header.strip().splitlines():
        if ": " in line:
            pair = line.split(": ", maxsplit = 1)
            attribute, value = pair[0], pair[1]
            metadata[attribute.strip()] = value.strip()
    return (metadata, body.strip())

def load_notes(folder="data/notes"):
    docs = []
    files = os.listdir(folder)
    for file in files:
        if not file.endswith(".md"):
            continue
        path = os.path.join(folder, file)
        text = open(path, encoding='utf-8').read()
        metadata, body = parse_front_matter(text)
        header = f"{metadata['title']} ({metadata['topic']}, {metadata['level']})"
        docs.append(Document(page_content=f"{header}\n{body}", metadata=metadata))
    return docs     

def build_store():
    notes = load_notes()
    ids = [doc.metadata['topic'] for doc in notes]
    store = Chroma(
        collection_name="notes",
        embedding_function= OpenAIEmbeddings(model = "text-embedding-3-small"),
        persist_directory="kb_db"
    )
    store.add_documents(notes, ids = ids)
    return store

def load_store():
    store = Chroma(
            collection_name="notes",
            embedding_function= OpenAIEmbeddings(model = "text-embedding-3-small"),
            persist_directory="kb_db"
    )
    return store

if __name__ == "__main__":
    store = build_store()
    print(f"Built the notes store: {len(store.get()['ids'])} notes in kb_db.")