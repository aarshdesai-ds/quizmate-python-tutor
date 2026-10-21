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
    import sys

    print("--- parse (free) ---")
    docs = load_notes()
    by_topic = {d.metadata["topic"]: d for d in docs}
    print(f"{len(docs)} notes:", sorted(by_topic))
    print("ok  " if len(docs) == 12 else "FAIL", "12 notes")
    d = by_topic["dictionaries"]
    print("ok  " if d.metadata == {"topic": "dictionaries", "title": "Dictionaries", "level": "beginner"} else "FAIL",
          "dictionaries metadata:", d.metadata)
    print("ok  " if d.page_content.startswith("Dictionaries (dictionaries, beginner)") else "FAIL",
          "header line:", d.page_content.splitlines()[0])
    print("ok  " if all(s in d.page_content for s in ["## Key idea", "## Example", "## Common mistakes"]) else "FAIL",
          "all three sections present")
    levels = sorted(doc.metadata["level"] for doc in docs)
    print("ok  " if levels.count("beginner") ==  8 and levels.count("intermediate") == 4 else "FAIL",
          "8 beginner, 4 intermediate")

    if len(sys.argv) > 1 and sys.argv[1] == "build":
        print("\n--- build the store (one embedding call) ---")
        store = build_store()
        print("stored:", len(store.get()["ids"]))
        opened = load_store()
        print("ok  " if len(opened.get()["ids"]) == 12 else "FAIL", "load_store sees the same 12, without adding")
        for query in ["how do I avoid a KeyError", "what does range stop at", "catch an error"]:
            hits = opened.similarity_search(query, k=2)
            print(f"{query!r}: {[h.metadata['topic'] for h in hits]}")