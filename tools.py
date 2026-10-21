from ingest import load_store
from langchain_core.tools import tool
from typing import Optional, Literal

store = load_store()
TOPICS = [
    "variables", "strings", "lists", "tuples", "dictionaries", "sets",
    "conditionals", "loops", "functions", "comprehensions", "exceptions", "files",
]

@tool
def search_notes(query: str, topic: Optional[Literal["variables", "strings", "lists", "tuples", "dictionaries", "sets", "conditionals", "loops", "functions","comprehensions", "exceptions", "files"]] = None) -> str:
    """
Search the Python course notes for the ones that match a question.

Use this when the learner asks about a Python idea, an error, or the difference
between two things, and you need to find the right note.

Args:
    query: a short description in plain words, e.g. "how do I avoid a KeyError"
        or "difference between append and extend".
    topic: optional. One of "variables", "strings", "lists", "tuples",
        "dictionaries", "sets", "conditionals", "loops", "functions",
        "comprehensions", "exceptions" or "files". Leave empty unless the
        learner named a topic.

Returns up to 3 notes, each as a "topic | title | level" line followed by the
note's key idea. Use get_note with the topic to read the full note, including
its example and common mistakes.
    """
    if topic:
        notes = store.similarity_search(query, k = 3, filter = {"topic": topic})
    else:
        notes = store.similarity_search(query, k = 3)
    
    if not notes:
        if topic: 
            return f"No notes found for topic: {topic}."
        else:
            return f"No notes found."
    blocks = []
    for note in notes:
        key_idea_text = note.page_content.split("##")[1].strip()
        key_idea = key_idea_text.split("\n", maxsplit=1)[1].strip()

        blocks.append(f"{note.metadata['topic']} | {note.metadata['title']} | {note.metadata['level']}\n{key_idea}")
    return "\n\n".join(blocks)

@tool
def get_note(topic: str) -> str:
    """
Get the full text of one Python course note.

Use this when you need a note's example or common mistakes, or to explain a
topic in full. Find the topic first with search_notes if you are not sure which
note is right.

Args:
    topic: exactly one of "variables", "strings", "lists", "tuples",
        "dictionaries", "sets", "conditionals", "loops", "functions",
        "comprehensions", "exceptions" or "files".

Returns the note's title line, its key idea, an example in Python and its
common mistakes. If the topic is not one of these, it says so and lists the
valid topics: do not explain a topic that is not in the notes.
    """

    topic = topic.lower().strip()
    if topic not in TOPICS:
        topics = ", ".join(TOPICS)
        return f"'{topic}' is not one of the course topics.\nTopics: {topics}"
    note = store.get_by_ids([topic])
    if not note:
        topics = ", ".join(TOPICS)
        return f"'{topic}' is not one of the course topics.\nTopics: {topics}"
    return note[0].page_content

tools = [search_notes, get_note]

if __name__ == "__main__":
    import sys

    print("--- get_note (free: lookup by id, no embedding call) ---")
    loops = get_note.invoke({"topic": "loops"})
    print(loops[:120], "...\n")
    print("ok  " if loops.startswith("Loops (loops, beginner)") else "FAIL", "header line first")
    print("ok  " if all(s in loops for s in ["## Key idea", "## Example", "## Common mistakes"]) else "FAIL",
          "full note returned")
    unknown = get_note.invoke({"topic": "classes"})
    print("ok  " if "not" in unknown.lower() and "dictionaries" in unknown else "FAIL",
          "unknown topic lists the valid ones:", unknown[:90])
    print("tools:", [t.name for t in tools])

    if len(sys.argv) > 1 and sys.argv[1] == "run":
        print("\n--- search_notes (embedding calls only) ---")
        for args in [{"query": "how do I avoid a KeyError"},
                     {"query": "difference between append and extend"},
                     {"query": "anything", "topic": "sets"}]:
            result = search_notes.invoke(args)
            print(f"\n{args}:")
            print(result)
        one = search_notes.invoke({"query": "anything", "topic": "sets"})
        print("\nok  " if one.count(" | ") == 2 and "sets" in one else "\nFAIL", "topic filter returns only the sets note")