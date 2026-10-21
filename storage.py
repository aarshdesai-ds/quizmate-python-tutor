import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

def config_for(learner_id: str, session_id: str) -> dict:
    thread_id = f"{learner_id}-{session_id}"
    return {'configurable': {'thread_id': thread_id}}

DB_PATH = "quizmate.db"

conn = sqlite3.connect(DB_PATH, check_same_thread=False)

checkpointer = SqliteSaver(conn = conn)

