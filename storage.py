import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore

def config_for(learner_id: str, session_id: str) -> dict:
    thread_id = f"{learner_id}-{session_id}"
    return {'configurable': {'thread_id': thread_id}}

DB_PATH = "quizmate.db"

checkpoint_conn = sqlite3.connect(DB_PATH, check_same_thread=False)

store_conn = sqlite3.connect(DB_PATH, check_same_thread=False, isolation_level=None)

checkpointer = SqliteSaver(conn = checkpoint_conn)

store = SqliteStore(store_conn)

store.setup()
