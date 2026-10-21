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

print("\n--- store (free) ---")
namespace = ("learner", "test", "facts")
store.put(namespace, "t1", {"data": "test fact"})
item = store.get(namespace, "t1")
print("ok  " if item and item.value["data"] == "test fact" else "FAIL", "put then get")
print("ok  " if len(store.search(namespace)) == 1 else "FAIL", "search finds it")
store.delete(namespace, "t1")
print("ok  " if store.get(namespace, "t1") is None else "FAIL", "deleted again (test leaves nothing behind)")