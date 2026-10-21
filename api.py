from typing import Literal, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from pydantic import BaseModel

from chat_graph import learners, workflow
from storage import config_for
from tools import TOPICS

app = FastAPI(title="QuizMate", description="A Python tutor with quizzes and memory, built with LangGraph.")


class StartSession(BaseModel):
    learner_id: str

class Message(BaseModel):
    text: str

class QuizStart(BaseModel):
    topic: str

class Answer(BaseModel):
    answer: str

class SessionResponse(BaseModel):
    session_id: str
    learner_id: str
    name: str

class TurnResponse(BaseModel):
    status: Literal["paused", "done"]
    kind: Optional[Literal["quiz_question", "forget_confirm"]] = None
    pause: Optional[dict] = None
    reply: Optional[str] = None

class HistoryResponse(BaseModel):
    session_id: str
    summary: str
    messages: list[dict]


def session_config(session_id: str):
    learner_id, _, short_id = session_id.partition("-")
    if learner_id not in learners or not short_id:
        raise HTTPException(status_code=404, detail=f"Unknown session: {session_id}")
    config = {**config_for(learner_id, short_id), "run_name": "quizmate-api", "tags": ["quizmate", "api"],
              "metadata": {"learner_id": learner_id, "session_id": short_id}}
    return learner_id, config

def is_paused(config):
    return bool(workflow.get_state(config).next)

def require_not_paused(config):
    if is_paused(config):
        raise HTTPException(status_code=409, detail="Answer the pending question first, with /resume.")

def run_graph(graph_input, config):
    try:
        result = workflow.invoke(graph_input, config=config)
    except Exception:
        raise HTTPException(status_code=502, detail="The tutor could not answer. Please try again.")
    if result.get("__interrupt__"):
        payload = result["__interrupt__"][0].value
        kind = "quiz_question" if "options" in payload else "forget_confirm"
        return TurnResponse(status="paused", kind=kind, pause=payload)
    return TurnResponse(status="done", reply=result["messages"][-1].content)


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/sessions", response_model=SessionResponse)
def start_session(body: StartSession):
    learner_id = body.learner_id.strip().lower()
    if learner_id not in learners:
        raise HTTPException(status_code=404, detail=f"Unknown learner: {learner_id}. Learners: {', '.join(learners)}")
    session_id = f"{learner_id}-{uuid4().hex[:6]}"
    return SessionResponse(session_id=session_id, learner_id=learner_id, name=learners[learner_id]["name"])

@app.post("/sessions/{session_id}/messages", response_model=TurnResponse)
def send_message(session_id: str, body: Message):
    learner_id, config = session_config(session_id)
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="The message is empty.")
    require_not_paused(config)
    return run_graph({"messages": [HumanMessage(content=text)], "learner_id": learner_id}, config)

@app.post("/sessions/{session_id}/quiz", response_model=TurnResponse)
def start_quiz(session_id: str, body: QuizStart):
    learner_id, config = session_config(session_id)
    topic = body.topic.strip().lower()
    if topic not in TOPICS:
        raise HTTPException(status_code=422, detail=f"Unknown topic: {topic}. Topics: {', '.join(TOPICS)}")
    require_not_paused(config)
    return run_graph({"quiz_topic": topic, "learner_id": learner_id}, config)

@app.post("/sessions/{session_id}/forget", response_model=TurnResponse)
def forget_me(session_id: str):
    learner_id, config = session_config(session_id)
    require_not_paused(config)
    return run_graph({"messages": [], "learner_id": learner_id, "forget": True}, config)

@app.post("/sessions/{session_id}/resume", response_model=TurnResponse)
def resume(session_id: str, body: Answer):
    _, config = session_config(session_id)
    answer = body.answer.strip()
    if not answer:
        raise HTTPException(status_code=422, detail="The answer is empty.")
    if not is_paused(config):
        raise HTTPException(status_code=409, detail="Nothing is waiting for an answer.")
    return run_graph(Command(resume=answer), config)

@app.get("/sessions/{session_id}/history", response_model=HistoryResponse)
def history(session_id: str):
    _, config = session_config(session_id)
    values = workflow.get_state(config).values
    messages = []
    for message in values.get("messages", []):
        if isinstance(message, HumanMessage):
            messages.append({"role": "learner", "content": message.content})
        elif isinstance(message, AIMessage) and not message.tool_calls:
            messages.append({"role": "tutor", "content": message.content})
    return HistoryResponse(session_id=session_id, summary=values.get("summary", ""), messages=messages)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)