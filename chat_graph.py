from tools import tools
from storage import checkpointer, store
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, RemoveMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from tools import TOPICS
from quiz_graph import quiz_subgraph
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from langgraph.types import interrupt
from pydantic import BaseModel, Field
import json
from memory import facts_namespace, scores_namespace, read_facts, extract_memories, save_new_facts, read_scores, update_score, format_memory, forget_learner
from langgraph.store.base import BaseStore
from typing import TypedDict, Annotated, Literal
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)
model_with_tools = model.bind_tools(tools)

with open("data/learners.json", encoding="utf-8") as f:
    learners = json.loads(f.read())

class NextTopic(BaseModel):
    topic: Literal["variables", "strings", "lists", "tuples", "dictionaries", "sets", "conditionals", "loops", "functions","comprehensions", "exceptions", "files"] = Field(... , description="The one topic the learner should study next")
    reason: str = Field(... , description="One short sentence to the learner, saying why this topic, based on what they missed or passed")

structured_model = model.with_structured_output(NextTopic)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    learner_id: str
    summary: str
    quiz_topic: str
    quiz_result: dict
    next_topic: str
    next_reason: str
    forget: bool

def build_system_message(learner_id: str, summary: str, memory: str):
    learner = learners[learner_id]
    name = learner['name']
    level =learner['level']
    system_message = f"""
            You are QuizMate, a patient Python tutor.                                 
            You are helping {name}, a {level}.                                         
                                                                                     
            Rules:                                                                    
            1. Find the right note with search_notes, and use get_note for its example and common mistakes.                                          
            2. Answer Python questions only from the notes. If they don't cover something, say so.  
            3. Keep answers short. When you explain a topic, end with one small check question.   
            4. For questions about {name} (their goals, background, quiz results), use 'What you know about {name}' below. If it isn't there, say you don't know. Don't guess. 
            """
    if memory:
        system_message = f"{system_message}\n{memory}"
    if summary:
        system_message = f"{system_message}\nEarlier in this session:\n{summary}"
    

    return SystemMessage(content=system_message)

def forget(state: ChatState, store: BaseStore):
    facts = read_facts(store, state['learner_id'])
    scores = read_scores(store, state['learner_id'])
    topics = list(scores.keys())
    facts_space = facts_namespace(state['learner_id'])
    scores_space = scores_namespace(state['learner_id'])
    existing_facts = store.search(facts_space, limit = 50)
    existing_scores = store.search(scores_space, limit = 50)
    if not existing_facts and not existing_scores:
        reply = f"There's nothing saved about you."
        return {"messages": [AIMessage(content=reply)], "forget": False}
    payload = {"facts": facts, "topics": topics, "question": "Delete all of this? (yes/no)"}    
    answer = interrupt(payload)
    cleaned_answer = answer.strip().lower()
    if cleaned_answer == "yes":
        deleted_data = forget_learner(store, state['learner_id'])
        removals = [RemoveMessage(id=m.id) for m in state['messages']]
        reply =  f"Done. I've deleted {deleted_data['facts']} saved facts and your quiz history for {deleted_data['scores']} topics. I've also cleared this conversation."
        return {"messages": removals + [AIMessage(content = reply)], "summary":"", "forget": False}
    else:
        reply = "OK. I've kept everything."
    return {"messages": [AIMessage(content = reply)], "forget": False}


def remember(state: ChatState, store: BaseStore):
    last_message = state['messages'][-1].content
    learner_id = state['learner_id']
    learner = learners[learner_id]
    name = learner['name']
    existing = read_facts(store, learner_id)
    decision = extract_memories(name, existing, last_message)
    save_new_facts(store, learner_id, decision)
    return {}

def route_after_start(state: ChatState):
    if state.get("forget", False):
        return "forget"
    elif state.get('quiz_topic', ""):
        return "quiz"
    return "remember"

def save_scores(state: ChatState, store: BaseStore):
    learner_id = state['learner_id']
    quiz_result = state['quiz_result']
    update_score(store, learner_id, quiz_result)
    return {}

def suggest_next(state: ChatState, store: BaseStore):
    learner_id = state['learner_id']
    name = learners[learner_id]['name']
    level = learners[learner_id]['level']
    result = state['quiz_result']
    quiz_topic = result['topic']
    score = result['score']
    total = result['total']
    topic_list = ", ".join(TOPICS)
    missed = result.get("missed", [])
    if not missed:
        missed_lines = "none"
    else:
        lines = [m['question'] for m in missed]
        missed_lines = "; ".join(lines)
    scores = read_scores(store, learner_id)
    history = []
    for topic, record in scores.items():
        if topic == quiz_topic:
            continue
        history.append(f"- {topic}: last {record['last_score']} of {record['last_total']}, best {record['best']}")
    history_lines = "\n".join(history) if history else "none yet"
    prompt = f"""
You are choosing what {name}, a {level} Python learner, should study next.

Topics with notes (pick exactly one of these):
{topic_list}

The quiz {name} just finished:
- {quiz_topic}: {score} of {total}
- Missed: {missed_lines}

{name}'s earlier quizzes:
{history_lines}

How to choose:
- If {name} missed any questions, pick the same topic, {quiz_topic}, so they can review it.
- If {name} got every question right, pick a topic they have not passed yet. Prefer one that builds on {quiz_topic}.
- Never pick a topic that is not in the list above.

Write the reason as one short sentence to {name}, using "you".
- If they missed questions, say what those questions were about.
- If they got everything right, say why this topic is a good next step.
- Do not mention scores or numbers.
"""
    output = structured_model.invoke(prompt)
    return {"next_topic": output.topic, "next_reason": output.reason}

def report_quiz(state: ChatState):
    learner = learners[state['learner_id']]
    name = learner['name']
    result = state['quiz_result']
    score_line = f"{name}, you scored {result['score']} of {result['total']} on {result['topic']}."
    if result['missed']:
        lines = []
        for question in result['missed']:
            lines.append(f"You missed: {question['question']}\nThe answer was {question['correct_text']}: {question['explanation']}")
        feedback = "\n".join(lines)
    else:
        feedback = "Well done!"
    message = f"{score_line}\n\n{feedback}"
    if state.get("next_reason", ""):
        study_next_lines = f"Study next: {state.get('next_topic', '')}. {state.get('next_reason','')}"
        message = f"{message}\n\n{study_next_lines}"
    return {"messages":[AIMessage(content=message)], "quiz_topic": ""}

def chat_node(state: ChatState, store: BaseStore):
    facts = read_facts(store, state['learner_id']) 
    scores = read_scores(store, state["learner_id"])
    learner = learners[state['learner_id']]
    name = learner['name']
    memory = format_memory(name, facts, scores)
    messages = [build_system_message(state['learner_id'], state.get("summary", ""), memory)] + state['messages']
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

def route_after_chat(state: ChatState) -> Literal['tools', "summarise", "__end__"]:
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return "tools"
    elif len(state['messages']) > 8:
        return "summarise"
    else:
        return END

def messages_to_remove(messages):
    last_human_index = 0
    for i, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            last_human_index = i 
    messages = messages[:last_human_index]
    return messages     

def summarise(state: ChatState):
    if state.get('summary',""):
        instruction = f"Here is the summary so far:\n{state['summary']}.\nExtend it with the conversation above. Keep any facts the learner shared about themselves, their goals and what they found hard, and add what was covered since. Write the whole updated summary in a few sentences."
    else:
        instruction = f"Summarise the conversation so far in a few sentences. Keep any facts the learner shared about themselves, their goals, and what they found hard."
    model_input = state['messages'] + [HumanMessage(content = instruction)]
    summary = model.invoke(model_input).content
    old_messages = messages_to_remove(state['messages'])
    messages = [RemoveMessage(id = m.id) for m in old_messages]
    return {"messages": messages, "summary": summary}

graph = StateGraph(ChatState)

graph.add_node("chat", chat_node)
graph.add_node("forget", forget)
graph.add_node("tools", tool_node)
graph.add_node("summarise", summarise)
graph.add_node("suggest_next", suggest_next)
graph.add_node("quiz", quiz_subgraph)
graph.add_node("report_quiz", report_quiz)
graph.add_node("remember", remember)
graph.add_node("save_scores", save_scores)


graph.add_conditional_edges(START, route_after_start, {
    "forget": "forget",
    "quiz": "quiz",
    "remember": "remember"
})

graph.add_conditional_edges("chat", route_after_chat, {
    "tools": "tools",
    "summarise": "summarise",
    END: END
})
graph.add_edge("remember", "chat")
graph.add_edge("tools", "chat")
graph.add_edge("quiz", "save_scores")
graph.add_edge("quiz", "suggest_next")
graph.add_edge("suggest_next", "report_quiz")
graph.add_edge("save_scores", "report_quiz")
graph.add_edge("forget", END)
graph.add_edge("report_quiz", END)
graph.add_edge("summarise", END)

workflow = graph.compile(checkpointer=checkpointer, store=store)