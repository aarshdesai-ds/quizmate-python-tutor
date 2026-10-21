from tools import tools
from storage import checkpointer, store
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, RemoveMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from quiz_graph import quiz_subgraph
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
import json
from memory import read_facts, extract_memories, save_new_facts, read_scores, update_score, format_memory
from langgraph.store.base import BaseStore
from typing import TypedDict, Annotated, Literal
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)
model_with_tools = model.bind_tools(tools)

with open("data/learners.json", encoding="utf-8") as f:
    learners = json.loads(f.read())

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    learner_id: str
    summary: str
    quiz_topic: str
    quiz_result: dict

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
    if state.get('quiz_topic', ""):
        return "quiz"
    return "remember"

def save_scores(state: ChatState, store: BaseStore):
    learner_id = state['learner_id']
    quiz_result = state['quiz_result']
    update_score(store, learner_id, quiz_result)
    return {}


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
graph.add_node("tools", tool_node)
graph.add_node("summarise", summarise)
graph.add_node("quiz", quiz_subgraph)
graph.add_node("report_quiz", report_quiz)
graph.add_node("remember", remember)
graph.add_node("save_scores", save_scores)


graph.add_conditional_edges(START, route_after_start, {
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
graph.add_edge("save_scores", "report_quiz")
graph.add_edge("report_quiz", END)
graph.add_edge("summarise", END)

workflow = graph.compile(checkpointer=checkpointer, store=store)


if __name__ == "__main__":
    import sys
    from uuid import uuid4
    from langgraph.types import Command
    from langgraph.store.memory import InMemoryStore
    from storage import config_for
    from memory import facts_namespace, scores_namespace

    print("--- nodes (free) ---")
    print(sorted(workflow.get_graph().nodes))
    edges = {(e.source, e.target) for e in workflow.get_graph().edges}
    print("ok  " if ("remember", "chat") in edges and ("quiz", "save_scores") in edges
          and ("save_scores", "report_quiz") in edges else "FAIL", "new arrows in place")
    print("ok  " if ("tools", "chat") in edges and ("tools", "remember") not in edges else "FAIL",
          "tools go back to chat, not remember")

    print("\n--- system message with memory (free) ---")
    memory_text = "What you know about Riya:\n- Riya is preparing for a job interview next month."
    with_memory = build_system_message("riya", "Earlier we discussed sets.", memory_text).content
    print("ok  " if "job interview" in with_memory else "FAIL", "memory section included")
    print("ok  " if with_memory.index("job interview") < with_memory.index("Earlier in this session") else "FAIL",
          "memory comes before the session summary")
    print("ok  " if "don't know" in with_memory.lower() or "do not know" in with_memory.lower() else "FAIL",
          "the 'say you don't know' rule is there")
    print("ok  " if "What you know about Riya:" not in build_system_message("riya", "", "").content else "FAIL",
          "no memory section when there is none")

    print("\n--- save_scores, with an in-memory store (free) ---")
    test_store = InMemoryStore()
    result = {"topic": "loops", "score": 2, "total": 3, "missed": [{"question": "What does a for loop do?"}]}
    save_scores({"learner_id": "riya", "quiz_result": result}, test_store)
    saved = test_store.get(("learner", "riya", "scores"), "loops")
    print("ok  " if saved and saved.value["last_score"] == 2 else "FAIL", "score written by code:", saved.value if saved else None)

    if len(sys.argv) > 1 and sys.argv[1] == "run":
        print("\n--- two sessions as sam: say a goal and take a quiz, then come back (uses the model) ---")
        session_a = {**config_for("sam", f"test-{uuid4().hex[:6]}"), "tags": ["quizmate", "test", "memory"]}
        workflow.invoke({"messages": [HumanMessage(content="Hi! I'm learning Python to automate reports at work.")],
                         "learner_id": "sam"}, config=session_a)
        state = workflow.invoke({"messages": [], "learner_id": "sam", "quiz_topic": "loops"}, config=session_a)
        while state.get("__interrupt__"):
            state = workflow.invoke(Command(resume="A"), config=session_a)
        print("session A quiz:", state["messages"][-1].content.splitlines()[0])

        session_b = {**config_for("sam", f"test-{uuid4().hex[:6]}"), "tags": ["quizmate", "test", "memory"]}
        for question in ["Why am I learning Python?", "How did I do on my last quiz?", "What's my favourite colour?"]:
            reply = workflow.invoke({"messages": [HumanMessage(content=question)], "learner_id": "sam"},
                                    config=session_b)["messages"][-1].content
            print(f"\nsession B> {question}\nbot> {reply[:200]}")

        print("\nstored facts:", read_facts(store, "sam"))
        print("stored scores:", read_scores(store, "sam"))

        for item in store.search(facts_namespace("sam"), limit=50):
            store.delete(facts_namespace("sam"), item.key)
        for item in store.search(scores_namespace("sam"), limit=50):
            store.delete(scores_namespace("sam"), item.key)
        print("\ncleaned up: sam's test facts and scores removed")

    if len(sys.argv) > 1 and sys.argv[1] == "probe":
        print("\n--- probe: facts + quiz history, 'why' x5 and 'colour' x2 (7 model calls, no store) ---")
        scores = {"loops": {"topic": "loops", "attempts": 1, "last_score": 2, "last_total": 3, "best": 2,
                            "last_missed": ["What does `range(start, stop)` produce?"]}}
        memory_text = format_memory("Sam", ["Sam is learning Python to automate reports at work."], scores)
        with_scores = build_system_message("sam", "", memory_text).content
        why = HumanMessage(content="Why am I learning Python?")
        colour = HumanMessage(content="What's my favourite colour?")
        variants = [("why", why)] * 5 + [("colour", colour)] * 2
        tally = {"why": 0, "colour": 0}
        for label, question in variants:
            reply = model_with_tools.invoke([SystemMessage(content=with_scores), question],
                                            config={"tags": ["quizmate", "test", "probe"]}).content
            if label == "why":
                passed = "automate reports" in reply and not reply.lower().startswith("i don't know")
            else:
                passed = "don't know" in reply.lower()
            tally[label] += passed
            print(f"{'ok  ' if passed else 'FAIL'} {label:7} {reply[:120]}")
        print(f"\nwhy: {tally['why']}/5 answered from memory   colour: {tally['colour']}/2 said don't know")