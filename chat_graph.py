from tools import tools
from storage import checkpointer
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, RemoveMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from quiz_graph import quiz_subgraph
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
import json
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

def build_system_message(learner_id: str, summary: str):
    learner = learners[learner_id]
    name = learner['name']
    level =learner['level']
    system_message = f"""
            You are QuizMate, a patient Python tutor.                                 
            You are helping {name}, a {level}.                                         
                                                                                     
            Rules:                                                                    
            1. Find the right note with search_notes, and use get_note for its example and common mistakes.                                          
            2. Answer only from the notes. If they don't cover something, say so.  
            3. Keep answers short. When you explain a topic, end with one small check question.   
            """
    if summary:
        system_message = f"{system_message}\nEarlier in this session:\n{summary}"
    return SystemMessage(content=system_message)

def route_after_start(state: ChatState):
    if state.get('quiz_topic', ""):
        return "quiz"
    return "chat"

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

def chat_node(state: ChatState):
    messages = [build_system_message(state['learner_id'], state.get("summary", ""))] + state['messages']
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

graph.add_conditional_edges(START, route_after_start, {
    "quiz": "quiz",
    "chat": "chat"
})

graph.add_conditional_edges("chat", route_after_chat, {
    "tools": "tools",
    "summarise": "summarise",
    END: END
})
graph.add_edge("tools", "chat")
graph.add_edge("quiz", "report_quiz")
graph.add_edge("report_quiz", END)
graph.add_edge("summarise", END)

workflow = graph.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    import sys
    from uuid import uuid4
    from langgraph.types import Command
    from storage import config_for

    print("--- nodes (free) ---")
    print(sorted(workflow.get_graph().nodes))

    print("\n--- system message (free) ---")
    plain = build_system_message("riya", "").content
    with_summary = build_system_message("riya", "Riya is preparing for a job interview.").content
    print("ok  " if "Riya" in plain and "beginner" in plain else "FAIL", "name and level")
    print("ok  " if "Earlier in this session" not in plain else "FAIL", "no summary section when empty")
    print("ok  " if "Earlier in this session" in with_summary and "job interview" in with_summary else "FAIL",
          "summary section when present")

    print("\n--- messages_to_remove (free) ---")
    h1 = HumanMessage(content="hi, I'm preparing for an interview", id="h1")
    a1 = AIMessage(content="Great, what topic?", id="a1")
    h2 = HumanMessage(content="what is a set?", id="h2")
    call = AIMessage(content="", id="a2", tool_calls=[{"name": "search_notes", "args": {"query": "set"}, "id": "c1"}])
    tool = ToolMessage(content="sets | Sets | intermediate ...", tool_call_id="c1", id="t1")
    a3 = AIMessage(content="A set is an unordered collection...", id="a3")
    removed = [m.id for m in messages_to_remove([h1, a1, h2, call, tool, a3])]
    print("ok  " if removed == ["h1", "a1"] else "FAIL", "keeps the latest user message onwards; removed:", removed)
    print("ok  " if messages_to_remove([h1, a1]) == [] else "FAIL", "one turn: nothing removed")
    print("ok  " if messages_to_remove([a1, a3]) == [] else "FAIL", "no user message: nothing removed, no crash")

    print("\n--- route_after_chat (free) ---")
    long = [HumanMessage(content=f"q{i}", id=f"q{i}") for i in range(9)] + [AIMessage(content="answer", id="z")]
    for name, state, want in [
        ("tool request",    {"messages": [h2, call], "learner_id": "riya", "summary": ""}, "tools"),
        ("short, answered", {"messages": [h1, a1],   "learner_id": "riya", "summary": ""}, END),
        ("long, answered",  {"messages": long,       "learner_id": "riya", "summary": ""}, "summarise"),
    ]:
        got = route_after_chat(state)
        print(f"{'ok  ' if got == want else 'FAIL'} {name:<16} -> {got}")

    print("\n--- route_after_start (free) ---")
    for name, state, want in [
        ("new session, no quiz key", {"messages": [h1], "learner_id": "riya"}, "chat"),
        ("quiz requested",           {"messages": [], "learner_id": "riya", "quiz_topic": "loops"}, "quiz"),
        ("after a quiz (cleared)",   {"messages": [h1], "learner_id": "riya", "quiz_topic": ""}, "chat"),
    ]:
        got = route_after_start(state)
        print(f"{'ok  ' if got == want else 'FAIL'} {name:<26} -> {got}")

    print("\n--- report_quiz (free) ---")
    fake_result = {"topic": "loops", "score": 2, "total": 3, "missed": [
        {"question": "What does range(1, 5) give?", "given": "A", "correct": "B",
         "correct_text": "1, 2, 3, 4", "right": False, "explanation": "range stops before the stop value."}]}
    out = report_quiz({"learner_id": "riya", "quiz_result": fake_result, "quiz_topic": "loops"})
    msg = out["messages"][0]
    print(msg.content)
    print("ok  " if isinstance(msg, AIMessage) else "FAIL", "the report is a tutor (AI) message")
    print("ok  " if "2 of 3" in msg.content and "1, 2, 3, 4" in msg.content else "FAIL", "score and answer in the text")
    print("ok  " if out["quiz_topic"] == "" else "FAIL", "quiz_topic cleared")
    perfect = report_quiz({"learner_id": "riya", "quiz_topic": "loops",
                           "quiz_result": {"topic": "loops", "score": 3, "total": 3, "missed": []}})
    print("ok  " if "Well done" in perfect["messages"][0].content else "FAIL", "3 of 3 says well done")

    if len(sys.argv) > 1 and sys.argv[1] == "quiz":
        print("\n--- two quizzes and a chat turn in ONE session (2 quiz model calls + 1 chat) ---")
        config = {**config_for("riya", f"test-{uuid4().hex[:6]}"),
                  "run_name": "quizmate-quiz-test", "tags": ["quizmate", "test", "quiz"]}
        for topic in ["loops", "dictionaries"]:
            state = workflow.invoke({"messages": [], "learner_id": "riya", "quiz_topic": topic}, config=config)
            pauses = 0
            while state.get("__interrupt__"):
                pauses += 1
                state = workflow.invoke(Command(resume="A"), config=config)
            report = state["messages"][-1]
            total = state["quiz_result"]["total"]
            print(f"\n{topic}: {pauses} questions | total {total} | report: {report.content[:80]!r}")
            print("ok  " if pauses == 3 and total == 3 else "FAIL",
                  f"{topic}: exactly 3 questions, and the total is 3 (not carried over from an earlier quiz)")
            print("ok  " if isinstance(report, AIMessage) else "FAIL", f"{topic}: report is a tutor message")

        after = workflow.invoke({"messages": [HumanMessage(content="What did I just get quizzed on?")],
                                 "learner_id": "riya"}, config=config)
        print("\nchat after the quizzes:", after["messages"][-1].content[:150])
        print("ok  " if after.get("quiz_topic", "") == "" else "FAIL", "quiz_topic is clear, so chat resumed")

    if len(sys.argv) > 1 and sys.argv[1] == "run":
        print("\n--- one session, 6 turns, as riya (uses the model) ---")
        config = {**config_for("riya", f"test-{uuid4().hex[:6]}"),
                  "run_name": "quizmate-graph-test", "tags": ["quizmate", "test"]}
        turns = ["Hi! I'm preparing for a job interview next month.",
                 "What is a dictionary?",
                 "How do I avoid a KeyError?",
                 "What's the difference between append and extend?",
                 "What does range(1, 5) give?",
                 "Remind me, what am I preparing for?"]
        for number, text in enumerate(turns, start=1):
            result = workflow.invoke({"messages": [HumanMessage(content=text)], "learner_id": "riya"}, config=config)
            print(f"\n[{number}] you> {text}\nbot> {result['messages'][-1].content[:200]}")
            print(f"   [messages kept: {len(result['messages'])} | summary: {result.get('summary', '')[:90]!r}]")
        final = workflow.get_state(config).values
        print("\nfinal summary:", final.get("summary", ""))
        print("ok  " if "interview" in final.get("summary", "").lower() else "FAIL", "the goal survived in the summary")
        print("ok  " if "interview" in final["messages"][-1].content.lower() else "FAIL",
              "turn 6 answered from the summary")