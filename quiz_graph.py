import operator
import random
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from tools import get_note

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

LETTERS = ["A", "B", "C", "D"]
shuffle_rng = random.Random()


class Question(BaseModel):
    question: str = Field(..., description="The question, written so it can be answered using only the note. Ask about one idea from the note, in one sentence.")
    options: list[str] = Field(..., description="Exactly 4 answer choices, as plain text with no letters or numbers in front. Exactly one is correct according to the note; the other three are plausible but clearly wrong according to the note.", min_length=4, max_length=4)
    answer: Literal["A", "B", "C", "D"] = Field(..., description="The letter of the correct option: 'A' for the first option, 'B' for the second, 'C' for the third, 'D' for the fourth.")
    explanation: str = Field(..., description="One sentence saying why the correct option is right, taken from or closely following the note's wording.")


class Quiz(BaseModel):
    questions: list[Question] = Field(..., description="Exactly 3 multiple-choice questions about the note, each on a different idea.", min_length=3, max_length=3)


structured_model = model.with_structured_output(Quiz)


class QuizState(TypedDict):
    quiz_topic: str
    quiz_result: dict
    note: str
    questions: list[dict]
    current: int
    last_answer: str
    answers: Annotated[list[dict], operator.add]


def clean_answer(answer: str) -> str:
    cleaned_answer = answer.strip()
    cleaned_answer = cleaned_answer.rstrip(")").rstrip(".")
    cleaned_answer = cleaned_answer.upper()
    if cleaned_answer in LETTERS:
        return cleaned_answer
    return ""


def shuffle_question(question: dict, rng: random.Random) -> dict:
    options = question["options"]
    correct_text = options[LETTERS.index(question["answer"])]

    shuffled = list(options)
    rng.shuffle(shuffled)

    new_letter = LETTERS[shuffled.index(correct_text)]
    return {**question, "options": shuffled, "answer": new_letter}


def load_note(state: QuizState):
    topic = state["quiz_topic"]
    note = get_note.invoke({"topic": topic})
    return {"note": note}


def write_questions_raw(note: str) -> list[dict]:
    prompt = f"""Write 3 multiple-choice questions to test a learner on the Python note below.

Rules:
1. Use only the note. Every question must be answerable from the note, and every correct answer must be stated in it.
2. Each question tests a different idea from the note.
3. Give each question exactly 4 options, with exactly one correct. Make the other three plausible but clearly wrong according to the note.
4. Do not put letters or numbers in front of the options.
5. The explanation is one sentence, following the note's wording.

Note:
{note}
"""
    result = structured_model.invoke(prompt)
    return [question.model_dump() for question in result.questions]


def write_questions(state: QuizState):
    raw = write_questions_raw(state["note"])
    questions = [shuffle_question(question, shuffle_rng) for question in raw]
    return {"questions": questions, "current": 0}


def ask(state: QuizState):
    questions = state["questions"]
    current = state["current"]
    current_question = questions[current]
    current_options = [f"{letter}. {option}" for letter, option in zip(LETTERS, current_question["options"])]
    answer = interrupt({
        "number": current + 1,
        "total": len(questions),
        "question": current_question["question"],
        "options": current_options,
    })
    return {"last_answer": clean_answer(answer)}


def mark(state: QuizState):
    current = state["current"]
    current_question = state["questions"][current]
    last_answer = state["last_answer"]
    correct_letter = current_question["answer"]
    right = last_answer == correct_letter
    correct_text = current_question["options"][LETTERS.index(correct_letter)]
    record = {
        "question": current_question["question"],
        "given": last_answer,
        "correct": correct_letter,
        "correct_text": correct_text,
        "right": right,
        "explanation": current_question["explanation"],
    }
    return {"answers": [record], "current": current + 1}


def finish(state: QuizState):
    score = 0
    missed = []
    for answer in state["answers"]:
        if answer["right"]:
            score += 1
        else:
            missed.append(answer)
    quiz_result = {"topic": state["quiz_topic"], "score": score, "total": len(state["answers"]), "missed": missed}
    return {"quiz_result": quiz_result}


def route_after_mark(state: QuizState) -> Literal["ask", "finish"]:
    if state["current"] < len(state["questions"]):
        return "ask"
    return "finish"


graph = StateGraph(QuizState)

graph.add_node("load_note", load_note)
graph.add_node("write_questions", write_questions)
graph.add_node("ask", ask)
graph.add_node("mark", mark)
graph.add_node("finish", finish)

graph.add_edge(START, "load_note")
graph.add_edge("load_note", "write_questions")
graph.add_edge("write_questions", "ask")
graph.add_edge("ask", "mark")
graph.add_conditional_edges("mark", route_after_mark, {"ask": "ask", "finish": "finish"})
graph.add_edge("finish", END)

quiz_subgraph = graph.compile()


if __name__ == "__main__":
    import sys
    from langgraph.checkpoint.memory import InMemorySaver

    print("--- clean_answer (free) ---")
    for given, want in [("b", "B"), (" B ", "B"), ("B)", "B"), ("b.", "B"),
                        ("maybe c", ""), ("", ""), ("E", ""), ("AB", "")]:
        got = clean_answer(given)
        print(f"{'ok  ' if got == want else 'FAIL'} {given!r:<10} -> {got!r}")

    print("\n--- mark, router and finish, called directly (free) ---")
    FAKE = [
        {"question": "What does d.get('x', 0) return when 'x' is missing?",
         "options": ["None", "0", "KeyError", "'x'"], "answer": "B", "explanation": "get returns the default."},
        {"question": "Which of these can be a dictionary key?",
         "options": ["a tuple", "a list", "a set", "a dict"], "answer": "A", "explanation": "Keys must be immutable."},
        {"question": "What does .items() give?",
         "options": ["only keys", "only values", "key-value pairs", "the length"], "answer": "C", "explanation": "items gives pairs."},
    ]
    records = []
    for current, given in enumerate(["B", "C", ""]):          # right, wrong, unclear
        out = mark({"questions": FAKE, "current": current, "last_answer": given})
        records += out["answers"]
        rec = out["answers"][0]
        print(f"  q{current + 1}: given {given!r:<4} right={rec['right']!s:<5} "
              f"correct_text={rec['correct_text']!r}  next={out['current']}")
    print("ok  " if [r["right"] for r in records] == [True, False, False] else "FAIL", "marking: right, wrong, unclear")
    print("ok  " if records[1]["correct_text"] == "a tuple" else "FAIL", "correct_text follows the letter")
    print("ok  " if route_after_mark({"questions": FAKE, "current": 1}) == "ask" else "FAIL",
          "router: more questions -> ask")
    print("ok  " if route_after_mark({"questions": FAKE, "current": 3}) == "finish" else "FAIL",
          "router: after the 3rd -> finish")
    summary = finish({"quiz_topic": "dictionaries", "answers": records})["quiz_result"]
    print("ok  " if summary["score"] == 1 and summary["total"] == 3 and len(summary["missed"]) == 2 else "FAIL",
          f"finish: {summary['score']} of {summary['total']}, {len(summary['missed'])} missed")

    print("\n--- shuffle_question (free, seeded) ---")
    seeded = random.Random(42)
    letters = []
    broken = False
    for _ in range(200):
        for q in FAKE:
            shuffled = shuffle_question(q, seeded)
            letters.append(shuffled["answer"])
            kept_text = shuffled["options"][LETTERS.index(shuffled["answer"])] == q["options"][LETTERS.index(q["answer"])]
            if not kept_text or sorted(shuffled["options"]) != sorted(q["options"]):
                broken = True
    print("ok  " if not broken else "FAIL", "the correct text and the option set never change")
    counts = {l: letters.count(l) for l in LETTERS}
    print("ok  " if all(130 <= c <= 170 for c in counts.values()) else "FAIL",
          "600 shuffles spread the correct letter evenly:", counts)
    original = {**FAKE[0], "options": list(FAKE[0]["options"])}
    shuffle_question(FAKE[0], seeded)
    print("ok  " if FAKE[0] == original else "FAIL", "the original question is not changed")

    print("\n--- the subgraph compiles without its own checkpointer (free) ---")
    print("ok  " if quiz_subgraph.checkpointer is None else "FAIL", "quiz_subgraph has no checkpointer of its own")
    print(sorted(quiz_subgraph.get_graph().nodes))

    if len(sys.argv) > 1 and sys.argv[1] == "run":
        print("\n--- a real quiz on loops, through the subgraph (one model call) ---")
        test_quiz = graph.compile(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "quiz-run-test"}}
        state = test_quiz.invoke({"quiz_topic": "loops"}, config=config)
        pauses = 0
        while state.get("__interrupt__"):
            payload = state["__interrupt__"][0].value
            pauses += 1
            print(f"\n{payload['number']} of {payload['total']}: {payload['question']}")
            for option in payload["options"]:
                print(f"   {option}")
            print("   answer given: A")
            state = test_quiz.invoke(Command(resume="A"), config=config)
        quiz = state["quiz_result"]
        saved = test_quiz.get_state(config).values["answers"]
        print(f"\nscore {quiz['score']} of {quiz['total']}")
        print("ok  " if pauses == 3 else "FAIL", "exactly 3 questions asked")
        print("ok  " if len(saved) == 3 else "FAIL", "the reducer kept all 3 records")
        print("ok  " if quiz["score"] == sum(r["correct"] == "A" for r in saved) else "FAIL",
              "the score matches the records (always answering A)")
        print("\ncorrect answers, to check against the loops note:")
        for r in saved:
            print(f"   {r['question'][:70]}\n      → {r['correct']} = {r['correct_text']!r}   ({r['explanation']})")

    if len(sys.argv) > 1 and sys.argv[1] == "letters":
        print("\n--- correct-letter positions, before and after the shuffle (12 model calls) ---")
        seeded = random.Random(7)
        topics = ["variables", "strings", "lists", "tuples", "dictionaries", "sets",
                  "conditionals", "loops", "functions", "comprehensions", "exceptions", "files"]
        before, after = [], []
        for topic in topics:
            note = load_note({"quiz_topic": topic})["note"]
            raw = write_questions_raw(note)                      # the model's own order, not shuffled
            before += [q["answer"] for q in raw]
            after += [shuffle_question(q, seeded)["answer"] for q in raw]
            print(f"  {topic:<15} model: {[q['answer'] for q in raw]}")
        print("\nas the model wrote them:", {l: before.count(l) for l in LETTERS})
        print("after the shuffle:      ", {l: after.count(l) for l in LETTERS})