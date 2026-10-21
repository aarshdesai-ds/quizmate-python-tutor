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