from uuid import uuid4
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

class MemoryItem(BaseModel):
    text: str = Field(..., description="One short sentence about the learner, written in the third person with their name, e.g. 'Riya is preparing for a job interview next month.'")
    is_new: bool = Field(... , description="True only if this adds information not already in the existing memories. False if it means the same as something already stored")

class MemoryDecision(BaseModel):
    should_write: bool = Field(... , description="Whether the message contains anything worth remembering long term")
    memories: list[MemoryItem] = Field(default=[], description="The facts found")

model = ChatOpenAI(model = "gpt-4o-mini", temperature=0)
structured_model = model.with_structured_output(MemoryDecision)

def facts_namespace(learner_id: str):
    return ("learner", learner_id, "facts")

def scores_namespace(learner_id: str):
    return ("learner", learner_id,"scores")

def read_facts(store, learner_id):
    facts = []
    namespace = facts_namespace(learner_id)
    results = store.search(namespace, limit = 20)
    for item in results:
        facts.append(item.value['data'])
    return facts

def extract_memories(name, existing, message, config = None):
    if existing:
        facts = ["- " + line for line in existing]
        existing_memories = "\n".join(facts)
    else:
        existing_memories = "none yet"
    extraction_prompt = f"""You keep a short list of lasting facts about one learner, {name}, who is learning Python with a tutor.

Facts already stored about {name}:
{existing_memories}

Read only {name}'s latest message below, and decide what is worth remembering for future sessions.

Store a fact only if it is:
- about {name}, not about Python;
- stable: likely to still be true in a week (goals, deadlines, background, how they like to learn, what they find hard).

Do not store:
- Questions {name} asks, or the topic they are asking about right now. But if {name} says they find a topic hard or confusing, that is a difficulty: store it.
- anything about quiz scores or results (scores are recorded separately);
- greetings, thanks or small talk;
- anything {name} did not actually say. Do not guess.

For each fact:
- write one short sentence in the third person, using the name, e.g. "{name} is preparing for a job interview next month.";
- set is_new to false if it means the same as a fact already stored, even if worded differently; otherwise true.
- use the learner's name instead of he, she or they, e.g. "...so tables make sense to {name}."

If there is nothing worth storing, set should_write to false and return no memories.

{name}'s latest message:
{message}
"""
    result = structured_model.invoke(extraction_prompt, config=config)
    return result

def save_new_facts(store, learner_id, decision):
    saved = []
    if decision.should_write:
        namespace = facts_namespace(learner_id)
        for memory in decision.memories:
            if memory.is_new:
                random_key = uuid4().hex
                store.put(namespace, random_key, {"data":memory.text})
                saved.append(memory.text)
    return saved

def read_scores(store, learner_id):
    scores = {}
    namespace = scores_namespace(learner_id)
    items = store.search(namespace, limit = 20)
    for item in items:
        scores[item.key] = item.value
    return scores

def update_score(store, learner_id, quiz_result):
    namespace = scores_namespace(learner_id)
    quiz_topic = quiz_result['topic']
    item = store.get(namespace, quiz_topic)
    if not item:
        old_attempts = 0
        old_best = 0
    else:
        old_record = item.value
        old_attempts = old_record['attempts']
        old_best = old_record['best']
    topic = quiz_topic
    attempts = old_attempts + 1
    last_score = quiz_result['score']
    last_total = quiz_result['total']
    best = max(quiz_result['score'], old_best)
    last_missed = [m['question'] for m in quiz_result['missed']]
    record = {"topic": topic, "attempts": attempts,"last_score": last_score, "last_total": last_total, "best": best, "last_missed": last_missed}
    store.put(namespace, quiz_topic, record)
    return record

def format_memory(name, facts, scores):
    if not facts and not scores:
        return ""
    sections = []
    if facts:
        sections.append(f"What you know about {name}:")
        for line in facts:
            sections.append(f"- {line}")
    if scores:
        sections.append(f"\nQuiz history:")
        for topic in scores:
            data = scores[topic]
            if data['last_missed']:
                missed = "; ".join(data['last_missed'])
                sections.append(f"- {topic}: {data['last_score']} of {data['last_total']} last time (best {data['best']}, {data['attempts']} attempts). Missed: {missed}")
            else:
                sections.append(f"- {topic}: {data['last_score']} of {data['last_total']} last time (best {data['best']}, {data['attempts']} attempts).")
    return "\n".join(sections)

if __name__ == "__main__":
    import sys
    from langgraph.store.memory import InMemoryStore

    print("--- scores, by code (free) ---")
    test_store = InMemoryStore()
    first = {"topic": "loops", "score": 2, "total": 3,
             "missed": [{"question": "What does a for loop do?", "correct_text": "goes through each item"}]}
    second = {"topic": "loops", "score": 1, "total": 3,
              "missed": [{"question": "What does range(1, 5) give?"}, {"question": "When does a while loop stop?"}]}
    update_score(test_store, "riya", first)
    record = update_score(test_store, "riya", second)
    print(record)
    print("ok  " if record["attempts"] == 2 else "FAIL", "attempts counted")
    print("ok  " if record["last_score"] == 1 and record["best"] == 2 else "FAIL", "last is 1, best stays 2")
    print("ok  " if len(record["last_missed"]) == 2 else "FAIL", "last_missed is this quiz's misses")
    print("ok  " if list(read_scores(test_store, "riya")) == ["loops"] else "FAIL", "one record per topic")

    print("\n--- facts, from a decision (free: a made-up decision, no model) ---")
    decision = MemoryDecision(should_write=True, memories=[
        MemoryItem(text="Riya is preparing for a job interview next month.", is_new=True),
        MemoryItem(text="Riya is getting ready for an interview.", is_new=False)])
    saved = save_new_facts(test_store, "riya", decision)
    print("ok  " if saved == ["Riya is preparing for a job interview next month."] else "FAIL", "only the new one saved:", saved)
    print("ok  " if len(read_facts(test_store, "riya")) == 1 else "FAIL", "one fact stored")
    nothing = save_new_facts(test_store, "riya", MemoryDecision(should_write=False, memories=[]))
    print("ok  " if nothing == [] else "FAIL", "should_write false saves nothing")
    print("ok  " if read_facts(test_store, "sam") == [] and read_scores(test_store, "sam") == {} else "FAIL",
          "another learner has nothing")

    print("\n--- format_memory (free) ---")
    text = format_memory("Riya", read_facts(test_store, "riya"), read_scores(test_store, "riya"))
    print(text)
    print("ok  " if "What you know about Riya" in text and "job interview" in text else "FAIL", "facts section")
    print("ok  " if "Quiz history" in text and "loops" in text and "1 of 3" in text else "FAIL", "scores section")
    print("ok  " if format_memory("Sam", [], {}) == "" else "FAIL", "empty when there is nothing")
    perfect = {"dictionaries": {"topic": "dictionaries", "attempts": 1, "last_score": 3, "last_total": 3,
                                "best": 3, "last_missed": []}}
    print("ok  " if "Missed" not in format_memory("Sam", [], perfect) else "FAIL", "no 'Missed:' after a perfect quiz")

    if len(sys.argv) > 1 and sys.argv[1] == "run":
        print("\n--- extract_memories on 5 messages (5 model calls) ---")
        existing = ["Riya is preparing for a job interview next month."]
        for message, expect in [
            ("I'm getting ready for an interview soon.", "nothing new: a duplicate"),
            ("What is a dictionary?", "nothing: a question"),
            ("I learn best with short code examples.", "a new preference"),
            ("I got 3 out of 3 on loops, update my score.", "nothing: scores aren't facts"),
            ("I used to work in Excel a lot, so tables make sense to me.", "a new background fact"),
        ]:
            d = extract_memories("Riya", existing, message)
            new = [m.text for m in d.memories if m.is_new] if d.should_write else []
            print(f"\n{message}\n   expect: {expect}\n   got: should_write={d.should_write}, new={new}")