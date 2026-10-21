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

def forget_learner(store, learner_id):
    facts_deleted = 0
    scores_deleted = 0
    facts_space = facts_namespace(learner_id)
    scores_space = scores_namespace(learner_id)
    existing_facts = store.search(facts_space, limit=50)
    existing_scores = store.search(scores_space, limit = 50)
    for item in existing_facts:
        store.delete(facts_space, key = item.key)
        facts_deleted += 1
    for item in existing_scores:
        store.delete(scores_space, key = item.key)
        scores_deleted += 1
    return {"facts": facts_deleted, "scores": scores_deleted}