# QuizMate: A Python Tutor That Remembers You

A LangGraph chatbot that teaches beginner Python from a small set of notes. It answers questions from the notes, runs short quizzes that pause for each answer, and remembers each learner across sessions: their goals, their background, what they find hard and how they did on past quizzes. A learner can ask it to forget them, and nothing is deleted until they say yes.

The project was built in four steps, each one adding a group of LangGraph ideas on top of the last. Every step was checked with tests and real runs, and the replies were read as well as counted. The sections below report what each step found, including the fixes that only half worked.

*This is an independent learning project. All learners, notes and evaluation data are synthetic and were written for this project.*

---

## Table of Contents

- [What It Does](#what-it-does)
- [How It Works](#how-it-works)
- [The Notes](#the-notes)
- [Step 1: The Tutor](#step-1-the-tutor)
- [Step 2: Quizzes](#step-2-quizzes)
- [Step 3: Long-Term Memory](#step-3-long-term-memory)
- [Step 4: Forgetting and the Parallel Step](#step-4-forgetting-and-the-parallel-step)
- [What the Project Taught](#what-the-project-taught)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Known Limitations](#known-limitations)
- [Ideas for Extension](#ideas-for-extension)

---

## What It Does

```
python main.py sam
Quizmate for Sam (beginner). Session 90a42e.

you> I'm a nurse and I want to automate my shift reports.
bot> That's great! Automating shift reports can save you time and improve efficiency.
     What specific aspects of the automation are you looking to learn about in Python?

you> forget me
Saved facts:
- Sam is a nurse.
- Sam wants to automate their shift reports.
Delete all of this? (yes/no)
answer> yes
bot> Done. I've deleted 2 saved facts and your quiz history for 0 topics.
     I've also cleared this conversation.

you> What do I do for work?
bot> I don't know what you do for work.
```

After a quiz, the report ends with a suggestion chosen while the score is being saved:

```
Riya, you scored 0 of 3 on loops.

You missed: What does `range(start, stop)` produce?
The answer was Numbers from start up to but not including stop: ...

Study next: loops. You missed questions about how `for` loops work, what
`range(start, stop)` produces, and the behavior of `while` loops.
```

There are three synthetic learners:

| Learner | Level |
|---|---|
| Riya | beginner |
| Omar | intermediate |
| Sam | beginner |

### Commands

| Type | Does |
|---|---|
| any message | Asks the tutor |
| `quiz <topic>` | A 3-question quiz on one of the 12 topics |
| `history` | This session's conversation, with the summary of earlier turns first |
| `summary` | The summary of earlier turns only |
| `forget me` | Shows what is saved about you, and deletes it after a yes |
| `exit` | Quits |

---

## How It Works

```
                                   START
                                     │ route_after_start
            forget is set            │ quiz topic set                 otherwise
     ┌───────────────────────────────┼──────────────────────────────────────┐
     ▼                               ▼                                      ▼
  forget                           quiz   (subgraph, pauses ×3)         remember
  (pauses)                           │                                      ▼
     │                 ┌─────────────┴─────────────┐                      chat ◄──────┐
     │                 ▼                           ▼                        │          │
     │            save_scores                suggest_next          route_after_chat    │
     │              (code)                  (1 model call)        ┌─────────┼──────┐   │
     │                 └─────────────┬─────────────┘              ▼         ▼      ▼   │
     │                               ▼                          tools   summarise END  │
     │                          report_quiz                       │         │          │
     ▼                          (runs once)                       │         ▼          │
    END                              ▼                            │        END         │
                                    END                           └────────────────────┘
```

| Piece | Role |
|---|---|
| `route_after_start` | Checks `forget` first, then `quiz_topic`, and sends everything else to `remember` |
| `remember` | Reads the learner's stored facts, asks the model whether the new message holds anything lasting, and saves only facts marked new |
| `chat` | Calls `gpt-4o-mini` (temperature 0) with the tools. The system message is built fresh on every call from the learner's profile, their stored facts and quiz history, and the session summary. It is never stored in the conversation |
| `tools` | LangGraph's `ToolNode` with `search_notes` and `get_note` |
| `summarise` | Runs when a turn ends with more than 8 messages. It extends the running summary and removes older messages with `RemoveMessage`, keeping everything from the latest user message onward so a tool call is never separated from its result |
| `quiz` | A compiled subgraph added as a node. It shares `quiz_topic` and `quiz_result` with the parent and keeps its questions and answers private |
| `save_scores` | Code only. Writes the quiz result to the learner's score record |
| `suggest_next` | A structured model call that picks one of the 12 topics to study next, with a one-sentence reason |
| `report_quiz` | Builds the quiz report in code, adding the suggestion when there is one |
| `forget` | Shows what is stored, pauses with `interrupt()`, and deletes only after an exact "yes" |
| Checkpointer | `SqliteSaver` on `quizmate.db`. Each run of `main.py` is a new session, with thread id `learner-session` |
| Store | `SqliteStore` on the same file, with namespaces `("learner", id, "facts")` and `("learner", id, "scores")` |
| LangSmith | Every run is traced with `run_name`, `tags` and `metadata` (learner id, session id, eval row id) |

### LangGraph ideas used

| Idea | Where |
|---|---|
| Conditional edges with path maps, including from `START` | `route_after_start`, `route_after_chat` |
| Tools and `ToolNode` | `search_notes`, `get_note` |
| Reducers | `add_messages` on the conversation, `operator.add` on quiz answers |
| Subgraph with shared and private state | The quiz |
| Human in the loop with `interrupt()` and `Command(resume=...)` | Quiz answers, forget approval |
| Persistence and threads | `SqliteSaver`, one thread per session |
| Short-term memory | `summarise` with `RemoveMessage` and a summary key |
| Long-term memory | `SqliteStore`, with `store` passed into nodes |
| Structured output | Quiz questions, memory extraction, next-topic choice |
| Fan-out and fan-in | `save_scores` and `suggest_next` in parallel, joined at `report_quiz` |

---

## The Notes

There are 12 synthetic notes, one Markdown file each, with front matter and three sections (Key idea, Example, Common mistakes):

```
topic: loops
title: Loops
level: beginner
```

| Level | Topics |
|---|---|
| beginner | variables, strings, lists, tuples, dictionaries, conditionals, loops, functions |
| intermediate | comprehensions, exceptions, files, sets |

`ingest.py` stores each file whole, as one document, in Chroma. A header line, "Title (topic, level)", goes at the top of its text. There is no splitter, because the notes are short. The ids are the topic names, so re-running the ingest replaces the 12 documents instead of adding 12 more.

| Tool | Does |
|---|---|
| `search_notes(query, topic)` | A similarity search, optionally filtered to one topic. It returns up to 3 results, each with its topic, title, level and Key idea |
| `get_note(topic)` | The full note for one topic, by id. The topic is checked against the 12 first |

---

## Step 1: The Tutor

The first version was the chat, the tools and the summary: a `chat` node with the two tools, `summarise` for long sessions, and the `history` and `summary` commands.

Three findings from testing:

| Finding | How it was found |
|---|---|
| The model said "the notes don't cover" the difference between `append` and `extend`, although `search_notes` had returned the note that does | Reading the tool result in `get_state_history` |
| In a new session, it guessed the learner's goal ("you're preparing to learn Python!") instead of saying it didn't know | Manual test |
| The summary keeps growing, and mixes facts about the learner with facts about the session | Reading summaries across several sessions |

The second and third findings were the reason for step 3. Facts about the learner belong in a store that outlives the session, not in a summary that ends with it.

---

## Step 2: Quizzes

The quiz is a subgraph with five nodes:

```
load_note ──► write_questions ──► ask ──► mark ──route──┬──► ask     (next question)
                                   ▲                     └──► finish
                                   └── pauses with interrupt()
```

- **`write_questions`** asks the model for exactly 3 questions with 4 options each, using structured output, then shuffles the options in code.
- **`ask`** pauses with the question and the lettered options.
- **`mark`** compares the answer with the correct letter in code. The model never marks.
- **`finish`** returns the score, the total and the missed questions to the parent.

The subgraph is compiled **without** its own checkpointer, because the parent's covers it. Its private answer list starts fresh each time the quiz is entered, and two quizzes in one session were tested to confirm it.

**The model's right answers were not random.** Over 36 generated questions, the correct option was:

| Position | A | B | C | D |
|---|---|---|---|---|
| As written by the model | 3 | 23 | 10 | 0 |
| After shuffling in code | 10 | 7 | 8 | 11 |

A learner who always answered B would have scored about 64%. The fix is a shuffle in code after the model writes the questions, not an instruction to the model.

---

## Step 3: Long-Term Memory

| Piece | Does |
|---|---|
| `remember` | Before every chat turn: extracts lasting facts from the user's latest message, compares them with the stored ones, and saves only those marked new |
| `save_scores` | After every quiz: updates attempts, last score, best score and last missed questions, in code |
| The system message | Gains a "What you know about {name}" section (facts and quiz history) and a rule: for questions about the learner, use that section, and say you don't know if it isn't there |

The extraction prompt stores goals, deadlines, background, learning preferences and difficulties. It skips questions, quiz scores, small talk and anything the learner didn't say.

### The memory evaluation

`eval/memory_messages.jsonl` has 20 labelled messages across 9 groups. Each row gives the learner, the facts already stored, the message, whether a write is expected, and a word the saved fact must or must not contain. `eval_memory.py` runs the extraction on each row with no store, and scores it in code.

| Group | Rows | Expected |
|---|---|---|
| goal, preference, background, difficulty | 10 | A new fact |
| duplicate | 2 | Nothing new: the same fact is already stored |
| question | 3 | Nothing |
| score claim | 2 | Nothing: scores come only from quizzes |
| small talk | 2 | Nothing |
| mixed | 1 | "I'm a nurse… how do dictionaries work?": keep the nurse fact, not the topic |

| Run | Change | Passed | Failures |
|---|---|---|---|
| 1 | — | 18 of 20 | D01 ("Loops always confuse me") saved nothing. R03 saved a fact |
| Rescore | R03 relabelled | 19 of 20 | D01 |
| 2 | One line in the extraction prompt | 20 of 20 | none |

**R03 was a labelling mistake, not a model mistake.** "As a JavaScript person, this syntax feels odd" was labelled a duplicate of "Omar already knows some JavaScript". The model saved "Omar finds the syntax of Python odd compared to JavaScript", which is new information. The label was changed, and the saved results were re-scored at no cost.

**D01 was a clash between two rules.** "Store what they find hard" and "don't store the topic they are asking about" both matched "Loops always confuse me", and the second won. The trace showed `should_write: false`. The fix was one sentence: a topic they say they find hard is a difficulty, and is stored. The question rows and the mixed row, the ones the fix put at risk, still passed.

### Memory across sessions

A test as Sam: in session A, he says he is learning Python to automate reports at work, and takes a loops quiz. Session B starts fresh.

| Session B question | First run | Second run (rules 2 and 4 narrowed) |
|---|---|---|
| How did I do on my last quiz? | Correct, from the store | Correct |
| What's my favourite colour? | "I don't know" | "I don't know" |
| Why am I learning Python? | **"I don't know"** | **"I don't know"** |

The trace showed the fact **was** in the system message, and the model said "I don't know" without calling a tool. A probe sent the same question straight to the model with four variants of the system message, one change each: as is, without indentation, with the memory first, and with the question reworded. The first time, all four answered correctly. Repeated three times each, both the plain version and the version with quiz history failed some of the time. It was flaky, not caused by one part of the prompt.

The hedged replies showed why: "I don't know why you're learning Python, but I do know that you're learning it to automate reports at work." The model treated "why" as asking for more than the stored fact, and the "say you don't know" rule won. Rule 4 was reworded so that it answers from what is stored, even when that only partly answers the question:

| Probe, 5 runs of "why" and 2 of "colour" | Before | After |
|---|---|---|
| Clean answer from memory | 0 of 3 | 1 of 5 |
| Hedged, with the right fact | 1 of 3 | 4 of 5 |
| Only "I don't know" | 2 of 3 | 0 of 5 |
| Colour question still "I don't know" | — | 2 of 2 |

In the probe, the learner now always gets the fact. The wording is still clumsy, and that is listed as a limitation rather than tuned further. The full two-session test was not re-run after this last rewrite, so the probe is the only measure of it.

### Scores cannot be talked up

Sam was given a stored loops score of 1 of 3, then said: "I actually got 3 out of 3 on loops last time, please update my score."

| Check | Result |
|---|---|
| The stored score | Unchanged at 1 of 3 |
| A fact saved from the claim | None |
| "How did I do on loops last time?" | "1 out of 3", with the two missed questions |

Nothing on the chat path can write a score. The first reply to the claim, though, was "I don't know about your quiz results", which is wrong: it does know, and could have said so.

---

## Step 4: Forgetting and the Parallel Step

### Forget me

```
forget ──► read facts and scores
             ├── nothing stored ──► "There's nothing saved about you."   (no pause)
             └── pause: the facts, the quiz topics, "Delete all of this? (yes/no)"
                   ├── "yes" ──► delete facts and scores, clear this session ──► "Done…"
                   └── anything else ──► "OK. I've kept everything."
```

The delete comes **after** the pause, because a node runs again from the top when it resumes. Only an exact "yes", after stripping and lowercasing, deletes anything: "y" and "sure" keep everything. The router checks `forget` before anything else, so "forget me" never reaches `remember`.

**The first version left the session behind.** After a yes, the store was empty, but "What do I do for work?" got "they are a nurse…", read straight from the conversation in the same session. The forget node now also removes every message in the thread and empties the summary. After that change, the same question got "I don't know what you do for work."

| Free test, on a copy of the graph with an in-memory store | Result |
|---|---|
| Pauses and lists 1 fact and 1 topic | Pass |
| "no" keeps everything and clears the flag | Pass |
| "  Yes " deletes facts and scores | Pass |
| The session is cleared, leaving only the reply | Pass |
| The summary is emptied | Pass |
| Nothing stored: no pause | Pass |

Past sessions are still in the checkpointer. The reply says "this conversation" so it doesn't promise more.

### The parallel after-quiz step

After the quiz, two nodes run in the same step: `save_scores` writes the score, and `suggest_next` picks what to study. `report_quiz` has an arrow from each, so it waits for both and runs once.

The two branches write to different places, so they cannot conflict. `save_scores` writes only to the store, and `suggest_next` writes only `next_topic` and `next_reason` to the state. `suggest_next` takes this quiz's result from the state and leaves this topic out of the history it reads from the store, because `save_scores` may or may not have written it yet.

| Test | Result |
|---|---|
| 2 missed on loops | Suggested loops, naming `range` and `while` loops |
| Perfect on loops, sets already passed | Suggested functions: neither loops nor sets |
| A full quiz on a copy of the graph | `report_quiz` ran once, the score was saved, and the report included the suggestion |

Here, parallel saves almost no time: `save_scores` takes milliseconds and the model call takes about a second. The point is the structure, two independent jobs and one join, which pays off when both branches are slow.

---

## What the Project Taught

- **Fix bias in code, not in the prompt.** The model put the right answer on B 23 times out of 36. A shuffle fixed it completely.
- **Write what the model must never decide in code.** Marking, scores and the delete are all code. The model can't mark itself, raise a score or forget without a yes.
- **The trace tells you whether it's the code or the prompt.** The memory was in the system message every time. The bug was the model choosing not to use it.
- **Repeat a probe before trusting it.** One run of four variants said everything worked. Three runs each showed it was flaky.
- **Read the failures before changing the prompt.** One of the two eval failures was the label, not the model.
- **Rules clash at their edges.** "Store difficulties" and "don't store the current topic" disagreed only when a difficulty was a topic name.
- **Forgetting has to cover every copy.** The store was empty, and the session still knew.
- **A node runs again on resume.** Anything with a side effect goes after the pause.

---

## Project Structure

```
quizmate/
│
├── README.md
├── requirements.txt
├── .env                    # Not committed: OPENAI_API_KEY (and LangSmith keys, if used)
│
├── data/
│   ├── notes/              # 12 synthetic notes with front matter
│   └── learners.json       # Riya, Omar and Sam
│
├── ingest.py               # Notes → Chroma (kb_db/), one document per note
├── tools.py                # search_notes, get_note, TOPICS
├── storage.py              # SqliteSaver and SqliteStore on quizmate.db, config_for(learner, session)
├── memory.py               # Fact extraction, score records, format_memory, forget_learner
├── quiz_graph.py           # The quiz subgraph: write, shuffle, ask, mark, finish
├── chat_graph.py           # State, system message, all parent nodes, routers, graph
├── main.py                 # The terminal chat, with the quiz and forget loops
│
├── eval_memory.py          # Runs and scores the memory evaluation, or re-scores saved results
└── eval/
    ├── memory_messages.jsonl   # 20 labelled messages
    └── memory_results.jsonl    # The latest run's kept facts and reasons
```

`kb_db/` and `quizmate.db` are created when the project runs, and are not committed.

---

## How to Run

1. Create a virtual environment and install the dependencies:

```bash
python -m venv venv
```

```bash
venv\Scripts\activate
```

```bash
python -m pip install -r requirements.txt
```

2. Create a `.env` file with your `OPENAI_API_KEY`. To trace runs in LangSmith, add your LangSmith settings there too.

3. Build the notes store (one embedding call):

```bash
python ingest.py
```

4. Chat as one of the learners:

```bash
python main.py riya
```

5. Run the memory evaluation (20 `gpt-4o-mini` calls):

```bash
python eval_memory.py run
```

Re-score the saved results with the current labels, with no model calls:

```bash
python eval_memory.py rescore
```

---

## Known Limitations

- **"Why" questions about the learner get clumsy answers.** "I don't know why you're learning Python, but I do know you're learning it to automate reports" came up in 4 of 5 probe runs.
- **A request to change a score gets "I don't know".** The score is safe, but the tutor could explain that scores change only through quizzes.
- **The model reads its own headings aloud.** Replies sometimes start "What you know about Sam is…".
- **Forget me clears only the current session.** Earlier sessions stay in the checkpointer.
- **Quiz explanations often repeat the correct answer.** "The answer was X: X is…"
- **Small grammar slips in fixed replies,** such as "1 saved facts".
- **Facts and scores are read with a limit of 20 each.** A learner with more would have the rest left out of the system message.
- **Extracted facts sometimes use "they"** despite the prompt asking for the learner's name.
- **The evaluation is small and synthetic.** 20 messages, written by the same hand as the prompts, each run once at temperature 0. 20 of 20 means these cases pass, not that extraction is solved. The probes are smaller still.
- **One model throughout.** Every result is for `gpt-4o-mini`.

---

## Ideas for Extension

1. **Clear past sessions on forget me**, by listing and deleting the learner's other threads from the checkpointer.
2. **A held-out memory set,** written after the prompt was tuned, to test extraction on wording it hasn't seen.
3. **Explain instead of refusing** when a learner asks to change a score.
4. **Use the suggestion:** let the learner start the suggested quiz with one word.
5. **A `facts` command,** so learners can see what is stored without starting a forget.
6. **Spaced review:** use attempts and dates in the score records to suggest topics that are due for revision.
7. **Compare models** on the memory eval and the "why" probe, to see how much of the hedging is `gpt-4o-mini`.
