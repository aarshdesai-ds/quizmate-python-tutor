import json
import sys
from collections import defaultdict
from memory import extract_memories

ROWS_PATH = "eval/memory_messages.jsonl"
RESULTS_PATH = "eval/memory_results.jsonl"

with open("data/learners.json",encoding="utf-8") as f:
    learners = json.loads(f.read())

def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def kept_facts(decision):
    if not decision.should_write:
        return []
    facts = []
    for memory in decision.memories:
        if memory.is_new:
            facts.append(memory.text)
    return facts

def score_row(row, kept):
    if row['expect_write'] and not kept:
        return "expected a write, kept nothing"
    elif not row['expect_write'] and kept:
        if len(kept) > 1:
            return f"expected no write, kept {len(kept)} facts"
        else:
            return f"expected no write, kept 1 fact"
    elif row['must_contain']:
        if not any(row['must_contain'].lower() in fact.lower() for fact in kept):
            return f"no fact contains '{row['must_contain'].lower()}'"
    if row['must_not_contain']:
        if any(row['must_not_contain'].lower() in fact.lower() for fact in kept):
            return f"a fact contains '{row['must_not_contain'].lower()}'"
    return ""

def run_eval(rows):
    results = []
    for row in rows:
        learner_id = row['learner_id']
        name = learners[learner_id]['name']
        existing = row['existing']
        message = row['message']
        config = {"tags":["quizmate", "eval", "memory"], "metadata": {"row_id": row['id'], "group": row['group']}}
        decision = extract_memories(name, existing, message, config = config)
        kept = kept_facts(decision)
        reason = score_row(row, kept)
        record = {"id": row['id'], "group": row['group'], "message": message, "kept": kept, "reason": reason}
        results.append(record)
    return results

def save_results(results, path):
    with open(path, "w", encoding="utf-8") as f:
        for result in results:
            json_line = json.dumps(result, ensure_ascii=False)
            f.write(json_line + "\n")

def print_report(results):
    total = len(results)
    passes = len([result for result in results if not result['reason']])
    print(f"total: {passes}/{total}")
    group_tally = defaultdict(lambda:{"total":0, "passed":0})
    for result in results:
        group = result['group']
        reason = result['reason']
        group_tally[group]['total'] += 1
        if not reason:
            group_tally[group]['passed'] += 1
    for group_name, stats in group_tally.items():
        print(f"     {group_name:<14} {stats['passed']}/{stats['total']}")
    failures = [result for result in results if result['reason']]
    if not failures:
        print("no failures")
    else:
        print("FAILED")
        for failure in failures:
            print(f"{failure['id']:<4} {failure['group']:<10} {failure['reason']}")
            print(f"message: {failure['message']}")
            print(f"kept:    {failure['kept']}")
            print()

if __name__ == "__main__":
    print("--- rows (free) ---")
    rows = load_rows(ROWS_PATH)
    ids = [row["id"] for row in rows]
    print("ok  " if len(rows) == 20 else "FAIL", "rows loaded:", len(rows))
    print("ok  " if len(set(ids)) == len(ids) else "FAIL", "ids are unique")
    print("ok  " if sum(row["expect_write"] for row in rows) == 11 else "FAIL", "11 write rows, 9 no-write rows")
    
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        print("\n--- eval: 20 rows (20 model calls) ---")
        results = run_eval(rows)
        save_results(results, RESULTS_PATH)
        print_report(results)
        print(f"\nsaved to {RESULTS_PATH}")

    if len(sys.argv) > 1 and sys.argv[1] == "rescore":
        print("\n--- rescore saved results with current labels (free) ---")
        rows_by_id = {row["id"]: row for row in rows}
        results = load_rows(RESULTS_PATH)
        for result in results:
            row = rows_by_id[result["id"]]
            result['group'] = row['group']
            result["reason"] = score_row(row, result["kept"])
        print_report(results)