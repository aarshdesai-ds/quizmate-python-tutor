import sys
from chat_graph import learners, workflow
from langchain_core.messages import AIMessage, HumanMessage
from storage import config_for
from uuid import uuid4

learners_list = list(learners.keys())
existing_learners = ", ".join(learners_list)

if len(sys.argv) == 1:
        print(f"Usage: python main.py <learner_id>\nLearners: {existing_learners}")
        sys.exit()
else:
        learner_id = sys.argv[1].lower().strip()
        if not learner_id:
            print(f"Usage: python main.py <learner_id>\nLearners: {existing_learners}")
            sys.exit()
        elif learner_id not in learners_list:
            print(f"Unknown learner: {learner_id}\nLearners: {existing_learners}")
            sys.exit()
        else:
              learner = learners[learner_id]
              session_id = uuid4().hex[:6]
              name = learner['name']
              level = learner['level']
              print(f"Quizmate for {name} ({level}). Session {session_id}. Type 'summary', 'history', or 'exit'.")

config = {**config_for(learner_id, session_id), "run_name": "quizmate-chat", "tags":['quizmate', 'cli'], "metadata": {"learner_id": learner_id, "session_id": session_id}}
while True:
    user_input = input("you> ").strip()
    if not user_input:
        continue
    if user_input.lower() == "exit":
        print(f"Bye, {name}!")
        break
    elif user_input.lower() == "history":
            snapshot = workflow.get_state(config).values
            messages = snapshot.get("messages",[])
            summary = snapshot.get("summary", "")
            if not messages and not summary:
                print("No conversation yet.")
                continue
            print(f"--- Session {session_id} with {name} ---")
            if summary:
                 print(f"Earlier (summarised): {summary}")
            for message in messages:
    
                if isinstance(message, HumanMessage):
                    print(f"you: {message.content}")
    
                elif isinstance(message, AIMessage):
                    if message.tool_calls:
                        continue
                    print(f"bot: {message.content}")
                else:
                    continue
            print(f"--- end ---")
    elif user_input.lower() == "summary":
        snapshot = workflow.get_state(config).values
        summary =  snapshot.get("summary")
        if not summary:
             print("No summary yet.")
        else:
             print(f"Summary:\n{summary}")
    else:
        try:
            result = workflow.invoke({"messages":[user_input], "learner_id": learner_id}, config=config)
            last_message = result['messages'][-1].content
            print(f"bot> {last_message}")
            
        except Exception:
            print("bot> Sorry, something went wrong...")     

