import os

from memledger import Ledger, Policy


def generate_reply(user_message: str, context: str) -> str:
    return f"Context:\n{context}\n\nUser: {user_message}"


def main() -> None:
    ledger = Ledger(
        path="./example-memory.db",
        policy=Policy.default(),
        memory_model=os.environ.get(
            "MEMORY_MODEL",
            "openai-compat:http://localhost:11434/v1|qwen3:4b",
        ),
    )
    session = ledger.session(user_id="demo")
    try:
        while True:
            message = input("> ").strip()
            if not message:
                break
            memories = session.recall(message, k=5)
            context = session.build_context(instinct=True, episodic=memories, working="tail")
            reply = generate_reply(message, context.system)
            session.observe(user=message, assistant=reply)
            print(reply)
        print(session.checkpoint())
    finally:
        ledger.close()


if __name__ == "__main__":
    main()