from memledger import Ledger, Policy


def main() -> None:
    ledger = Ledger(path="./coding-memory.db", policy=Policy.default())
    try:
        first = ledger.session(user_id="dev_123")
        first.observe(
            user="Please use Python only; I don't read Go.",
            assistant="Understood, I will keep examples in Python.",
        )
        first.checkpoint()

        follow_up = ledger.session(user_id="dev_123")
        memories = follow_up.recall("show me an example", k=5)
        context = follow_up.build_context(instinct=True, episodic=memories, working="tail")
        print(context.system)
    finally:
        ledger.close()


if __name__ == "__main__":
    main()