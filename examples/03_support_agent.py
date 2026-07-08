from memledger import Ledger, Policy


def main() -> None:
    ledger = Ledger(path="./support-memory.db", policy=Policy.default())
    session = ledger.session(user_id="support")
    try:
        session.observe(
            user="The staging deploy failed due to missing env vars.",
            assistant="I will note that and help diagnose it.",
        )
        session.outcome("failure", task="staging_deploy")
        print(session.checkpoint())
        print(ledger.stats())
    finally:
        ledger.close()


if __name__ == "__main__":
    main()