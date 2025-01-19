from dotenv import load_dotenv

from agentrec.datasets import PromptPool
from agentrec.models import SBERTAgentRec

import math

def main():
    pool = PromptPool()

    pool.load(path="./data/prompts.jsonl",
              agent_path="./data/agents.jsonl")

    classifier = SBERTAgentRec("all-mpnet-base-v2")
    classifier.fit(pool.pool)

    embeddings = classifier.embeddings

if __name__ == "__main__":
    load_dotenv()
    main()
