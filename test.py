from dotenv import load_dotenv

from agentrec.datasets import PromptPool
from agentrec.models import SBERTAgentRec

import math

def main():
    pool = PromptPool()

    pool.load(path="./data/prompts.jsonl",
              agent_path="./data/agents.jsonl")

    classifier = SBERTAgentRec("./models/test_model/")
    #classifier = SBERTAgentRec("all-mpnet-base-v2")
    classifier.fit(pool.pool)
    
    while stdin := input("> "):
        raw = classifier.transform(stdin)
        scores = {}

        for agent in raw:
            scores[agent] = sum(raw[agent]) / len(raw[agent])

        best = ""
        best_score = -math.inf

        for agent in scores:
            if scores[agent] > best_score:
                best = agent
                best_score = scores[agent]

        print("Selected Agent:", best)

if __name__ == "__main__":
    load_dotenv()
    main()
