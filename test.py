from dotenv import load_dotenv
import numpy as np

from agentrec.datasets import PromptPool
from agentrec.models import SBERTAgentRec

import math

OUTPUT_ALGO = "pmean"
PMEAN = 2

def main():
    pool = PromptPool()
    pool.load(path="./data/train.jsonl",
              agent_path="./data/agents.jsonl")

    test_pool = PromptPool()
    test_pool.load(path="./data/test.jsonl",
              agent_path="./data/agents.jsonl")

    classifier = SBERTAgentRec("./models/test_model/")
    #classifier = SBERTAgentRec("all-mpnet-base-v2")
    classifier.fit(pool.pool)

    accurate = 0
    total    = 0
    for obj in test_pool.pool:
        agent_name = obj["agent_name"]
        prompt = obj["prompt"]

        raw = classifier.transform(prompt)
        scores = {}

        match OUTPUT_ALGO:
            case "arithmetic_mean":
                for agent in raw:
                    scores[agent] = sum(raw[agent]) / len(raw[agent])
            case "pmean":
                for agent in raw:
                    score_total = 0
                    for score in raw[agent]:
                        score_total += score ** PMEAN

                    score_total /= len(raw[agent])
                    scores[agent] = score_total ** (1 / PMEAN)
        best = ""
        best_score = -math.inf

        for agent in scores:
            if scores[agent] > best_score:
                best = agent
                best_score = scores[agent]

        if best == agent_name:
            accurate += 1

        print(total, "/", len(test_pool.pool))
        total += 1

    print("Test accuracy:", float(accurate) / float(total))

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
