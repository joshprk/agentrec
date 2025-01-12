from dotenv import load_dotenv
from transformers import AutoTokenizer, pipeline

from agentrec.datasets import Agent, PromptPool
from agentrec.models import SBERTAgentRec

import math

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
AGENTS = [
    Agent("Biology Agent"),
    Agent("Cooking Agent"),
    Agent("Math Agent"),
    Agent("Gaming Agent"),
    Agent("Therapy Agent"),
    Agent("Reading Agent"),
    Agent("Health Agent"),
    Agent("Fitness Agent"),
]

DEVICE = "cuda:4"
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 50
REPETITION_PENALTY = 1.2
CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'user' %}{{ ' ' }}{% endif %}{{ message['content'] }}{% if not loop.last %}{{ ' ' }}{% endif %}{% endfor %}{{ eos_token }}"

class Llama3:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.tokenizer.chat_template = CHAT_TEMPLATE
        self.model = pipeline(task="text-generation",
                              model=MODEL_ID,
                              torch_dtype="bfloat16",
                              device_map=DEVICE,
                              tokenizer=self.tokenizer)

    def __call__(self, context):
        text = self.model(
            context,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=MAX_NEW_TOKENS,
            return_full_text=False,
            temperature=TEMPERATURE,
            do_sample=True,
            top_p=TOP_P,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
            truncation=True,
            )[0]["generated_text"] # pyright: ignore

        context.append({
            "role": "assistant",
            "content": text,
        })

        return context

def main():
    model = Llama3()
    pool  = PromptPool()
    pool.set(AGENTS)
    pool.generate(model, per_agent=50, batch_size=50, progress=True)

    pool.save(path="./data/prompts.jsonl",
              agent_path="./data/agents.jsonl")

def main2():
    pool = PromptPool()
    pool.load(path="./data/prompts.jsonl",
              agent_path="./data/agents.jsonl")

    classifier = SBERTAgentRec("all-mpnet-base-v2")
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
    main2()
