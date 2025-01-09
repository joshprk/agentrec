from dotenv import load_dotenv
from transformers import AutoTokenizer, pipeline
import torch

from agentrec.datasets import PromptPool

import os

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
AGENTS = [
    "Biology Agent",
    "Cooking Agent",
    "Math Agent",
]

class Llama3:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ ' ' }}{% endif %}{{ message['content'] }}{% if not loop.last %}{{ ' ' }}{% endif %}{% endfor %}{{ eos_token }}"
        self.model = pipeline(task="text-generation",
                              model=MODEL_ID,
                              torch_dtype=torch.bfloat16,
                              device_map="cuda:3",
                              tokenizer=self.tokenizer)

    def __call__(self, context):
        text = self.model(
            context,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
            max_length=1024,
            return_full_text=False,
            temperature=0.6,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.2,
            truncation=True,
        )[0]["generated_text"]

        context.append({
            "role": "assistant",
            "content": text,
        })

        print(text)

        return context

def main():
    model = Llama3()
    pool  = PromptPool()
    pool.set(AGENTS)
    pool.generate(model, per_agent=1)

    pool.save(
        path="./data/prompts.jsonl",
        agent_path="./data/agents.jsonl",
    )


if __name__ == "__main__":
    load_dotenv()
    main()
