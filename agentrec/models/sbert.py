from sentence_transformers import SentenceTransformer

import math

class SBERTAgentRec:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.embeddings = {}

    def fit(self, training_samples: dict):
        samples = {}

        for sample in training_samples:
            agent  = sample["agent_name"]
            prompt = sample["prompt"]
            if agent not in samples:
                samples[agent] = [prompt]
            else:
                samples[agent].append(prompt)

        self.embeddings = {}
        for agent in samples:
            self.embeddings[agent] = self.model.encode(samples[agent])

    def transform(self, prompt):
        embedded_prompt = self.model.encode(prompt)
        similarities = {}

        for agent in self.embeddings:
            raw = self.model.similarity(embedded_prompt, self.embeddings[agent])
            similarities[agent] = sum(raw) / len(raw)

        return similarities
