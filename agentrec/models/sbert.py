from sentence_transformers import SentenceTransformer

import math

class SBERTAgentRec:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.embeddings = {}

    def fit(self, training_samples: dict):
        samples = {}

        for sample in training_samples:
            agent = training_samples["agent_name"]
            content    = training_samples["content"]
            if agent not in samples:
                samples[agent] = [content]
            else:
                samples[agent].append(content)

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
