from sentence_transformers import SentenceTransformer

import math

class AgentRec:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.embeddings = {}

    def fit(self, training_samples: list[dict]):
        """
        Generates initial embeddings for AgentRec. These are used to generate
        agent recommendations by comparing these embeddings to an unseen
        candidate.

        Args:
            training_samples: A list of training samples. Each sample is a
                              dictionary with keys "agent_name" and "prompt"
        """
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

    def transform(self, prompt: str):
        """
        Returns cosine similarity scores that must be manually put into a score
        function. For a more streamlined function, use `get_agent`.

        Args:
            prompt: The prompt to compare to the initial embeddings
        """
        embedded_prompt = self.model.encode(prompt)
        similarities = {}

        for agent in self.embeddings:
            raw = self.model.similarity(embedded_prompt, self.embeddings[agent])
            similarities[agent] = sum(raw) / len(raw)

        return similarities

    def get_agent(
        self,
        prompt: str,
    ):
        """
        Returns the name of an agent given a natural language prompt. Note that
        the prompt must be reworded by a LLM before calling this function if
        the prompt is not a single sentence task description.

        Args:
            prompt: The prompt to generate a recommendation from.
        """
        similarities = self.transform(prompt)
