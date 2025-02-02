import jsonlines

from agentrec.datasets import Agent, Generator

from typing import Any, Optional
from pathlib import Path
import copy
import random

class PromptPool:
    """
    A class which generates and stores generated prompts from `Generator` in
    order to allow proper cleaning and deduplication of data. This class is
    serializable into a jsonl file which can be loaded at a later time.

    `PromptPool` ensures that prompts are sorted by the specific agent to make
    the manual cleaning of data more simple.
    """
    def __init__(self):
        self.pool = []
        self.agents = []

    def __len__(self):
        return len(self.pool)

    def set(self, agents: list[Agent]):
        """
        Sets the list of agents that the prompt pool should generate from. Note
        that if a pool was already generated before calling this function, that
        pool will be wiped in order to maintain consistency of the class.

        Args:
            agents: A list of agents which the prompt pool should generate from
        """
        self.agents = agents
        self.pool = []

    def generate(
        self,
        model: Any,
        per_agent: int,
        batch_size: Optional[int],
        store_context: Optional[int],
        progress: bool = False,
    ):
        """
        Generates the specified number of training samples and stores them into
        the class, so that they can be retrieved or saved to file. An argument
        must be specified, either `per_agent` or `total`, in order to generate
        training samples. If neither or both are specified, then a `ValueError`
        is thrown.

        Args:
            model: A class implementing __call__ to inference a LLM given an
                   untokenized OpenAI-like context.
            per_agent: The number of training samples that should be generated
                       for each agent.
            total: The number of training samples that should be generated for
                   the entire dataset, where this is divided by the number of
                   agents in order to find how many samples should be created
                   for each agent. If this number is not cleanly divisible by
                   the number of agents, a best-effort approach is made.
        """
        if not len(self.agents) > 0:
            raise ValueError("A list of agents must be specified first")

        per_agent = per_agent if per_agent is not None else total // len(self.agents)
        generator = Generator(model,
                              self.agents,
                              batch_size=batch_size,
                              store_context=store_context)

        for agent in self.agents:
            name      = agent.name
            agent_gen = generator(name)
            n         = 0

            if progress:
                print("[AgentRec] Generating prompts for", name)

            while n < per_agent:
                prompt  = next(agent_gen)
                n      += 1

                if progress:
                    print("[AgentRec]", str(n), "/", str(per_agent))

                self.pool.append(prompt)

    def shuffle(self, seed: Optional[int] = None):
        """
        Shuffles the `PromptPool` randomly. A seed can be provided to perform
        this operation deterministically.

        Args:
            seed: An optional random seed which allows deterministic shuffling
                  shuffling when specified.
        """
        return random.Random(seed).shuffle(self.pool)

    def split(self, n: int):
        """
        Splits the `PromptPool` by popping the last `n` training samples. Note
        that this does not check for safety, meaning that the user must make
        sure that there are enough training samples available in the pool to
        begin with.

        Args:
            n: The number of training samples to remove from the `PromptPool`
               and return.
        """
        popped    = self.pool[n:]
        self.pool = self.pool[:n]
        return popped

    def uniform(self, n: int):
        """
        Uniformly splits the `PromptPool`. This function will attempt to return
        a list of prompts where each agent has the same number of training
        samples out of a list of n training samples. It is recommended to
        shuffle the pool afterwards.

        Args:
            n: The number of training samples to remove from the `PromptPool`
               and return. The sum of all prompts of all agents will equal to
               this argument.
        """
        per_agent = n // len(self.agents)
        prompts = {}
        for agent in self.agents:
            name = agent["name"]
            prompts[name] = []
            idx = 0
            while (len(prompts[name]) < per_agent) and (idx < len(self.pool)):
                prompt = self.pool[idx]
                if prompt["agent_name"] == name:
                    prompts[name].append(self.pool.pop(idx))
                else:
                    idx += 1

        pool = []
        for agent in prompts:
            pool += prompts[agent]

        return pool

    def save(
        self,
        path: str,
        agent_path: str,
    ):
        """
        Serializes and saves the PromptPool such that it can be loaded into
        memory later. This requires the saving of two files, one to store the
        agent metadata at `agent_path` and another to store the prompts
        themselves at `path`. It is suggested that these files are stored
        together in the same directory.

        Args:
            agent_path: The file path where the agent metadata is stored
            path: The file path where the agent prompts are stored
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(agent_path).parent.mkdir(parents=True, exist_ok=True)

        agents = [agent.to_jsonl() for agent in self.agents]
        with jsonlines.open(agent_path, mode="w") as agent_file:
            agent_file.write_all(agents)
            agent_file.close()

        with jsonlines.open(path, mode="w") as prompt_file:
            prompt_file.write_all(self.pool)
            prompt_file.close()

    def save_split(
        self,
        train_path: str,
        test_path: str,
        test_split: float = 0.2,
    ):
        """
        Saves a train and test split which are uniformly split. It is
        recommended to use the `save` method alongside this method, as it is
        not possible to load a `PromptPool` from split files.

        Args:
            train_path: The file path where the train split is stored
            test_path: The file path where the test split is stored
            test_split: The percentage of all samples in the pool which are
                        allocated to the test split. Defaults to 20%.
        """
        train_pool = copy.deepcopy(self)
        test_pool = train_pool.uniform(int(len(train_pool) * test_split))

        Path(train_path).parent.mkdir(parents=True, exist_ok=True)
        Path(test_path).parent.mkdir(parents=True, exist_ok=True)

        with jsonlines.open(train_path, mode="w") as train_file:
            train_file.write_all(train_pool.pool)
            train_file.close()

        with jsonlines.open(test_path, mode="w") as test_file:
            test_file.write_all(test_pool)
            test_file.close()

        del train_pool
        del test_pool

    def load(
        self,
        path: str,
        agent_path: str,
    ):
        """
        Loads a PromptPool from two jsonlines files, an agent metadata file
        at `agent_path` and a prompt storage file at `path`.

        Args:
            agent_path: The file path where the agent metadata is stored
            path: The file path where the agent prompts are stored
        """
        self.agents = []
        with jsonlines.open(agent_path) as agent_file:
            for agent in agent_file:
                self.agents.append(agent)
            agent_file.close()

        with jsonlines.open(path) as prompt_file:
            for prompt in prompt_file:
                self.pool.append(prompt)
            prompt_file.close()
