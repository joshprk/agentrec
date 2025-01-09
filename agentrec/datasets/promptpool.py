import jsonlines

from agentrec.datasets import Agent, Generator

from typing import Any, Optional
from pathlib import Path

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
        self.agents = None

    def set(self, agents: list[Agent | str]):
        """
        Sets the list of agents that the prompt pool should generate from. Note
        that if a pool was already generated before calling this function, that
        pool will be wiped in order to maintain consistency of the class.

        Args:
            agents: A list of agents which the prompt pool should generate from
        """
        if self.agents is not None:
            self.pool = []

        for idx, agent in enumerate(agents):
            if isinstance(agent, str):
                agents[idx] = Agent(agent)

        self.agents = agents

    def generate(
        self,
        model: Any,
        per_agent: Optional[int | list[int]] = None,
        total: Optional[int] = None,
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
        if self.agents is None:
            raise ValueError("A list of agents must be specified first")
        elif per_agent is None and total is None:
            raise ValueError("A number of training samples to generate must be \
                              specified in order to generate")
        elif per_agent is not None and total is not None:
            raise ValueError("Only one parameter can be set which specifies the \
                              number of training samples to generate.")

        per_agent = per_agent if per_agent is not None else total // len(self.agents)
        generator = Generator(model, self.agents)
        
        for agent in self.agents:
            name      = agent.name
            agent_gen = generator(name)
            n         = 0

            while n < per_agent:
                prompt  = next(agent_gen)
                n      += 1
                self.pool.append(prompt)

    def save(
        self,
        path: str,
        agent_path: str,
    ):
        """
        Serializes and saves the PromptPool such that it can be loaded into
        memory later. This requires the saving of two files, one to store
        the agent metadata at `agent_path` and another to store the prompts
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
