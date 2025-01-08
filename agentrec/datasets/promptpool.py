from agentrec.datasets import Agent

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

    def generate(
        self,
        per_agent: Optional[int | list[int]] = None,
        total: Optional[int] = None,
    ):
        """
        Generates the specified number of training samples and stores them into
        the class, so that they can be retrieved or saved to file. An argument
        must be specified, either `per_agent` or `total`, in order to geerate
        training samples. If neither or both are specified, then a `ValueError`
        is thrown.

        Args:
            per_agent: The number of training samples that should be generated
                       for each agent.
            total: The number of training samples that should be generated for
                   the entire dataset, where this is divided by the number of
                   agents in order to find how many samples should be created
                   for each agent. If this number is not cleanly divisible by
                   the number of agents, a best-effort approach is made.
        """
        pass

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
        pass

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
        pass
