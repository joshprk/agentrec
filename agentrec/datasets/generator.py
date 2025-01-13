from agentrec.datasets import Agent
import jsonlines

from typing import Any, Optional

BATCH_SIZE = 50
CONTEXT_SIZE = 0
SYSTEM_PROMPT = """
You are a synthetic dataset generator specializing in creating diverse and \
realistic prompts for Large Language Models (LLMs). Your task is to generate \
a dataset of prompts, each labeled with the intended "agent" that would \
ideally handle it. Focus on creating a wide variety of prompts in terms of \
length, complexity, and format (e.g., questions, instructions, requests,
problem-solving tasks, open-ended challenges, FAQs, or myth-busting). Explore \
diverse topics related to the agent's domain to cater to different audiences,
skill levels, and contexts.

**Output Format:**

Each entry in the dataset should be JSON with the following structure:

```json
{
    "agent_name": "Agent name here"
    "content": "Prompt text here"
}
```
"""

class Generator:
    """
    Describes a generator which can generate a dataset of prompts addressed to
    multiple different agents. Each training example is a dictionary containing
    the keys `agent_name` and `prompt`. This generator internally utilizes the
    `AgentGenerator` class in order to generate prompts for each agent.

    Args:
        model: A class implementing __call__ for inferencing a LLM given an
               untokenized OpenAI-compatible context.
        agents: A list of agents that the generator should be able to generate.
                Each listed agent instantiates an `AgentGenerator` internally.
        batch_size: The number of training samples to generate per LLM call.
                    Defaults to `50`.
        store_context: Determines whether each LLM batch call should contain
                       the context generated from the previous LLM batch calls.
                       If `store_context` is not `0`, then this will decide the
                       number of messages to send to the LLM. This is defaulted
                       to `0` as it was found that context may cause duplicates,
                       but your mileage may vary.
    """
    def __init__(
        self,
        model: Any,
        agents: Agent | list[Agent],
        batch_size: int = BATCH_SIZE,
        store_context: int = CONTEXT_SIZE,
    ):
        if isinstance(agents, Agent):
            agents = [agents]
        
        self.agents = agents
        self.generators = {}
        for agent in agents:
            generator = AgentGenerator(model,
                                       agent=agent.name,
                                       agent_desc=agent.description,
                                       agent_examples=agent.examples if agent.examples is not None else [],
                                       batch_size=batch_size,
                                       store_context=store_context)
            self.generators[agent.name] = generator

    def get_agents(self):
        """
        Returns the list of agents that this generator was initialized with.
        """
        return self.agents

    def __len__(self):
        """
        Returns the number of agents that this generator was initialized with.
        """
        return len(self.agents)

    def __call__(self, agent: str):
        """
        Returns an `AgentGenerator` for the given `agent`.

        Args:
            agent: The agent which the returned `AgentGenerator` should handle.
        """
        return self.generators[agent]

class AgentGenerator:
    """
    Describes a generator which provides a number of raw training examples for
    agent classification tasks, explicitly focusing on a specific agent. Each
    training example is a dictionary containing the keys `agent_name` and `prompt`.
    This generator is intended to be created through the `Generator` class.

    Note that these samples are not deduplicated or cleaned. It is therefore the
    responsibility of the user to deduplicate and clean the training samples 
    through some method such as the MinHash algorithm.

    Args:
        model: A class implementing __call__ for inferencing a LLM given an
               untokenized OpenAI-compatible context.
        agent: The name of the AI agent that the training samples should refer to.
        agent_desc: A description of the agent passed to the system prompt. If it
                    is not specified, then no mention of a description is made.
                    Defaults to `None`.
        agent_examples: An optional list of prompts to prime the LLM context. If
                        it is not specified, then no mention of any examples are
                        made. A list of strings or a list of dictionaries with
                        a `content` key can be given. Defaults to `None`.
        batch_size: The number of training samples to generate per LLM call.
                    Defaults to `50`.
        store_context: Determines whether each LLM batch call should
                       contain the context generated from the previous
                       LLM batch calls. Defaults to `True`.
    """
    def __init__(
        self,
        model: Any,
        agent: str,
        agent_desc: Optional[str] = None,
        agent_examples: list[str | dict] = [],
        batch_size: Optional[int] = BATCH_SIZE,
        store_context: Optional[int] = CONTEXT_SIZE,
    ):
        self.model = model
        self.agent = agent
        self.agent_desc = agent_desc
        self.agent_examples = []
        self.batch_size = batch_size if batch_size is not None else BATCH_SIZE
        self.store_context = store_context if store_context is not None else CONTEXT_SIZE

        for example in agent_examples:
            if isinstance(example, str):
                self.agent_examples.append(example)
            elif isinstance(example, dict):
                self.agent_examples.append(example["content"])

        self.context = []
        self.batch = []

    def next_batch(self):
        """
        An internal function which inferences a LLM to generate the next batch
        of prompts to be given through the iterator. This can be called
        directly in order to receive a list of prompts.
        """
        if len(self.batch) > 0: 
            return self.batch

        if self.store_context > 0:
            while len(self.context) > self.store_context:
                self.context.pop(0)
        else:
            self.context = []

        agent_data = f"""
        The name of the agent is {self.agent}.
        """

        if self.agent_desc is not None:
            agent_data += \
                f"Here is a description of the agent: {self.agent_desc}\n\n"

        if len(self.agent_examples) > 0:
            examples = """
            Below are a few example prompts. Note that this is not a
            representative set of prompts and serves only as a few examples for
            what prompts the agent should be able to handle.\n\n
            """

            for example in self.agent_examples:
                examples += f"""
                {{
                    "agent_name": {self.agent},
                    "content": {example} }}\n """

            agent_data += examples

        instruction = f"Generate {self.batch_size} prompts for the given agent. \
                        Do not output anything else other than valid JSON."
        system_prompt = SYSTEM_PROMPT + agent_data + instruction
        user_prompt = [{"role": "user", "content": instruction}]
        model_context = [{"role": "system", "content": system_prompt}] + \
                        self.context + \
                        user_prompt

        # Expect OpenAI-compatible context list
        response = self.model(model_context)[-1]
        res = response["content"]
        batch = []

        self.context.extend(user_prompt)
        self.context.append(response)

        try:
            while True:
                start = res.index("{")
                end   = res.index("}")
                json  = res[start:end+1]
                res   = res[end+1:]
                batch.append(json)
        except ValueError:
            pass

        reader = jsonlines.Reader(batch)
        processed = []

        if reader is not None:
            for prompt in reader.iter(type=dict, skip_invalid=True):
                if "content" not in prompt:
                    continue

                processed.append({
                    "agent_name": self.agent,
                    "prompt": prompt["content"].strip(),
                })

        return processed

    def __next__(self):
        """
        Returns a single prompt in the form of a dictionary containing the keys
        `agent_name` and `content`. Do not call this in an iterator as it will
        block forever as it will never raise `StopIteration`. Instead, use the
        `next` function.
        """
        while not len(self.batch) > 0:
            self.batch = self.next_batch()

        return self.batch.pop()
