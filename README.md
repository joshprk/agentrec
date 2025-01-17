# Agent Recommendation

This repository contains code that assists with the synthetic generation of
a prompt dataset labeled with the appropriate AI agent that would handle it,
as well as a method to recommend the use of a specific agent when a new prompt
is given through the use of human-aligned latent space embeddings.

## Usage

It is important to first install the dependencies. This repository was written
using Python 3.12. It is recommended to use a venv with the following commands:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

A LLM is also necessary, which must allow the input and output of an OpenAI
formatted context. More details about implementing this are in the documentation.

A synthetic dataset can be generated using the `agentrec.datasets.PromptPool`
class. A list of agent names must be given, which can have optional
descriptions and examples tied to them. This dataset can then be saved through
serializing the `PromptPool`. A usage example is given in `generatedata.py`.
Note that a Huggingface API key must be inserted in `.env`.

Afterwards, one can finetune the model by using `finetune.py`. After finetuning,
it is possible to try out the agent recommendation system by using `test.py`.

## Advantages

Traditional classification methods must be re-trained if more classes, or
agents, are added or removed over time. Furthermore, other classification
methods can be extremely slow. This method solves these problems, and more.
Below are some of the benefits of using this implementation of agent
recommendation:

- **Adaptive.** Theoretically, removing or adding agents does not change the
classification of the existing ones as the prompt encodings are still
calculated in the same way. You will not need to re-train from scratch if you
add a new agent. Furthermore, there are emerging properties that allow some
correct classification of agents that were not necessary trained with to begin
with.
- **Fast.** SBERT is a Siamese architecture which encodes the prompts before
comparison. Even if you do not cache the embeddings, a set of 1,000 prompts can
take only 5 seconds to fully encode into embeddings. Classifying a single
prompt generally takes less than 0.01 milliseconds.
- **Interpretable.** Prompts that belong to the same agent have embeddings
which cluster together as near neighbors. The embeddings themselves can be
interpreted as encoding semantic characteristics which are relevant to
recommending an agent.
