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
serializing the `PromptPool`. A usage example is given in `test.py`. Note that
a Huggingface API key must be inserted in `.env`.

Afterwards, one can finetune the model by using `finetune.py`. After finetuning,
it is possible to try out the agent reocmmendation system by using `test.py`.
