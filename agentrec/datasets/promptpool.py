class PromptPool:
    """
    A class which generates and stores generated prompts from `Generator` in
    order to allow proper cleaning and deduplication of data. This class is
    serializable into a jsonl file which can be loaded at a later time.

    `PromptPool` ensures that prompts are sorted by the specific agent to make
    the manual cleaning of data more simple.
    """
    def __init__(self):
        pass
