from enum import Enum

class PromptDatasetType(Enum):
    POSITIVE_PAIR = 0

class PromptDataset:
    """
    A prompt dataset which supports multiple possible format types as listed at
    <https://sbert.net/docs/sentence_transformer/dataset_overview.html>.

    This is primarily used for training the encoder. For training the reward
    model, see `RewardDataset`.

    Currently only supports positive pairs.
    """
    def __init__(
        self,
        pool: dict,
        type: PromptDatasetType = PromptDatasetType.POSITIVE_PAIR,
    ):
        self.type = type
        match PromptDatasetType:
            case PromptDatasetType.POSITIVE_PAIR:
                True

    def get_type(self):
        return self.type
