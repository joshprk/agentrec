from agentrec.datasets import PromptPool
from agentrec.models import SBERTAgentRec
from datasets import Dataset
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import BatchAllTripletLoss

AGENT_FILE = "./data/agents.jsonl"
PROMPT_FILE = "./data/prompts.jsonl"
OUTPUT_DIR = "./models/test_model/"
BASE_MODEL_ID = "all-mpnet-base-v2"
SHUFFLE_SEED = 42

def main():
    pool = PromptPool()
    pool.load(PROMPT_FILE, AGENT_FILE)
    pool.shuffle(SHUFFLE_SEED)

    label_map = []

    data = {
        "sentence": [],
        "label": [],
    }

    for item in pool.pool:
        data["sentence"].append(item["prompt"])

        label = None
        if item["agent_name"] in label_map:
            label = label_map.index(item["agent_name"])
        else:
            label = len(label_map)
            label_map.append(item["agent_name"])

        data["label"].append(label)

    train_dataset = Dataset.from_dict(data)
    model = SBERTAgentRec(BASE_MODEL_ID)
    loss = BatchAllTripletLoss(model.model)
    args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,
        per_device_eval_batch_size=16,
        per_device_train_batch_size=16,
        warmup_ratio=0.1,
        num_train_epochs=1,
    )

    trainer = SentenceTransformerTrainer(
        model=model.model,
        train_dataset=train_dataset,
        loss=loss,
        args=args,
    )

    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    main()
