import torch
from model import get_model
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
from torch.nn.attention import sdpa_kernel, SDPBackend
from config import Config


class CipherPlainData(Dataset):
    def __init__(self, directory_path):
        # TODO: find out what we need here
        pass

    def __getitem__(self, idx):
        # TODO: Find out what we need here
        pass


def train():
    model = get_model()

    args = TrainingArguments(
        output_dir=Config.output_dir,
        num_train_epochs=Config.epochs,
        per_device_train_batch_size=Config.batch_size,
        gradient_accumulation_steps=Config.grad_accum,
        learning_rate=Config.learning_rate,
        weight_decay=Config.weight_decay,
        gradient_checkpointing=Config.grad_checkpoint,
        logging_steps=Config.log_steps,
        save_steps=Config.save_steps,
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        # TODO: Change below to the correct path
        train_dataset=CipherPlainData("cipher"),
    )

    print(f"Training RecurrentGemma on {torch.cuda.get_device_name(0)}...")

    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        trainer.train()

    trainer.save_model(f"{Config.output_dir}/model")


if __name__ == "__main__":
    train()
