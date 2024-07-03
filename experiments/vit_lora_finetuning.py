import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from utils import Config, UtilsManager, make_lora_model
from datasets.data_utils import get_dataset, get_transforms
from datasets.datasets import CUB200Dataset, load_cub200_dataset
from models.vit import create_vit_model

logger = logging.getLogger(__name__)

class VitLoraFinetuningExperiment:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.device = UtilsManager.get_device()
        self.model = self._create_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.training['learning_rate'])
        self.train_loader, self.val_loader = self._get_data_loaders()

    def _create_model(self):
       
        num_classes= self.config.datasets[str(self.args.dataset)]['num_classes']
        base_model = create_vit_model(num_classes)
        lora_model = make_lora_model(base_model, self.config.lora['rank'])
        return lora_model.to(self.device)

    def _get_data_loaders(self):
        if self.args.dataset == "cub200":
            train_loader, val_loader = load_cub200_dataset(self.args.data_path, batch_size=self.config.training['batch_size'],image_size=224)
        else:

            train_transform, val_transform = get_transforms(self.args.dataset)
            train_dataset = get_dataset(self.args.dataset, self.args.data_path, train=True, transform=train_transform)
            val_dataset = get_dataset(self.args.dataset, self.args.data_path, train=False, transform=val_transform)
            
            train_loader = DataLoader(train_dataset, batch_size=self.config.training['batch_size'], shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=self.config.training['batch_size'], shuffle=False, num_workers=4)
            
        return train_loader, val_loader

    def train_epoch(self):
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        for inputs, labels in tqdm(self.train_loader, desc="Training"):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return total_loss / len(self.train_loader), 100. * correct / total

    def validate(self):
        self.model.eval()
        total, correct = 0, 0

        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc="Validating"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return 100. * correct / total

    def run(self):
        logger.info(f"Starting ViT-LoRA finetuning on {self.args.dataset}")
        UtilsManager.setup_wandb(self.config)

        for epoch in range(self.config.training['num_epochs']):
            train_loss, train_acc = self.train_epoch()
            val_acc = self.validate()
            
            logger.info(f"Epoch {epoch+1}/{self.config.training['num_epochs']}:")
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Acc: {val_acc:.2f}%")
            
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_acc": val_acc
            })

        # Save the final model
        UtilsManager.save_model(self.model, f"{self.config.output_path}/vit_lora_finetuned_{self.args.dataset}.pth")
        logger.info(f"Experiment completed. Model saved to {self.config.output_path}/vit_lora_finetuned_{self.args.dataset}.pth")

        wandb.finish()