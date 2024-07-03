from .base_experiment import BaseExperiment
from models.vit import create_vit_model
from datasets.data_utils import get_dataloaders
from utils.helpers import train_epoch, validate
import torch
import torch.nn as nn
import torch.optim as optim
from peft import get_peft_model, LoraConfig

class LoRAFinetuning(BaseExperiment):
    def setup(self):
        self.model = create_vit_model(self.config['model'], self.config['data']['num_classes'])
        lora_config = LoraConfig(
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['alpha'],
            target_modules=self.config['lora']['target_modules'],
            lora_dropout=self.config['lora']['dropout'],
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.train_loader, self.test_loader = get_dataloaders(self.config['data'])
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['training']['learning_rate'])

    def run(self):
        self.setup()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        for epoch in range(self.config['training']['epochs']):
            train_loss, train_acc = train_epoch(self.model, self.train_loader, self.criterion, self.optimizer, device)
            val_acc = validate(self.model, self.test_loader, device)
            
            print(f"Epoch {epoch+1}/{self.config['training']['epochs']}:")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Acc: {val_acc:.2f}%")

        torch.save(self.model.state_dict(), self.config['output']['path'])