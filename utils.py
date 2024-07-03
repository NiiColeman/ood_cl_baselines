import json
import logging
import torch
import wandb
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics, loss_metrics
from avalanche.logging import InteractiveLogger, TensorboardLogger, WandBLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.core import SupervisedPlugin
import torch.nn.utils.prune as prune
import peft
import numpy as np
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class Config:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        self.__dict__.update(config_data)

class UtilsManager:
    @staticmethod
    def setup_wandb(config):
        wandb.init(project=config.wandb_project, mode=config.wandb_mode, name=config.wandb_name)

    @staticmethod
    def get_device():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_evaluation_plugin():
        loggers = [InteractiveLogger(), TensorboardLogger(), WandBLogger()]
        return EvaluationPlugin(
            accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            forgetting_metrics(experience=True, stream=True),
            loggers=loggers
        )

    @staticmethod
    def save_model(model, path):
        torch.save(model.state_dict(), path)
        logger.info(f"Model saved to {path}")



# ... (rest of the utils.py content)

class PruningPlugin(SupervisedPlugin):
    def __init__(self, theta, task_name, num_classes):
        super().__init__()
        self.theta = theta
        self.task_name = task_name
        self.num_classes = num_classes

    def before_backward(self, strategy, **kwargs):
        device = UtilsManager.get_device()
        mask = strategy.experience.classes_in_this_experience
        not_mask = np.setdiff1d(np.arange(self.num_classes), mask)
        not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
        strategy.mb_output.index_fill_(dim=1, index=not_mask, value=float("-inf"))
        strategy.loss = F.cross_entropy(strategy.mb_output, strategy.mb_y)

    def after_training_epoch(self, strategy, **kwargs):
        for block in strategy.model.base_model.model.blocks:
            qkv_layer = block.attn.qkv
            for name, child in qkv_layer.named_children():
                if name in ['lora_A', 'lora_B']:
                    prune.random_unstructured(getattr(child, self.task_name), name='weight', amount=self.theta)
                    prune.remove(getattr(child, self.task_name), 'weight')
        return strategy.model

def save_model(model, path):
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to {path}")

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    logger.info(f"Model loaded from {path}")
    return model

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def log_metrics(metrics, step=None):
    for key, value in metrics.items():
        wandb.log({key: value}, step=step)

def create_optimizer(model, lr, optimizer_type='adam'):
    if optimizer_type.lower() == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type.lower() == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def make_lora_model(model, rank):
    lora_config = peft.LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=['qkv'],
        lora_dropout=0.1,
        bias="none",
    )
    return peft.get_peft_model(model, lora_config)