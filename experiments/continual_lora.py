import logging
from itertools import product
import torch
import torch.nn as nn
import wandb
from avalanche.training.supervised import Naive
from avalanche.training.plugins import ReplayPlugin
from utils.utils import Config, UtilsManager, PruningPlugin, make_lora_model
from datasets.data_utils import get_transforms, get_cl_benchmark
from models.vit import create_vit_model
#

logger = logging.getLogger(__name__)

class ContinualLoRAExperiment:
    def __init__(self, args):
        self.args = args
        self.config = Config(args.config)

    def run(self):
        UtilsManager.setup_wandb(self.config)

        device = UtilsManager.get_device()
        train_transform, test_transform = get_transforms(self.args.dataset)

        scenario = get_cl_benchmark(
            self.args.dataset, 
            self.args.data_path, 
            train_transform, 
            test_transform, 
            self.args.benchmark_type,
            n_experiences=self.config.n_experiences
        )

        hyperparameters = list(product(self.args.ranks, self.args.epochs, self.args.thetas))

        for rank, epoch, theta in hyperparameters:
            model = create_vit_model(self.config.num_classes)
            self.train_model(rank, epoch, theta, model, scenario)

    def train_model(self, rank, epoch, theta, model, scenario):
        logger.info(f"Training Continual LoRA model with rank {rank}, epoch {epoch}, theta {theta}")
        
        device = UtilsManager.get_device()
        eval_plugin = UtilsManager.get_evaluation_plugin()
        
        cl_strategy = Naive(
            model,
            torch.optim.Adam(model.parameters(), lr=self.config.learning_rate),
            nn.CrossEntropyLoss(),
            train_mb_size=self.config.batch_size,
            train_epochs=epoch,
            eval_mb_size=self.config.batch_size,
            device=device,
            evaluator=eval_plugin,
            plugins=[
                PruningPlugin(theta, 'default', self.config.num_classes),
                ReplayPlugin(mem_size=self.config.replay_size)
            ]
        )

        results = []
        for experience in scenario.train_stream:
            logger.info(f"Start of experience: {experience.current_experience}")

            cl_strategy.train(experience)
            results.append(cl_strategy.eval(scenario.test_stream))

            logger.info(f"End of experience: {experience.current_experience}")

        # Log results
        wandb.log({
            "rank": rank,
            "epoch": epoch,
            "theta": theta,
            "final_accuracy": results[-1]['Top1_Acc_Stream/eval_phase/test_stream']
        })

        # Save model
        UtilsManager.save_model(cl_strategy.model, f"{self.config.output_path}/continual_lora_{self.args.dataset}_{self.args.benchmark_type}_{rank}_{epoch}_{theta}.pth")