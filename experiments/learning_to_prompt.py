import logging
import torch
import torch.nn as nn
import wandb
from utils import Config, UtilsManager
from datasets.data_utils import get_transforms, get_cl_benchmark
from models.vit_l2p import create_vit_l2p_model

logger = logging.getLogger(__name__)

class LearningToPromptExperiment:
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
            # train_transform, 
            # test_transform, 
            self.args.benchmark_type,
            n_experiences=10  #self.config.cl['n_experiences']
        )

        self.train_model(scenario)

    def train_model(self, scenario):
        logger.info("Training Learning to Prompt model")
        
        device = UtilsManager.get_device()
        eval_plugin = UtilsManager.get_evaluation_plugin()
        
        num_classes = self.config.datasets[str(self.args.dataset)]['num_classes']
        model = create_vit_l2p_model(num_classes)
        
        cl_strategy = LearningToPrompt(
            model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=self.config.training['learning_rate']),
            criterion=nn.CrossEntropyLoss(),
            train_mb_size=self.config.training['batch_size'],
            train_epochs=self.config.training['num_epochs'],
            eval_mb_size=self.config.training['batch_size'],
            device=device,
            evaluator=eval_plugin,
        )

        results = []
        for experience in scenario.train_stream:
            logger.info(f"Start of experience: {experience.current_experience}")

            cl_strategy.train(experience)
            results.append(cl_strategy.eval(scenario.test_stream))

            logger.info(f"End of experience: {experience.current_experience}")

        # Log results
        wandb.log({
            "final_accuracy": results[-1]['Top1_Acc_Stream/eval_phase/test_stream']
        })

        # Save model
        UtilsManager.save_model(cl_strategy.model, f"{self.config.output_path}/l2p_{self.args.dataset}_{self.args.benchmark_type}.pth")