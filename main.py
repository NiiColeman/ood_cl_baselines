import argparse
import logging
from experiments.continual_lora import ContinualLoRAExperiment
from experiments.learning_to_prompt import LearningToPromptExperiment
from experiments.naive_finetuning import NaiveFineTuningExperiment
from experiments.vit_lora_finetuning import VitLoraFinetuningExperiment
from utils import Config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Continual Learning and Finetuning Experiments")
    parser.add_argument("--config", type=str, default="resources/config.json", help="Path to the configuration file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=["cifar100", "stream51", "cub200", "fgvc_aircraft", "tinyimagenet"], 
                        help="Dataset to use")
    parser.add_argument("--experiment", type=str, required=True, 
                        choices=["continual_lora", "l2p", "naive_finetuning", "vit_lora_finetuning"], 
                        help="Experiment to run")
    parser.add_argument("--output_path", type=str, default="output", help="Path to save outputs")
    parser.add_argument("--benchmark_type", type=str, choices=["nc", "ni", "cir"], default="nc", 
                        help="Type of continual learning benchmark (for CL experiments)")
    parser.add_argument("--lora_rank", type=int, default=4, help="Rank for LoRA adaptation")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")

    args = parser.parse_args()
    config = Config(args.config)
    if args.experiment == "continual_lora":
        experiment = ContinualLoRAExperiment(args)
    elif args.experiment == "l2p":
        experiment = LearningToPromptExperiment(args)
    elif args.experiment == "naive_finetuning":
        experiment = NaiveFineTuningExperiment(args)
    elif args.experiment == "vit_lora_finetuning":
        experiment = VitLoraFinetuningExperiment(args,config)
    else:
        raise ValueError("Invalid experiment type")

    experiment.run()

if __name__ == "__main__":
    main()