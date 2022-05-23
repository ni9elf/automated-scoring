import torch
import torch.backends.cudnn as cudnn
import random

from transformers import logging

from utils.utils import add_params
from train.train_bert import train_bert
from train.train_meta_learning_via_bert_incontext_tuning import train_meta_learning_via_bert_incontext_tuning


# Disable warnings in hugging face logger
logging.set_verbosity_error()
# Set your Neptune token for logging
NEPTUNE_API_TOKEN = "SET_TOKEN"


def main():
    args = add_params()

    # Local saved models dir
    saved_models_dir = "../../../saved_models/"
    # Set random seed 
    if args.seed != -1:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    if args.cuda: assert device.type == 'cuda', 'no gpu found!'
    # Logging to neptune
    run = None
    if args.neptune:
        import neptune.new as neptune
        run = neptune.init(
            project = args.neptune_project,
            api_token = NEPTUNE_API_TOKEN,
            capture_hardware_metrics = False,
            name = args.name,
            )  
        run["parameters"] = vars(args)

    if( args.amp ):
        # Using pytorch automatic mixed precision (fp16/fp32) for faster training
        # https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # Train-val-test model
    if( args.meta_learning ):
        train_meta_learning_via_bert_incontext_tuning(args, run, device, saved_models_dir, scaler)
    else:
        train_bert(args, run, device, saved_models_dir, scaler)


if __name__ == '__main__':
    main()