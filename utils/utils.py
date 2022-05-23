import numpy as np
from sklearn.metrics import cohen_kappa_score
import argparse
import torch
import pathlib


def tonp(x):
    if isinstance(x, (np.ndarray, float, int)):
        return np.array(x)
    
    return x.detach().cpu().numpy()


def agg_all_metrics(outputs):
    # Aggregate metrics for entire epoch across all batches

    if( len(outputs) == 0 ):
        return outputs
    
    res = {}
    keys = [ k for k in outputs[0].keys() if not isinstance(outputs[0][k], dict) ]
    for k in keys:
        all_logs = np.concatenate([tonp(x[k]).reshape(-1) for x in outputs])
        if( k != 'epoch' ):
            res[k] = np.mean(all_logs)
        else:
            res[k] = all_logs[-1]

    if 'kappa' in outputs[0]:
        pred_logs =  np.concatenate([tonp(x['kappa']['preds']).reshape(-1) for x in outputs])
        label_logs = np.concatenate([tonp(x['kappa']['labels']).reshape(-1) for x in outputs])
        if( np.array_equal(pred_logs, label_logs) ):
            # Edge case: cohen_kappa_score from sklearn returns a value of NaN if perfect agreement
            res['kappa'] = 1
        else:
            res['kappa'] = cohen_kappa_score(pred_logs, label_logs, weights= 'quadratic')
    
    return res


def add_params():
    parser = argparse.ArgumentParser(description='automated_scoring')
    
    parser.add_argument('--name', default='automated_scoring', help='Name of the experiment')
    parser.add_argument('--neptune_project', default="user_name/project_name", help='Name of the neptune project')
    # Problem definition
    parser.add_argument('--lm', default='bert-base-uncased', help='Base language model (provide any Hugging face model name)')
    parser.add_argument('--task', default="item_name", help='Item name (not required for meta learning via in-context tuning)')
    # Add demographic information for fairness analysis - generative models don't have this option
    parser.add_argument('--demographic', action='store_true', help='Use demographic information of student')
    
    # Meta learning BERT via in-context tuning
    # At testing, batch_size = batch_size * num_test_avg, similarly for validation
    # Ensure increased batch_size at test/val can be loaded onto GPU
    parser.add_argument('--meta_learning', action='store_true', help='Enable meta-learning via BERT in-context tuning')
    parser.add_argument('--num_test_avg', default=8, type=int, help='Number of different sets of randomly sampled examples per test datapoint to average score predictions')
    parser.add_argument('--num_val_avg', default=8, type=int, help='Number of different sets of randomly sampled examples per val datapoint to average score predictions')
    parser.add_argument('--num_examples', default=25, type=int, help='Number of in-context examples from each score class to add to input')
    parser.add_argument('--trunc_len', default=70, type=int, help='Max number of words in each in-context example')

    # Optimizer params
    parser.add_argument('--lr_schedule', default='warmup-const', help='Learning rate schedule to use')
    parser.add_argument('--opt', default='adam', choices=['sgd', 'adam', 'lars'], help='Optimizer to use')
    parser.add_argument('--iters', default=100, type=int, help='Number of epochs')
    parser.add_argument('--lr', default=2e-5, type=float, help='Base learning rate')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')

    # Data loading
    parser.add_argument('--data_folder', default="data_split_answer_spell_checked_submission", help='Dataset folder name containing train-val-test splits for each cross validation fold')
    parser.add_argument('--cross_val_fold', default=1, type=int, help='Cross validation fold to use')

    # Extras
    parser.add_argument('--save_freq', default=1, type=int, help='Epoch frequency to save the model')
    parser.add_argument('--eval_freq', default=1, type=int, help='Epoch frequency for evaluation')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loader workers')
    parser.add_argument('--seed', default=999, type=int, help='Random seed')   
    parser.add_argument('--cuda', action='store_true', help='Use cuda')
    parser.add_argument('--save', action='store_true', help='Save model every save_freq epochs')
    parser.add_argument('--neptune', action='store_true', help='Enable logging to Neptune')
    parser.add_argument('--debug', action='store_true', help='Debug mode with less items and smaller datasets')
    # Automatic mixed precision training -> faster training but might affect accuracy 
    parser.add_argument('--amp', action='store_true', help='Apply automatic mixed precision training')

    params = parser.parse_args()
    
    return params


def save_model(dir_model, model):
    pathlib.Path(dir_model).mkdir(parents=True, exist_ok=True)
    model.tokenizer.save_pretrained(dir_model)
    if( torch.cuda.device_count() > 1 ):
        # For an nn.DataParallel object, the model is stored in .module
        model.model.module.save_pretrained(dir_model)
    else:
        model.model.save_pretrained(dir_model)  