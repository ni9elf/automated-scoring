import time
import json
from torch import nn
from tqdm import tqdm

from models.meta_learning_via_bert_incontext_tuning import MetaLearningViaLanguageModelInContextTuning
from utils.utils import agg_all_metrics, save_model



def train_meta_learning_via_bert_incontext_tuning(args, run, device, saved_models_dir, scaler):
    # Load list of tasks for meta learning
    with open("data/tasks.json", "r") as f:
        task_list = json.load(f)
        if( args.debug ):
            task_list = task_list[0:2]
    # Load task to question map for each task
    with open("data/task_to_question.json", "r") as f:
        task_to_question = json.load(f)
    # Load task to passage map for each task
    with open("data/task_to_passage.json", "r") as f:
        task_to_passage = json.load(f)
    # Load passage texts
    with open("data/passages.json", "r") as f:
        passages = json.load(f)
    
    # Prepare data and model for training
    model = MetaLearningViaLanguageModelInContextTuning(args, device, task_list, task_to_question, task_to_passage, passages)
    model.prepare_data()    
    model.prepare_model()

    # Dict of metric variables = kappa for each task trained on during meta learning
    metrics = {}
    for task in task_list:
        metrics[task] = {}
        # Best test kappa
        metrics[task]["best_test_metric"] = -1
        # Test kappa corresponding to best validation kappa
        metrics[task]["test_metric_for_best_valid_metric"] = -1
        # Best validation kappa
        metrics[task]["best_valid_metric"] = -1

    # Train-val-test loop
    loss_func = nn.CrossEntropyLoss()
    for cur_iter in tqdm(range(args.iters)):
        train_loader, valid_loaders, test_loaders = model.dataloaders()
        
        # Train epoch on one big train dataset = union of train datasets across items for meta learning
        start_time = time.time()
        # Set model to train mode needed if dropout, etc is used
        model.train()
        train_logs = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logs = model.train_step(batch, scaler, loss_func)  
            train_logs.append(logs)
        train_it_time = time.time() - start_time

        # Aggregate logs across all batches
        train_logs = agg_all_metrics(train_logs)
        # Log to neptune
        if args.neptune:
            run["metrics/train/accuracy"].log(train_logs['acc'])
            run["metrics/train/kappa"].log(train_logs['kappa'])
            run["metrics/train/loss"].log(train_logs['loss'])
            run["logs/train/it_time"].log(train_it_time)

        # Set model to test mode needed if dropout, etc is used
        model.eval()
        if( (cur_iter % args.eval_freq == 0) or (cur_iter >= args.iters) ):
            # Dict of validation and test logs for all items
            test_logs, valid_logs = {}, {}
            for task in task_list:
                test_logs[task], valid_logs[task] = [], []
            
            # Validation epoch for each item
            eval_start_time = time.time()
            for task in task_list:
                valid_loader = valid_loaders[task]
                for batch in valid_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    logs = model.eval_step(batch)
                    valid_logs[task].append(logs)
            eval_it_time = time.time()-eval_start_time
            
            # Test epoch for each item
            test_start_time = time.time()
            for task in task_list:
                test_loader = test_loaders[task]
                for batch in test_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    logs = model.test_step(batch)
                    test_logs[task].append(logs)
            test_it_time = time.time()-test_start_time
            
            # Aggregate logs across batches and and across items
            for task in task_list:
                valid_logs[task] = agg_all_metrics(valid_logs[task])
                test_logs[task] = agg_all_metrics(test_logs[task])
            
            for task in task_list:
                # Update metrics 
                if( len(test_logs[task]) > 0 ):
                    metrics[task]["best_test_metric"] =  max(test_logs[task]['kappa'], metrics[task]["best_test_metric"])
                if( len(valid_logs[task]) > 0 ):
                    if( float(valid_logs[task]["kappa"]) > metrics[task]["best_valid_metric"] ):
                        metrics[task]["best_valid_metric"] = valid_logs[task]["kappa"]
                        if( len(test_logs[task]) > 0 ):
                            metrics[task]["test_metric_for_best_valid_metric"] =  float(test_logs[task]["kappa"])
                
                # Log to neptune for all items
                if args.neptune:
                    if( len(test_logs[task]) > 0 ):
                        run["metrics/{}/test/accuracy".format(task)].log(test_logs[task]['acc'])
                        run["metrics/{}/test/kappa".format(task)].log(test_logs[task]['kappa'])
                        run["metrics/{}/test/loss".format(task)].log(test_logs[task]['loss'])
                        run["metrics/{}/test/best_kappa".format(task)].log(metrics[task]["best_test_metric"])
                        run["metrics/{}/test/best_kappa_with_valid".format(task)].log(metrics[task]["test_metric_for_best_valid_metric"])
                    if( len(valid_logs[task]) > 0 ):
                        run["metrics/{}/valid/accuracy".format(task)].log(valid_logs[task]['acc'])
                        run["metrics/{}/valid/kappa".format(task)].log(valid_logs[task]['kappa'])
                        run["metrics/{}/valid/loss".format(task)].log(valid_logs[task]['loss'])
                        run["metrics/{}/valid/best_kappa".format(task)].log(metrics[task]["best_valid_metric"])
                    run["logs/cur_iter"].log(cur_iter)
                    run["logs/valid/it_time"].log(eval_it_time)
                    run["logs/test/it_time"].log(test_it_time)
        
        # Save model after every epoch irrespective of save_freq param
        dir_model = saved_models_dir + args.name + "/" + run.get_run_url().split("/")[-1] + "/" + args.task + "/" + "cross_val_fold_{}".format(args.cross_val_fold) + "/" + "/epoch_{}/".format(cur_iter)
        save_model(dir_model, model)