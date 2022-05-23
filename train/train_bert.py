import time
from tqdm import tqdm

from models.bert import LanguageModelBase
from utils.utils import agg_all_metrics, save_model


def train_bert(args, run, device, saved_models_dir, scaler):
    # prepare data and model for training
    model = LanguageModelBase(args, device=device)
    model.prepare_data()
    model.prepare_model()

    # Log item info to neptune
    if args.neptune:
        run["parameters/n_labels"] = model.max_label - model.min_label + 1

    # Metric used is Quadratic Weighted Kappa (QWK)
    # Best test kappa
    best_test_metric = -1 
    # Test kappa corresponding to best validation kappa
    test_metric_for_best_valid_metric = -1 
    # Best validation kappa
    best_valid_metric = -1 


    # Train-val-test loop
    for cur_iter in tqdm(range(args.iters)):
        train_loader, valid_loader, test_loader = model.dataloaders()
        
        # Train epoch
        start_time = time.time()
        # Set model to train mode needed if dropout, etc is used
        model.train()
        train_logs = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logs = model.train_step(batch, scaler)  
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
            test_logs, valid_logs = [], []
            # Validation epoch
            eval_start_time = time.time()
            for batch in valid_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logs = model.eval_step(batch)
                valid_logs.append(logs)
            eval_it_time = time.time()-eval_start_time
            
            # Test epoch
            test_start_time = time.time()
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logs = model.test_step(batch)
                test_logs.append(logs)
            test_it_time = time.time()-test_start_time
            
            # Aggregate logs across batches
            valid_logs = agg_all_metrics(valid_logs)
            test_logs = agg_all_metrics(test_logs)
            
            # Update metrics and save model
            if( float(test_logs['kappa']) > best_test_metric ):
                best_test_metric = float(test_logs['kappa'])
                # Save model with best test kappa (not based on validation set)
                dir_best_test_metric = saved_models_dir + args.name + "/" + run.get_run_url().split("/")[-1] + "/" + args.task + "/" + "/best_test_kappa/"
                save_model(dir_best_test_metric, model)
            if( float(valid_logs['kappa']) > best_valid_metric ):
                best_valid_metric = float(valid_logs['kappa'])
                test_metric_for_best_valid_metric =  float(test_logs['kappa'])
                # Save model with best validation kappa
                dir_best_valid_metric = saved_models_dir + args.name + "/" + run.get_run_url().split("/")[-1] + "/" + args.task + "/" + "/best_valid_kappa/"
                save_model(dir_best_valid_metric, model)
            
            # Push logs to neptune
            if args.neptune:
                run["metrics/test/accuracy"].log(test_logs['acc'])
                run["metrics/test/kappa"].log(test_logs['kappa'])
                run["metrics/test/loss"].log(test_logs['loss'])
                run["metrics/valid/accuracy"].log(valid_logs['acc'])
                run["metrics/valid/kappa"].log(valid_logs['kappa'])
                run["metrics/valid/loss"].log(valid_logs['loss'])
                run["metrics/test/best_kappa"].log(best_test_metric)
                run["metrics/test/best_kappa_with_valid"].log(test_metric_for_best_valid_metric)
                run["logs/cur_iter"].log(cur_iter)
                run["logs/valid/it_time"].log(eval_it_time)
                run["logs/test/it_time"].log(test_it_time)
        
        # Save model every save_freq epochs
        if( (cur_iter % args.save_freq == 0) or (cur_iter >= args.iters) ):
            dir_model = saved_models_dir + args.name + "/" + run.get_run_url().split("/")[-1] + "/" + args.task + "/" + "cross_val_fold_{}".format(args.cross_val_fold) + "/" + "/epoch_{}/".format(cur_iter)
            save_model(dir_model, model)
        