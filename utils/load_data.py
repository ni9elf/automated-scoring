import json
from collections import defaultdict
import os


RAW_DIR = "../../data/NAEP_AS_Challenge_Data/Items for Item-Specific Models/"


def compute_distribution(output):
    append_keys = {}
    
    for k, v in output.items():
        dist, new_key, total_count  = defaultdict(float), k+'_dist', 0.
        for d in v:
            n_rating =  (d['l1']>=0) + (d['l2']>=0)
            total_count +=n_rating
            if( d['l1'] != -1 ): 
                dist[d['l1']] += 1./n_rating
            if( d['l2'] != -1): 
                dist[d['l2']] += 1./n_rating
        for l in dist:
            dist[l] /= total_count
        append_keys[new_key] = dist
    
    for k in append_keys:
        output[k] = append_keys[k]


def load_dataset_base(task, debug=False, data_folder="data_split_answer_spell_checked", cross_val_fold=1):
    """
    Returns a dictionary: 
        { 
            'train' : [{key:value}], # training dataset
            'val' : [{key:value}], # validation dataset
            'test' : [{key:value}], # test dataset
            'train_dist' : {label:percentage} # distribution of scores in train-set
            'val_dist' : {label:percentage} # distribution of scores in val-set
            'test_dist' : {label:percentage} # distribution of scores in test-set
        }

    Each of the train/val/test datasets are a list of samples. Each sample is a dictionary of following (key, value) pairs:
    {'bl':string, 'l1':int, 'l2':int, 'sx':string, 'rc':string, 'txt':string}  
    
    The keys above are:
    bl: unique database like key to identify student response
    'l1' : score by human rater 1
    'l2' : score by human rater 2 (set as -1 if not available)
    'sx' : sex of student (optional)
    'rc' : race of student
    'txt' : student response text to the reading comprehension item to be scored 
    """

    suffix = "_".join(data_folder.split("_")[1:])

    if( cross_val_fold == 0 ):
        dir_name = RAW_DIR + task + "/" + data_folder 
    else:
        # Use spell checked version and cross validation fold number
        dir_name = RAW_DIR + task + "/" + "cross_val_fold_{}".format(cross_val_fold) + "/" + data_folder 
    
    data = {}
    filenames = [("train", "train"), ("val", "valid"), ("test", "test")]
    for i in range(len(filenames)):
        filename = os.path.join(dir_name, task.split('/')[1] + "_{}_{}.json".format(filenames[i][0], suffix))
        with open(filename, "r") as f:
            data[filenames[i][1]] = json.load(f)

    # Compute score distribution
    compute_distribution(data)

    # Debug with less data if required
    if(debug):
        if( len(data["train"]) > 0 ):
            data["train"] = data["train"][:4]
        if( len(data["valid"]) > 0 ):
            data["valid"] = data["valid"][:4]
        if( len(data["test"]) > 0 ):
            data["test"] = data["test"][:4]

    return data


def load_dataset_in_context_tuning(task, debug=False, data_folder="data_split_answer_spell_checked", cross_val_fold=1):
    data = load_dataset_base(task, debug, data_folder, cross_val_fold)
    
    # Construct in_context examples from training dataset partitioned according to class score label
    examples_train = None
    max_label = max(data['train_dist'].keys())
    min_label = min(data['train_dist'].keys())
    examples_train = {}
    for label in range(min_label, max_label + 1):
        examples_train[label] = []
    for datapoint in data["train"]:
        label = datapoint['l1'] if datapoint['l1']>=0 else datapoint['l2']
        examples_train[label].append(datapoint)
    
    return data, examples_train, min_label, max_label


def load_dataset_in_context_tuning_with_meta_learning(debug=False, data_folder="data_split_answer_spell_checked", task_list=[], cross_val_fold=1):
    # Load list of item names
    with open("data/tasks.json", "r") as f:
        task_list = json.load(f)
        if( debug ):
            task_list = task_list[0:2]
    
    data_meta = {}
    for task in task_list:
        data_meta[task] = {}
        data, examples_train, min_label, max_label = load_dataset_in_context_tuning(task, debug, data_folder, cross_val_fold)
        data_meta[task]["train"] = data["train"]
        data_meta[task]["valid"] = data["valid"]
        data_meta[task]["test"] = data["test"]
        # Add in-context examples from training dataset => no information leakage from val/test sets
        data_meta[task]["examples"] = {}
        for label in range(min_label, max_label + 1):
            data_meta[task]["examples"][label] = examples_train[label]
        # Add task, min_label and max_label info to each sample
        for set in ["train", "valid", "test"]:
            for sample in data_meta[task][set]:
                sample["min"] = min_label
                sample["max"] = max_label
                sample["task"] = task
    
    # Union of training datasets across tasks
    data_meta["train"] = []
    for task in task_list:
        data_meta["train"] += data_meta[task]["train"]

    return data_meta, None, None