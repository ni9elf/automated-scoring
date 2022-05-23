import torch
import random


def tokenize_function(tokenizer, sentences_1, sentences_2=None):
    if(sentences_2 == None):
        return tokenizer(sentences_1, padding=True, truncation=True, return_tensors="pt")
    else:
        return tokenizer(sentences_1, sentences_2, padding=True, truncation=True, return_tensors="pt")


class CollateWraperParent(object):
    def __init__(self, tokenizer, min_label):
        self.tokenizer = tokenizer
        self.min_label = min_label


class CollateWraper(CollateWraperParent):
    # batch collator for BERT fine-tuning
    def __init__(self, tokenizer, min_label):
        super().__init__(tokenizer, min_label)
    
    def __call__(self, batch):
        # Construct features
        features = [d['txt'] for d in batch]
        inputs = tokenize_function(self.tokenizer, features)

        # Construct labels
        labels  =  torch.tensor([d['l1'] if d['l1']>=0 else d['l2'] for d in batch]).long() - self.min_label
        inputs['labels'] = labels

        return {"inputs" : inputs}


class CollateWraperInContextTuningMetaLearning(CollateWraperParent):
    # batch collator for meta learning via BERT in-context tuning
    def __init__(self, tokenizer, data_meta, task_to_question, num_examples, trunc_len, mode, num_test_avg=8,  
                num_val_avg=8, test_batch_size=1, val_batch_size=1, max_seq_len=512, use_demographic=False):
        super().__init__(tokenizer, min_label = 1)  
        
        self.data_meta = data_meta
        self.num_examples = num_examples
        self.trunc_len = trunc_len
        self.task_to_question = task_to_question
        self.mode = mode
        # Adding an extra 50 words in case num_tokens < num_words after tokenization
        self.max_seq_len = max_seq_len + 50
        
        # Convert numeric scores to meaningful words
        self.label_to_text = {
            1 : "poor",
            2 : "fair",
            3 : "good",
            4 : "excellent"
        }

        # Demographic information
        self.use_demographic = use_demographic
        self.gender_map = {
            "1" : "male",
            "2" : "female"
        }
        self.race_map = {
            "1" : "white",
            "2" : "african american",
            "3" : "hispanic",
            "4" : "asian",
            "5" : "american indian",
            "6" : "pacific islander",
            "7" : "multiracial"
        }
        
        # Meta learning via BERT in-context tuning        
        self.num_test_avg = num_test_avg
        self.num_val_avg = num_val_avg
        self.test_batch_size = test_batch_size
        self.val_batch_size = val_batch_size


    def __call__(self, batch):
        if( self.mode == "test" or self.mode == "val" ):
            # Since drop_last=False in test/val loader, record actual test_batch_size/val_batch_size for last batch constructed
            actual_batch_size = torch.tensor(len(batch)).long()

            # Repeat each test/val sample num_test_avg/num_val_avg times sequentially
            new_batch = []
            for d in batch:
                if( self.mode == "test" ):
                    new_batch += [d for _ in range(self.num_test_avg)]
                else:
                    new_batch += [d for _ in range(self.num_val_avg)]
            batch = new_batch
        else:
            actual_batch_size = torch.tensor(-1).long()
        
        # Construct features: features_1 (answer txt) will have different segment embeddings than features_2 (remaining txt)
        features_1 = []
        features_2 = []
        for d in batch:
            # Randomly sample num_examples in-context examples from each class in train set for datapoint d
            examples_many_per_class = []
            # List examples_each_class stores one example from each class
            examples_one_per_class = []
            labels = list(range(d["min"], d["max"] + 1))
            
            for label in labels:
                examples_class = self.data_meta[d["task"]]["examples"][label]
                
                # Remove current datapoint d from examples_class by checking unique booklet identifiers => no information leakage
                examples_class = [ex for ex in examples_class if ex["bl"] != d["bl"]]
                
                # Sampling num_examples without replacement
                if( len(examples_class) < self.num_examples ):
                    random.shuffle(examples_class)
                    examples_class_d = examples_class
                else:
                    examples_class_d = random.sample(examples_class, self.num_examples)

                if( len(examples_class_d) > 1 ):
                    examples_one_per_class += [examples_class_d[0]]
                    examples_many_per_class += examples_class_d[1:]
                elif( len(examples_class_d) == 1 ):
                    examples_one_per_class += [examples_class_d[0]]
                    examples_many_per_class += []
                else:
                    examples_one_per_class += []
                    examples_many_per_class += []
            
            # Construct input text with task instructions
            if( self.use_demographic ):
                input_txt = "score this answer written by {} {} student: ".format(self.gender_map[d["sx"]], self.race_map[d["rc"]]) + d['txt']
            else:
                input_txt = "score this answer: " + d['txt']
            features_1.append(input_txt)
            
            # Add range of valid score classes for datapoint d
            examples_txt = " scores: " + " ".join([ (self.label_to_text[label] + " ") for label in range(d["min"], d["max"] + 1) ])
            # Add question text
            examples_txt += "[SEP] question: {} [SEP] ".format(self.task_to_question[d["task"]])
            
            # Shuffle examples across classes
            random.shuffle(examples_one_per_class)
            random.shuffle(examples_many_per_class)
            
            # Since truncation might occur if text length exceed max input length to LM, 
            # we ensure at least one example from each score class is present 
            examples_d = examples_one_per_class + examples_many_per_class
            curr_len = len(input_txt.split(" ") + examples_txt.split(" "))
            for i in range(len(examples_d)):
                example = examples_d[i]
                example_txt_tokens = example['txt'].split(" ")
                curr_example_len = len(example_txt_tokens)
                example_txt = " ".join(example_txt_tokens[:self.trunc_len])
                example_label = (example['l1'] if example['l1']>=0 else example['l2'])
                # [SEP] at the end of the last example is automatically added by tokenizer
                if( i == (len(examples_d)-1) ):
                    if( self.use_demographic ):
                        examples_txt += ( " example written by {} {} student: ".format(self.gender_map[example["sx"]], self.race_map[example["rc"]]) + example_txt + " score: " + self.label_to_text[example_label] )
                    else:
                        examples_txt += ( " example: " + example_txt + " score: " + self.label_to_text[example_label] )
                else:
                    if( self.use_demographic ):
                        examples_txt += ( " example written by {} {} student: ".format(self.gender_map[example["sx"]], self.race_map[example["rc"]]) + example_txt + " score: " + self.label_to_text[example_label] + " [SEP] " )
                    else:
                        examples_txt += ( " example: " + example_txt + " score: " + self.label_to_text[example_label] + " [SEP] " )
                
                # Stop adding in-context examples when max_seq_len is reached
                if( (curr_example_len + curr_len) > self.max_seq_len):
                    break
                else:
                    curr_len += curr_example_len
            features_2.append(examples_txt)
        
        inputs = tokenize_function(self.tokenizer, features_1, features_2)
        
        # Construct labels
        labels  =  torch.tensor([ (d['l1']-d["min"]) if d['l1']>=0 else (d['l2']-d["min"]) for d in batch]).long()
        inputs['labels'] = labels
        
        # Store max_label for each d in batch which is used during softmax masking 
        max_labels = torch.tensor([( d["max"]-d["min"]+1) for d in batch]).long()

        return {
                "inputs" : inputs, 
                "max_labels" : max_labels, 
                "actual_batch_size" : actual_batch_size
            }