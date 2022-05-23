import torch
from torch import nn

from models.bert import LanguageModelBase
from utils.batch_collator import CollateWraperInContextTuningMetaLearning
from utils.load_data import load_dataset_in_context_tuning_with_meta_learning


class MetaLearningViaLanguageModelInContextTuning(LanguageModelBase):
    def __init__(self, params, device, task_list, task_to_question, task_to_passage, passages):
        super().__init__(params, device)
        self.task_list = task_list
        self.task_to_question = task_to_question
        self.task_to_passage = task_to_passage
        self.passages = passages


    def prepare_data(self):
        self.data_meta, _, _ = load_dataset_in_context_tuning_with_meta_learning(self.params.debug, self.params.data_folder, 
                            self.task_list, self.params.cross_val_fold)
        self.trainset = self.data_meta['train']
        self.validsets = {}
        self.testsets = {}
        for task in self.task_list:
            self.validsets[task] = self.data_meta[task]['valid']
            self.testsets[task] = self.data_meta[task]['test']
        self.test_batch_size = 12
        self.val_batch_size = 12

        # For meta-trained model, min_label=1 and max_label=4 is the same for all items since a fixed classification layer is used for all items
        self.min_label = 1
        self.max_label = 4
        self.num_labels = self.max_label - self.min_label + 1


    def dataloaders(self):
        collate_fn_train = CollateWraperInContextTuningMetaLearning(self.tokenizer, self.data_meta, self.task_to_question, 
                                                                    self.params.num_examples, 
                                                                    self.params.trunc_len, mode="train", 
                                                                    use_demographic = self.params.demographic)
        collate_fn_val = CollateWraperInContextTuningMetaLearning(self.tokenizer, self.data_meta, self.task_to_question, 
                                                                    self.params.num_examples, 
                                                                    self.params.trunc_len, mode="val", num_val_avg=self.params.num_val_avg,
                                                                    val_batch_size=self.val_batch_size, 
                                                                    use_demographic = self.params.demographic)
        collate_fn_test = CollateWraperInContextTuningMetaLearning(self.tokenizer, self.data_meta, self.task_to_question, 
                                                            self.params.num_examples, 
                                                            self.params.trunc_len, mode="test", num_test_avg=self.params.num_test_avg,
                                                            test_batch_size=self.test_batch_size, 
                                                            use_demographic = self.params.demographic)
        
        train_loader = torch.utils.data.DataLoader(self.trainset, collate_fn=collate_fn_train, batch_size=self.params.batch_size, 
                                                    num_workers=self.params.workers, shuffle=True, drop_last=False)
        valid_loaders = {}
        test_loaders = {}
        for task in self.task_list:
            # For validation, batch_size after collating = batch_size * num_val_avg 
            valid_loaders[task] = torch.utils.data.DataLoader(self.validsets[task], collate_fn=collate_fn_val, 
                                                batch_size=self.val_batch_size, num_workers=self.params.workers, 
                                                shuffle=False, drop_last=False)
            # For testing, batch_size after collating = batch_size * num_test_avg 
            test_loaders[task] = torch.utils.data.DataLoader(self.testsets[task], collate_fn=collate_fn_test, 
                                                batch_size=self.test_batch_size, num_workers=self.params.workers, 
                                                shuffle=False, drop_last=False)

        return train_loader, valid_loaders, test_loaders


    def train_step(self, batch, scaler, loss_func):
        self.zero_grad()
        
        if( self.params.amp ):
            # Cast operations to mixed precision
            with torch.cuda.amp.autocast():
                outputs = self.model(**batch["inputs"])
        else:
            outputs = self.model(**batch["inputs"])

        logits = outputs.logits
        
        # Mask invalid score classes as negative infinity
        # https://stackoverflow.com/questions/57548180/filling-torch-tensor-with-zeros-after-certain-index
        mask = torch.zeros(logits.shape[0], logits.shape[1] + 1, dtype=logits.dtype, device=logits.device)
        mask[(torch.arange(logits.shape[0]), batch["max_labels"])] = 1
        mask = mask.cumsum(dim=1)[:, :-1]
        masked_logits = logits.masked_fill_(mask.eq(1), value=float('-inf')) 

        # Calculate masked cross entropy loss
        loss = loss_func(masked_logits.view(-1, self.num_labels), batch["inputs"]["labels"].view(-1))

        # Apply a softmax over valid score classes only
        softmax_outs = nn.functional.softmax(masked_logits, dim=-1)

        # Calculate accuracy
        predictions = torch.argmax(softmax_outs, dim=-1)
        acc = ( predictions == batch["inputs"]["labels"] )

        # Multi gpu mode
        if( torch.cuda.device_count() > 1 ):
            if( self.params.amp ):
                scaler.scale(loss.sum()).backward()
            else:
                loss.sum().backward()
        else:
            if( self.params.amp ):
                scaler.scale(loss).backward()
            else:
                loss.backward()
        
        self.grad_step(scaler)

        if( self.params.amp ):
            scaler.update()
        
        return {'loss': loss.detach().cpu(),
                'acc':acc.detach().cpu(),
                'kappa':{
                    'preds':predictions.detach().cpu(), 
                    'labels':batch["inputs"]["labels"].detach().cpu()
                    }
                }


    def eval_step(self, batch):
        # Validation time averaging: Apply different sets of randomly sampled in-context examples per val datapoint to average score predictions

        with torch.no_grad():
            if( self.params.amp ):
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch["inputs"])
            else:
                outputs = self.model(**batch["inputs"])


        loss = outputs.loss
        # Dimension of logits = batch_size X num_classes
        # Where batch_size = val_batch_size * num_val_avg
        logits = outputs.logits

        # Mask invalid score classes as negative infinity
        # https://stackoverflow.com/questions/57548180/filling-torch-tensor-with-zeros-after-certain-index
        mask = torch.zeros(logits.shape[0], logits.shape[1] + 1, dtype=logits.dtype, device=logits.device)
        mask[(torch.arange(logits.shape[0]), batch["max_labels"])] = 1
        mask = mask.cumsum(dim=1)[:, :-1]
        masked_logits = logits.masked_fill_(mask.eq(1), value=float('-inf')) 

        # Apply a softmax over valid score classes only
        softmax_outs = nn.functional.softmax(masked_logits, dim=-1)

        # Reshaped dimension of softmax_outs = val_batch_size X num_val_avg X num_classes
        softmax_outs = torch.reshape(softmax_outs, (batch["actual_batch_size"], self.params.num_val_avg, -1))

        # Mean averaging on softmax_outs across val_samples
        # Dimension of outs = val_batch_size X num_classes
        outs = torch.mean(softmax_outs, dim=1)
        # Dimension of predictions = test_batch_size X 1
        predictions = torch.argmax(outs, dim=-1)

        # Pick every num_val_avg label since labels are repeated in batch
        batch["inputs"]["labels"] = batch["inputs"]["labels"][::self.params.num_val_avg]
        # Calculate accuracy
        acc = (predictions == batch["inputs"]["labels"])

        return {
            'loss': loss.detach().cpu(),
            'acc':acc.detach().cpu(),
            'kappa':{
                'preds':predictions.detach().cpu(), 
                'labels':batch["inputs"]["labels"].detach().cpu()
                }
            }


    def test_step(self, batch):
        # Test time averaging: Apply different sets of randomly sampled in-context examples per test datapoint to average score predictions

        with torch.no_grad():
            if( self.params.amp ):
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch["inputs"])
            else:
                outputs = self.model(**batch["inputs"])

        loss = outputs.loss
        # Dimension of logits = batch_size X num_classes
        # Where batch_size = test_batch_size * num_test_avg
        logits = outputs.logits

        # Mask invalid score classes as negative infinity
        # https://stackoverflow.com/questions/57548180/filling-torch-tensor-with-zeros-after-certain-index
        mask = torch.zeros(logits.shape[0], logits.shape[1] + 1, dtype=logits.dtype, device=logits.device)
        mask[(torch.arange(logits.shape[0]), batch["max_labels"])] = 1
        mask = mask.cumsum(dim=1)[:, :-1]
        masked_logits = logits.masked_fill_(mask.eq(1), value=float('-inf')) 

        # Apply a softmax over valid score classes only
        softmax_outs = nn.functional.softmax(masked_logits, dim=-1)

        # Reshaped dimension of softmax_outs = test_batch_size X num_test_avg X num_classes
        softmax_outs = torch.reshape(softmax_outs, (batch["actual_batch_size"], self.params.num_test_avg, -1))

        # Mean averaging on softmax_outs across test_samples
        # Dimension of outs = test_batch_size X num_classes
        outs = torch.mean(softmax_outs, dim=1)
        # Dimension of predictions = test_batch_size X 1
        predictions = torch.argmax(outs, dim=-1)

        # Pick every num_test_avg label since labels are repeated in batch
        batch["inputs"]["labels"] = batch["inputs"]["labels"][::self.params.num_test_avg]
        # Calculate accuracy
        acc = (predictions == batch["inputs"]["labels"])

        return {
            'loss': loss.detach().cpu(),
            'acc':acc.detach().cpu(),
            'kappa':{
                'preds':predictions.detach().cpu(), 
                'labels':batch["inputs"]["labels"].detach().cpu()
                }
            }