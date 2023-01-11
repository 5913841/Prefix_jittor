# 加载计图，以及其他通用包
import jittor as jt
import time
import json
from pathlib import Path
from transformers import BertTokenizerFast
from train_control import PrefixTuning
from bert_model import BertForQuestionAnswering, BertConfig
from jittor.optim import AdamW
from transformers import AutoConfig
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from arguments import ModelArguments, DataTrainingArguments
from transformers.training_args import TrainingArguments
from transformers import HfArgumentParser, PreTrainedTokenizer, AutoConfig, AutoTokenizer


parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased')

# 开启 GPU 加速
jt.flags.use_cuda = 1


def read_squad(path):
    path = Path(path)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)

    return contexts, questions, answers

train_contexts, train_questions, train_answers = read_squad(data_args.train_data_file)
val_contexts, val_questions, val_answers = read_squad(data_args.eval_data_file)

def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        # sometimes squad answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx-1:end_idx-1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
        elif context[start_idx-2:end_idx-2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters

add_end_idx(train_answers, train_contexts)
add_end_idx(val_answers, val_contexts)

# View the first data.
print(u"Context: \n%s\n" % train_contexts[0])
print(u"Question: \n%s\n" % train_questions[0])
print(u"Answer: %s" % train_answers[0]["text"])
print(train_answers[0])

# Choose how much data to train and test.
train_length = len(train_answers)
train_start, train_end = 0, 1000
train_contexts = train_contexts[train_start:train_end]
train_questions = train_questions[train_start:train_end]
train_answers = train_answers[train_start:train_end]

val_length = len(val_answers)
val_start, val_end = 0, 1000
val_contexts = val_contexts[val_start:val_end]
val_questions = val_questions[val_start:val_end]
val_answers = val_answers[val_start:val_end]

print("Train set length from %s to %s" % (train_length, train_end))
print("Validation set length from %s to %s" % (val_length, val_end))


# Tokenization
train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)

def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length

    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

add_token_positions(train_encodings, train_answers)
add_token_positions(val_encodings, val_answers)

class SquadDataset(jt.dataset.Dataset):
    def __init__(self, encodings, labels):
        super().__init__()
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: jt.array(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]["text"].lower()
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SquadDataset(train_encodings, train_answers).set_attrs(batch_size=training_args.per_device_train_batch_size, shuffle=True)
val_dataset = SquadDataset(val_encodings, val_answers).set_attrs(batch_size=1, shuffle=False)

# Loading the model
configuration = BertConfig(num_hidden_layers=24, hidden_size=1024, num_attention_heads=16, intermediate_size=4096)
# configuration = BertConfig.from_pretrained(model_args.model_name_or_path)
bert_model = BertForQuestionAnswering(configuration)
bert_model.load_state_dict(jt.load(model_args.model_name_or_path + "/bert-large-uncased-qa-jittor.pkl")) # load pre-trained model
# bert_model = BertForQuestionAnswering.from_pretrained(model_args.model_name_or_path, config = configuration, from_tf = False, cache_dir=model_args.cache_dir)
config_prefix = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
config_prefix.preseqlen = model_args.preseqlen
model = PrefixTuning(config_prefix)

# Loading the optimizer
optim = AdamW(model.parameters(), lr=training_args.learning_rate)

# Prediction function
def predict(inputs, outputs):
    answer_start_index = jt.argmax(outputs["start_logits"], dim=1)[0]
    answer_end_index = jt.argmax(outputs["end_logits"], dim=1)[0]
    predict_answer_tokens = [inputs["input_ids"][i][int(answer_start_index[i]) : int(answer_end_index[i]) + 1].numpy() for i in range(len(inputs["input_ids"]))]
    predictions = [tokenizer.decode(tokens) for tokens in predict_answer_tokens]
    return predictions

# Training

# Set the number of epoch
epoch = int(training_args.num_train_epochs)

# Start training
bert_model.eval()
model.train()

train_loss = list()
train_accuracies = list()
for epoch_i in range(epoch):
    print('Epoch %s/%s' % (epoch_i + 1, epoch))
    time.sleep(0.3)

    correct = 0
    count = 0
    epoch_loss = list()
    
    pbar = tqdm(train_dataset, total=len(train_dataset)//train_dataset.batch_size)
    steps = 0
    for batch in pbar:
        steps +=1
        optim.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        start_positions = batch['start_positions']
        end_positions = batch['end_positions']
        labels = batch['labels']
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions, bert_model = bert_model)
        loss = outputs['loss']
        optim.step(loss)
        
        # make predictions
        predictions = predict(batch, outputs)

        # count accuracy
        correct += np.sum(np.array(predictions) == np.array(labels))
        count += len(labels)
        accuracy = correct * 1.0 / count

        # show progress along with metrics
        pbar.set_postfix({
            'Loss': '{:.3f}'.format(loss.item()),
            'Accuracy': '{:.3f}'.format(accuracy)
        })
        
        # record the loss for each batch
        epoch_loss.append(loss.item())

        if steps > 0 and steps % training_args.save_steps == 0:
            model.save(training_args.output_dir+f'/checkpoint-epoch{epoch_i}-step{steps}.pkl')
        
    pbar.close()
    
    # record the loss and accuracy for each epoch
    train_loss += epoch_loss
    train_accuracies.append(accuracy)



# Plot Iteration vs Training Loss
plt.plot(train_loss, label="Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Iteration vs Training Loss")  
plt.legend()
plt.show()

# Plot Epoch vs Training Accuracy
acc_X = np.arange(len(train_accuracies))+1                          
plt.plot(acc_X, train_accuracies,"-", label="Training Accuracy")
plt.xticks(acc_X)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Epoch vs Training Accuracy")  
plt.legend()
plt.show()


# Testing

# Start testing
model.eval()

with jt.no_grad():
    
    correct = 0
    count = 0
    record = {"labels":list(), "predictions":list()}
    
    pbar = tqdm(val_dataset, total=len(val_dataset)//val_dataset.batch_size)
    for batch in pbar:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        start_positions = batch['start_positions']
        end_positions = batch['end_positions']
        labels = batch['labels']
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions, bert_model = bert_model)
        loss = outputs['loss']

        # make predictions
        predictions = predict(batch, outputs)

        # count accuracy
        correct += np.sum(np.array(predictions) == np.array(labels))
        count += len(labels)
        accuracy = correct * 1.0 / count

        # show progress along with metrics
        pbar.set_postfix({
            'loss': '{:.3f}'.format(loss.item()),
            'accuracy': '{:.3f}'.format(accuracy)
        })
    
        # record the results
        record["labels"] += labels
        record["predictions"] += predictions
        
    pbar.close()
    
time.sleep(0.3)
print("The final accuracy on the test dataset: %s%%" % round(accuracy*100,4))