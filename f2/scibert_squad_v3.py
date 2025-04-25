"""
Finetuning SciBERT on a custom SciQA dataset for Question Answering.
"""

import os
from huggingface_hub import login
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.metrics import f1_score

login()

# (Colab only) Code for mounting Google Drive was used during development.
# from google.colab import drive
# drive.mount('/content/gdrive')
# %cd gdrive/MyDrive
# !ls

import json

with open("Data/sciQA.json", "r") as f:
    dataset = json.load(f)

df = pd.DataFrame(dataset)

for col in df.columns:
    print(col)

for col in df.columns:
    print(col, df[col].isnull().sum())

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Testing set size: {len(test_df)}")

model_name = "allenai/scibert_scivocab_uncased"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_data(examples):
    inputs = tokenizer(
        examples["context"],
        examples["question"],
        truncation=True,
        padding="max_length",
        max_length=384,
        return_tensors="pt",
    )

    start_positions = []
    end_positions = []

    for i, context in enumerate(examples["context"]):
        start_idx = context.find(examples["answer"][i])
        if start_idx == -1:
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_positions.append(start_idx)
            end_positions.append(start_idx + len(examples["answer"][i]))

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

tokenized_datasets_train = Dataset.from_pandas(train_df).map(preprocess_data, batched=True)
tokenized_datasets_val = Dataset.from_pandas(val_df).map(preprocess_data, batched=True)
tokenized_datasets_test = Dataset.from_pandas(test_df).map(preprocess_data, batched=True)

"""
Model Preparation
"""

for param in model.bert.encoder.layer:
    param.requires_grad = False

model.bert.encoder.layer[-1].requires_grad = True
model.bert.encoder.layer[-2].requires_grad = True

def compute_metrics(pred):
    labels = pred.label_ids
    start_preds = pred.predictions[0].argmax(-1)
    end_preds = pred.predictions[1].argmax(-1)
    f1 = f1_score(labels, start_preds, average='weighted')
    return {"f1": f1}

training_args = TrainingArguments(
    output_dir="./scibert_finetuned_sciQA",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    gradient_accumulation_steps=4,
    fp16=True, 
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets_train,
    eval_dataset=tokenized_datasets_val,
    compute_metrics=compute_metrics
)

# Enable W&B (Weights and Biases) for experiment tracking
# os.environ["WANDB_DISABLED"] = "false"
# !pip install wandb
# import wandb
# wandb.init(project="scibert_finetuned_sciQA")

trainer.train()