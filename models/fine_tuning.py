import json
import time
import copy
import random
import optuna
import nlpaug.augmenter.word as naw
import torch
from collections import Counter
from torch.nn import CrossEntropyLoss
import numpy as np
from sklearn.model_selection import KFold
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset, DatasetDict
from seqeval.metrics import classification_report, f1_score, accuracy_score

# Overwrite trainer to handle class weights
class WeightedNERTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Flatten the logits and labels
        active_loss = labels.view(-1) != -100
        active_logits = logits.view(-1, model.config.num_labels)[active_loss]
        active_labels = labels.view(-1)[active_loss]

        loss_fct = CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(active_logits, active_labels)

        return (loss, outputs) if return_outputs else loss

# Creating list of labels and mapping them to IDs
labels = ["O", "B-EVENT", "I-EVENT", "B-TIME", "I-TIME", "B-LOCATION", "I-LOCATION"]
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}

def bio_conversion(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for entry in data:
        entry["ner_tags"] = [label2id[label] for label in entry["tags"]]
        del entry["tags"]
    return data

syn_aug = naw.SynonymAug(aug_src='wordnet', aug_p=0.1)
swap_aug = naw.RandomWordAug(action="swap", aug_p=0.05)

def augment_data(data, num_augments=1):
    augmented_data = []
    total_synswap_attempts = 0
    synswap_skipped = 0

    # Synonym and swap augmentation
    for entry in data:
        for _ in range(num_augments):
            total_synswap_attempts += 1
            aug_random = random.choice([syn_aug, swap_aug])
            sentence = " ".join(entry["tokens"])
            try:
                augmented = aug_random.augment(sentence, n=1)
                aug_sentence = augmented[0] if isinstance(augmented, list) else augmented
                aug_tokens = aug_sentence.split()
                if len(aug_tokens) == len(entry["ner_tags"]):
                    augmented_data.append({"tokens": aug_tokens, "ner_tags": entry["ner_tags"]})
                else:
                    synswap_skipped += 1
            except Exception:
                synswap_skipped += 1

    # Logging stats
    print(f"\n🔁 Augmentation Summary:")
    print(f"Synonym/Swap - Attempts: {total_synswap_attempts}, Skipped: {synswap_skipped}, Kept: {len(augmented_data)}")

    return data + augmented_data

train_val_data = bio_conversion("data/split/train_validation.json")
test_data = bio_conversion("data/split/test.json")

datasets = DatasetDict({
    "train/valid": Dataset.from_list(train_val_data),
    "test": Dataset.from_list(test_data)
})

model_name = "microsoft/MiniLM-L12-H384-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_and_align_labels(batch):
    tokenized_inputs = tokenizer(
        batch["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=128
    )

    all_labels = []
    for i, word_ids in enumerate(tokenized_inputs.word_ids(batch_index=i) for i in range(len(batch["tokens"]))):
        labels = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)
            elif word_idx != previous_word_idx:
                labels.append(batch["ner_tags"][i][word_idx])
            else:
                labels.append(-100)
            previous_word_idx = word_idx
        labels += [-100] * (len(tokenized_inputs["input_ids"][i]) - len(labels))
        all_labels.append(labels)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

tokenized_train_valid = datasets["train/valid"].map(tokenize_and_align_labels, batched=True, cache_file_name="cache/tokenized_train_valid.arrow")

def compute_metrics(p):
    predictions, labels_ids = p
    preds = predictions.argmax(-1)

    true_predictions = [[id2label[p] for (p, l) in zip(pred, label) if l != -100] for pred, label in zip(preds, labels_ids)]
    true_labels = [[id2label[l] for (p, l) in zip(pred, label) if l != -100] for pred, label in zip(preds, labels_ids)]

    return {
        "accuracy": accuracy_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }

def calculate_class_weights(dataset, num_labels):
    counts = Counter()
    total = 0
    for example in dataset:
        for label in example['ner_tags']:
            if label != -100:
                counts[label] += 1
                total += 1

    weights = [total / (counts[i] * num_labels) if counts[i] > 0 else 0.0 for i in range(num_labels)]
    weights = np.array(weights) * (len(weights) / np.sum(weights))
    return torch.tensor(weights, dtype=torch.float)

def optuna_objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 2e-4)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3)
    batch_size = trial.suggest_categorical("per_device_train_batch_size", [8, 16])
    num_epochs = trial.suggest_int("num_train_epochs", 5, 10)

    print(f"Trial with lr={learning_rate}, wd={weight_decay}, bs={batch_size}, epochs={num_epochs}")

    training_args = TrainingArguments(
        output_dir="optuna_outputs",
        eval_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_strategy="epoch",
        save_total_limit=1,
        seed=42,
        logging_dir="./logs",
        logging_steps=10,
        disable_tqdm=True,
    )

    avg_f1, _ = cross_validation(tokenized_train_valid, training_args=training_args)
    return avg_f1

def cross_validation(dataset_train_valid: Dataset, training_args, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []

    base_model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset_train_valid)):
        print(f"\n===== Fold {fold + 1} / {n_splits} =====")
        train_dict = dataset_train_valid.to_dict()
        train_fold = Dataset.from_dict({k: [v[i] for i in train_idx] for k, v in train_dict.items()})
        val_fold = Dataset.from_dict({k: [v[i] for i in val_idx] for k, v in train_dict.items()})

        cv_class_weights = calculate_class_weights(train_fold, num_labels=len(labels))
        print(f"Class weights: {cv_class_weights}")

        train_fold_list = [{k: v[i] for k, v in train_fold.to_dict().items()} for i in range(len(train_fold))]
        train_fold_augmented = augment_data(train_fold_list)
        train_fold = Dataset.from_list(train_fold_augmented)

        tokenized_train = train_fold.map(tokenize_and_align_labels, batched=True)
        tokenized_val = val_fold.map(tokenize_and_align_labels, batched=True)

        model = copy.deepcopy(base_model)

        trainer = WeightedNERTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=tokenizer,
            class_weights=cv_class_weights,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        trainer.train()
        metrics = trainer.evaluate()
        print(f"Fold {fold + 1} metrics: {metrics}")
        fold_metrics.append(metrics)

    avg_f1 = np.mean([m["eval_f1"] for m in fold_metrics])
    avg_acc = np.mean([m["eval_accuracy"] for m in fold_metrics])
    print(f"\n===== Cross-validation results =====")
    print(f"Average F1: {avg_f1:.4f}")
    print(f"Average Accuracy: {avg_acc:.4f}")

    return avg_f1, avg_acc

def train_final_model(dataset_train_valid: Dataset, dataset_test: Dataset, training_args, output_dir):
    print("\n===== Training final model on full train+valid data =====")

    # Optionally augment the entire train+valid dataset before training
    train_val_list = [{k: v[i] for k, v in dataset_train_valid.to_dict().items()} for i in range(len(dataset_train_valid))]  # convert to list
    train_val_augmented = augment_data(train_val_list, num_augments=1)
    full_train_dataset = Dataset.from_list(train_val_augmented)

    # Tokenize full train dataset and test dataset
    tokenized_train = full_train_dataset.map(tokenize_and_align_labels, batched=True)
    tokenized_test = dataset_test.map(tokenize_and_align_labels, batched=True)

    # Initialize fresh model
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    class_weights = calculate_class_weights(datasets["train/valid"], num_labels=len(labels))
    print(f"Class weights (Final Training): {class_weights}")

    # Trainer setup
    trainer = WeightedNERTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Train final model
    trainer.train()

    # Evaluate on test set
    print("\n===== Final evaluation on test set =====")
    predictions, labels_ids, _ = trainer.predict(tokenized_test)
    preds = predictions.argmax(-1)

    true_predictions = [
        [id2label[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(preds, labels_ids)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(preds, labels_ids)
    ]

    print(classification_report(true_labels, true_predictions))

    # Save the final model
    trainer.save_model(output_dir)

# 1. Run cross-validation + hyperparameter tuning (on train/valid set)
# study = optuna.create_study(direction="maximize")
# study.optimize(optuna_objective, n_trials=20)

timestamp = time.strftime("%Y%m%d-%H%M%S")
# trial_number = study.best_trial.number
output_dir = f"./{model_name}-{timestamp}"
# -trial{trial_number}" Whenever you trying using hyperparameters optimization, uncomment this line

# best_params = study.best_trial.params
# print("\n===== Best Hyperparameters from Optuna =====")
# for k, v in best_params.items():
#     print(f"{k}: {v}")
# best_training_args = TrainingArguments(
#     output_dir=output_dir,
#     eval_strategy="epoch",
#     learning_rate=best_params["learning_rate"],
#     per_device_train_batch_size=best_params["per_device_train_batch_size"],
#     per_device_eval_batch_size=best_params["per_device_train_batch_size"],
#     num_train_epochs=best_params["num_train_epochs"],
#     weight_decay=best_params["weight_decay"],
#     load_best_model_at_end=True,
#     metric_for_best_model="f1",
#     save_strategy="epoch",
#     save_total_limit=1,
#     seed=42,
#     logging_dir="./logs",
#     logging_steps=10,
# )
# avg_f1, avg_acc = cross_validation(tokenized_train_valid, training_args=training_args)

training_args = TrainingArguments(
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8, 
    per_device_eval_batch_size=8,
    num_train_epochs=8,
    weight_decay=0.001,
    metric_for_best_model="f1",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# 2. Train final model on full train/valid, evaluate on test
train_final_model(datasets["train/valid"], datasets["test"], training_args=training_args, output_dir=output_dir)
