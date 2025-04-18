from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoConfig
from datasets import Dataset
import evaluate
import pandas as pd
from src.data.load_financial_phrasebank import load_financial_phrasebank, map_labels
from src.models.custom_model import *

def prepare_dataset(file_path, delimiter="@"):
    """
    Load the FinancialPhraseBank dataset and convert it to a Hugging Face Dataset.
    
    Args:
        file_path (str): Path to the FinancialPhraseBank text file.
        delimiter (str): Delimiter used in the file.
    
    Returns:
        dict: A dictionary with "train" and "test" splits.
    """
    # Load data into a pandas DataFrame
    df = load_financial_phrasebank(file_path, delimiter)
    df = map_labels(df)

     # Rename 'label_id' to 'labels' so that Trainer sees the correct field.
    if "label_id" in df.columns:
        df = df.rename(columns={"label_id": "labels"})
    # drop the original "label" column.
    if "label" in df.columns:
        df = df.drop(columns=["label"])
    
    # Convert the pandas DataFrame to a Hugging Face Dataset.
    dataset = Dataset.from_pandas(df)

    # Shuffle the dataset for randomness and reproducibility
    dataset = dataset.shuffle(seed=42)

    # First, split into 80% training and 20% temporary set
    split1 = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split1["train"]
    temp_dataset = split1["test"]

    # Now split the temporary set equally into validation and test sets (i.e., 10% each of original data)
    split2 = temp_dataset.train_test_split(test_size=0.5, seed=42)
    validation_dataset = split2["train"]
    test_dataset = split2["test"]


    return {
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset
    }

def fine_tune_model(train_dataset, eval_dataset, model_name="distilbert/distilroberta-base"):
    """
    Fine-tune DistilRoBERTa for sentiment analysis using the provided training and evaluation datasets.

    This function performs the following steps:
      1. Loads a tokenizer and model from the pretrained model.
      2. Tokenizes the training and evaluation datasets.
      3. Defines training arguments such as number of epochs, batch size, and logging details.
      4. Initializes a Trainer object from the Hugging Face Transformers library.
      5. Fine-tunes the model using the Trainer.
    
    Args:
        train_dataset (Dataset): Hugging Face dataset for training.
        eval_dataset (Dataset): Hugging Face dataset for evaluation.
        model_name (str): Pretrained model name.
    
    Returns:
        Trainer: The Trainer instance after training.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = 3  # for negative, neutral, positive
    config._name_or_path = model_name  # make sure our custom model knows which checkpoint to use

    # Instantiate the custom model.
    model = CustomSentimentModel.from_pretrained(model_name, config=config)
    
    # Tokenize the dataset
    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)
    
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)
    
    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_eval.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    
    #  Define the training arguments.
    #    TrainingArguments is a configuration class that specifies details like:
    #      - output_dir: where model checkpoints and logs are saved.
    #      - num_train_epochs: how many passes through the training data.
    #      - per_device_train_batch_size: the number of training examples per GPU/CPU.
    #      - evaluation_strategy: when to evaluate the model (here at the end of each epoch).
    #      - logging_steps: how frequently to log training information.

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=50,
    )
    
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    # For F1, you can choose "macro" averaging (suitable for multiclass classification)
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"]
        return {"accuracy": accuracy, "f1": f1}
    
    # Initialize the Trainer.
    #    Trainer is a high-level API that handles the training loop, evaluation, saving, etc.

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    return trainer

# Example usage (for testing, run via a notebook or script):
if __name__ == "__main__":

    file_path = "../../data/processed/phraseBank/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt"  
    # Prepare the dataset (with shuffling and 80/10/10 split)
    splits = prepare_dataset(file_path)
    
    # Fine-tune the model using the training and validation sets
    trainer = fine_tune_model(splits["train"], splits["validation"])
    
    # Save the fine-tuned model
    trainer.save_model("./finetuned_distilroberta")
    
    # Evaluate on the held-out test set for final performance metrics
    test_results = trainer.evaluate(splits["test"])
    print("Test set evaluation:", test_results)
