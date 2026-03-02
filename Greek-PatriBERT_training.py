import os
import math
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed
)
from datasets import load_dataset

# ------------------------------------------------------------------------------
# CONFIGURATION & HYPERPARAMETERS
# ------------------------------------------------------------------------------
MODEL_NAME = "pranaydeeps/Ancient-Greek-BERT"

# Path to your 300MB text file
# Ensure this file contains the pre-processed (de-accented, lower-cased) text
TRAIN_FILE = "processed_corpus/greek_corpus_unaccented.txt"  
VALIDATION_FILE = None 

OUTPUT_DIR = "./greek-bert-adapted"

# Optimization Parameters
# NRI tasks benefit from long contexts. We maximize the BERT limit.
MAX_SEQ_LENGTH = 512 

# 512 tokens consume ~4x more memory than 128. 
# We lower the batch size and increase accumulation to compensate.
TRAIN_BATCH_SIZE = 4       # Lower to 4 if you hit CUDA OOM on smaller GPUs
GRADIENT_ACCUMULATION = 8   # 8 * 8 = 64 effective batch size
LEARNING_RATE = 2e-5        # Lower learning rate to preserve pre-trained knowledge
NUM_EPOCHS = 5              # 300MB is relatively small; 5 epochs ensures convergence
WARMUP_RATIO = 0.06
MLM_PROBABILITY = 0.15
SEED = 42

def main():
    set_seed(SEED)
    
    # --------------------------------------------------------------------------
    # 1. LOAD MODEL & TOKENIZER
    # --------------------------------------------------------------------------
    print(f"Loading model: {MODEL_NAME}...")
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.max_position_embeddings = MAX_SEQ_LENGTH
    
    # Use_fast=True is essential for efficient processing of large corpora
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    
    # Note: Since your corpus is already de-accented and lower-cased,
    # we rely on the corpus preprocessing. However, we ensure the model 
    # receives what it expects (uncased usually).
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, config=config)

    # --------------------------------------------------------------------------
    # 2. PREPARE DATASET
    # --------------------------------------------------------------------------
    print("Loading dataset...")
    data_files = {"train": TRAIN_FILE}
    if VALIDATION_FILE:
        data_files["validation"] = VALIDATION_FILE
        dataset = load_dataset("text", data_files=data_files)
    else:
        full_dataset = load_dataset("text", data_files={"train": TRAIN_FILE})
        # 10% validation split to monitor perplexity
        dataset = full_dataset["train"].train_test_split(test_size=0.1)
        dataset["validation"] = dataset.pop("test")

    def tokenize_function(examples):
        return tokenizer(examples["text"], return_special_tokens_mask=True)

    print("Tokenizing...")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=["text"],
        desc="Tokenizing dataset",
    )

    # --------------------------------------------------------------------------
    # 3. CHUNKING (Handling Long Documents > 510 tokens)
    # --------------------------------------------------------------------------
    # This logic is crucial for your "full documents" requirement.
    # Instead of truncating documents at 512, we concatenate them and then
    # split into 512 blocks. This allows the model to see cross-sentence 
    # context across the entire document length.
    
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # Drop the remainder if < MAX_SEQ_LENGTH
        if total_length >= MAX_SEQ_LENGTH:
            total_length = (total_length // MAX_SEQ_LENGTH) * MAX_SEQ_LENGTH
            
        result = {
            k: [t[i : i + MAX_SEQ_LENGTH] for i in range(0, total_length, MAX_SEQ_LENGTH)]
            for k, t in concatenated_examples.items()
        }
        return result

    print(f"Packing texts into chunks of {MAX_SEQ_LENGTH} tokens...")
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=os.cpu_count(),
        desc="Grouping texts",
    )

    print(f"Training chunks: {len(lm_datasets['train'])}")
    print(f"Validation chunks: {len(lm_datasets['validation'])}")

    # --------------------------------------------------------------------------
    # 4. TRAINING SETUP
    # --------------------------------------------------------------------------
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True, 
        mlm_probability=MLM_PROBABILITY
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        eval_strategy="steps",
        eval_steps=250,                  # More frequent eval for smaller corpus
        save_steps=500,
        save_total_limit=2,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        num_train_epochs=NUM_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        fp16=torch.cuda.is_available(),  # Critical for 512 seq length speed
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=50,
        report_to="none",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # --------------------------------------------------------------------------
    # 5. EXECUTE
    # --------------------------------------------------------------------------
    print("Starting domain adaptation...")
    train_result = trainer.train()
    
    print("Saving adapted model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    print("Evaluating final perplexity...")
    eval_metrics = trainer.evaluate()
    try:
        perplexity = math.exp(eval_metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    
    print(f"Perplexity: {perplexity:.2f}")
    print(f"Adapted model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()