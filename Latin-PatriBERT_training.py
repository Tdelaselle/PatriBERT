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
# Model identifier from Hugging Face Hub
MODEL_NAME = "ashleygong03/bamman-burns-latin-bert"

# Convert legacy latin-bert vocab format to standard WordPiece format.
# If True, this script creates/uses a local converted tokenizer before training.
USE_CONVERTED_VOCAB = True
CONVERTED_TOKENIZER_DIR = "./tokenizer-converted-wordpiece"

# Path to your 600MB text file (one document per line)
TRAIN_FILE = "processed_corpus/latin_corpus.txt"  
# Optional: Path to a validation file. If None, we split the train file.
VALIDATION_FILE = None 

# Directory to save the adapted model
OUTPUT_DIR = "./latin-bert-adapted"

# Optimization Parameters
# The original model was trained with max_seq_length=256. 
# Increasing this blindly to 512 can degrade performance without massive retraining.
MAX_SEQ_LENGTH = 256 
TRAIN_BATCH_SIZE = 4        # Adjust based on your GPU VRAM (Try 8 if OOM, 32 if 24GB+ VRAM)
GRADIENT_ACCUMULATION = 4   # effective_batch_size = 16 * 4 = 64
LEARNING_RATE = 5e-5        # Standard for domain adaptation (lower than initial pretraining)
NUM_EPOCHS = 4              # 600MB is approx 3-5 epochs for convergence
WARMUP_RATIO = 0.05         # Warmup over 5% of training steps
MLM_PROBABILITY = 0.15      # Standard BERT masking rate
SEED = 42


def build_converted_wordpiece_tokenizer(model_name, output_dir):
    """Convert legacy trailing-underscore vocab to WordPiece-compatible vocab.

    Legacy convention:
      - word-initial / whole-word tokens end with "_" (e.g., "et_")
      - continuation tokens have no marker (e.g., "im")

    WordPiece convention:
      - word-initial / whole-word tokens are plain (e.g., "et")
      - continuation tokens are prefixed with "##" (e.g., "##im")
    """
    os.makedirs(output_dir, exist_ok=True)
    converted_vocab_path = os.path.join(output_dir, "vocab.txt")

    # Reuse existing converted vocab if present
    if os.path.exists(converted_vocab_path):
        print(f"Using existing converted tokenizer at {output_dir}")
        return output_dir

    print("Building converted WordPiece tokenizer...")
    base_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    vocab = base_tokenizer.get_vocab()  # token -> id

    special_tokens = {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}
    max_id = max(vocab.values())
    id_to_token = [None] * (max_id + 1)
    for token, token_id in vocab.items():
        id_to_token[token_id] = token

    converted_id_to_token = [None] * len(id_to_token)
    seen_tokens = set()
    collisions = 0

    for token_id, token in enumerate(id_to_token):
        if token in special_tokens:
            new_token = token
        elif token.endswith("_"):
            new_token = token[:-1]
        else:
            new_token = f"##{token}"

        if new_token in seen_tokens:
            collisions += 1
            fallback = f"[UNUSED_CONV_{token_id}]"
            while fallback in seen_tokens:
                fallback += "_"
            new_token = fallback

        converted_id_to_token[token_id] = new_token
        seen_tokens.add(new_token)

    with open(converted_vocab_path, "w", encoding="utf-8") as f:
        for token in converted_id_to_token:
            f.write(token + "\n")

    converted_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    converted_tokenizer.vocab = {tok: i for i, tok in enumerate(converted_id_to_token)}
    converted_tokenizer.ids_to_tokens = {i: tok for i, tok in enumerate(converted_id_to_token)}
    converted_tokenizer.save_pretrained(output_dir)

    print(f"Converted tokenizer saved to {output_dir}")
    print(f"Vocab size: {len(converted_id_to_token)} | Collisions handled: {collisions}")
    return output_dir

def main():
    set_seed(SEED)
    
    # --------------------------------------------------------------------------
    # 1. LOAD MODEL & TOKENIZER
    # --------------------------------------------------------------------------
    print(f"Loading model: {MODEL_NAME}...")
    config = AutoConfig.from_pretrained(MODEL_NAME)
    # Keep the original max_position_embeddings (512) to match checkpoint weights

    tokenizer_source = MODEL_NAME
    if USE_CONVERTED_VOCAB:
        tokenizer_source = build_converted_wordpiece_tokenizer(
            MODEL_NAME,
            CONVERTED_TOKENIZER_DIR,
        )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, config=config)

    if len(tokenizer) != config.vocab_size:
        raise ValueError(
            f"Tokenizer vocab size ({len(tokenizer)}) != model vocab size ({config.vocab_size}). "
            "Check conversion output before training."
        )

    # --------------------------------------------------------------------------
    # 2. PREPARE DATASET
    # --------------------------------------------------------------------------
    print("Loading and processing dataset...")
    
    # Load dataset from text file
    data_files = {"train": TRAIN_FILE}
    if VALIDATION_FILE:
        data_files["validation"] = VALIDATION_FILE
        dataset = load_dataset("text", data_files=data_files)
    else:
        # Create a 90/10 train/val split if no validation file provided
        full_dataset = load_dataset("text", data_files={"train": TRAIN_FILE})
        dataset = full_dataset["train"].train_test_split(test_size=0.1)
        dataset["validation"] = dataset.pop("test")

    # Tokenization function
    # Note: The user specified "punctuation is separated with spaces". 
    # The WordPiece tokenizer handles this well, but we ensure basic cleaning.
    def tokenize_function(examples):
        # We don't truncate here yet; we truncate in the grouping stage
        return tokenizer(examples["text"], return_special_tokens_mask=True)

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=["text"],
        desc="Tokenizing dataset",
    )

    # --------------------------------------------------------------------------
    # 3. CHUNKING (Handling documents > 256 tokens)
    # --------------------------------------------------------------------------
    # We concatenate all texts and then split them into chunks of MAX_SEQ_LENGTH.
    # This is more efficient for MLM than padding every line.
    
    def group_texts(examples):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # Drop the small remainder at the end
        if total_length >= MAX_SEQ_LENGTH:
            total_length = (total_length // MAX_SEQ_LENGTH) * MAX_SEQ_LENGTH
            
        # Split by chunks of MAX_SEQ_LENGTH
        result = {
            k: [t[i : i + MAX_SEQ_LENGTH] for i in range(0, total_length, MAX_SEQ_LENGTH)]
            for k, t in concatenated_examples.items()
        }
        return result

    print(f"Grouping texts into chunks of {MAX_SEQ_LENGTH} tokens...")
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=os.cpu_count(),
        desc="Grouping texts",
    )

    print(f"Training set size: {len(lm_datasets['train'])} chunks")
    print(f"Validation set size: {len(lm_datasets['validation'])} chunks")

    # --------------------------------------------------------------------------
    # 3.b VOCAB TEST (quick diagnostic)
    # --------------------------------------------------------------------------
    # Print tokens for the first 256-token sequence coming from the loaded corpus.
    if len(lm_datasets["train"]) > 0:
        first_chunk_ids = lm_datasets["train"][0]["input_ids"]
        first_chunk_tokens = tokenizer.convert_ids_to_tokens(first_chunk_ids)
        unk_count = sum(tok == tokenizer.unk_token for tok in first_chunk_tokens)
        unk_ratio = (unk_count / len(first_chunk_tokens)) if first_chunk_tokens else 0.0
        unk_warn_threshold = 0.05

        print("\n--- VOCAB TEST: first 256-token training chunk ---")
        print(f"Chunk length: {len(first_chunk_tokens)}")
        print(f"[UNK] count : {unk_count}")
        print(f"[UNK] ratio : {unk_ratio:.2%}")
        print(first_chunk_tokens)
        if unk_ratio > unk_warn_threshold:
            print(
                f"WARNING: [UNK] ratio ({unk_ratio:.2%}) is above {unk_warn_threshold:.0%}. "
                "Tokenizer/model compatibility may hurt continued pretraining quality."
            )
        print("--- END VOCAB TEST ---\n")

    # --------------------------------------------------------------------------
    # 4. TRAINING SETUP
    # --------------------------------------------------------------------------
    # Data collator dynamically masks tokens for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True, 
        mlm_probability=MLM_PROBABILITY
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        eval_strategy="steps",
        eval_steps=500,                  # Evaluate every 500 steps
        save_steps=500,                  # Save checkpoint every 500 steps
        save_total_limit=2,              # Keep only last 2 checkpoints to save space
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,               # Regularization to prevent overfitting
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        num_train_epochs=NUM_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=100,
        report_to="none",                # Disable wandb/mlflow reporting for simple run
        disable_tqdm=False,
        load_best_model_at_end=True,     # Load best model after training
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # -------------------------------------------------------------------------
    # 6. LOAD CHECKPOINT (OPTIONAL)
    # -------------------------------------------------------------------------
    # If you have a checkpoint from a previous run, you can load it here to resume training.
    checkpoint_path = "./latin-bert-adapted/checkpoint-66777" # Adjust this path if you have a different checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        trainer.train(resume_from_checkpoint=checkpoint_path)

    # --------------------------------------------------------------------------
    # 5. EXECUTE TRAINING
    # --------------------------------------------------------------------------
    # print("Starting training...")
    # train_result = trainer.train()
    
    # load a model checkpoint to evaluate before training
    print("Evaluating original model before training...")
    eval_metrics = trainer.evaluate()
    try:
        perplexity = math.exp(eval_metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")    

    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    metrics = eval_metrics
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    
    # Final Evaluation
    print("Final Evaluation...")
    eval_metrics = trainer.evaluate()
    try:
        perplexity = math.exp(eval_metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    
    print(f"Perplexity: {perplexity:.2f}")
    print(f"Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()