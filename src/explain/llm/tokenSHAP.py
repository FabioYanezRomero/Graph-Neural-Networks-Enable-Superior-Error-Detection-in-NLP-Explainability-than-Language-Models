from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from token_shap import TokenSHAP, StringSplitter, LocalModel
import torch
import os
import time
from tqdm import tqdm
import sys
import pickle

# Configure tqdm for better progress bar display
tqdm.monitor_interval = 0

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Path to the finetuned model checkpoint
MODEL_PATH = "outputs/llm/stanfordnlp/sst2/sst2_2025-06-04_14-52-49"

# Load the model and tokenizer directly from the checkpoint directory
# This ensures we load the finetuned classification head weights
base_model_name = "google-bert/bert-base-uncased"
print("Loading base model...")
model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load the finetuned weights
checkpoint_path = os.path.join(MODEL_PATH, "model_epoch_2.pt")
if os.path.exists(checkpoint_path):
    print("Loading finetuned weights...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Load the state dict directly - this includes the finetuned classifier weights
    model.load_state_dict(checkpoint)
    print(f"Loaded finetuned model from {checkpoint_path}")
    print(f"Model contains {sum(p.numel() for p in model.parameters())} parameters")
else:
    print(f"Warning: Checkpoint not found at {checkpoint_path}, using base model")

# Move model to GPU
model = model.to(device)
model.eval()
print("Model loaded and ready!")

# Create a custom model wrapper for token_shap
class HuggingFaceModelWrapper:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def generate(self, prompt):
        # Tokenize the input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Move inputs to GPU
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
            
            # Get both class probabilities for more detailed output
            prob_0 = probabilities[0][0].item()
            prob_1 = probabilities[0][1].item()
        
        # Return a more detailed prediction that includes both probabilities
        # This helps TokenSHAP differentiate between different token combinations
        return f"Class: {predicted_class}, P(0): {prob_0:.6f}, P(1): {prob_1:.6f}, Conf: {confidence:.6f}"

# HF wordpiece splitter to align token count with graph nodes
class HFWordpieceSplitter:
    def __init__(self, tokenizer, include_special=True, max_length=512):
        self.tokenizer = tokenizer
        self.include_special = include_special
        self.max_length = max_length
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id

    def split(self, text):
        enc = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
        )
        ids = enc["input_ids"]
        attn = enc["attention_mask"]
        # keep only attention_mask==1 positions
        ids = [tid for tid, m in zip(ids, attn) if m == 1]
        if not self.include_special:
            ids = [tid for tid in ids if tid not in (self.cls_id, self.sep_id)]
        self._ids = ids
        self._tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return list(self._tokens)

    def join(self, tokens_subset):
        # Convert back to ids
        ids = self.tokenizer.convert_tokens_to_ids(tokens_subset)
        if self.include_special:
            # Ensure CLS/SEP are present for a valid input
            if len(ids) == 0 or ids[0] != self.cls_id:
                ids = [self.cls_id] + ids
            if ids[-1] != self.sep_id:
                ids = ids + [self.sep_id]
        # Decode to text; downstream wrapper will re-tokenize
        return self.tokenizer.decode(
            ids,
            skip_special_tokens=True,  # surface text; special tokens will be added by join above
            clean_up_tokenization_spaces=True,
        )

# Create the model wrapper
hf_model = HuggingFaceModelWrapper(model, tokenizer, device)
# splitter = StringSplitter()
# Exclude special tokens from the attribution set for fair comparison with GraphSVX.
splitter = HFWordpieceSplitter(tokenizer, include_special=False)
explainer = TokenSHAP(hf_model, splitter)

print("Loading dataset...")
ds = load_dataset("stanfordnlp/sst2", split="validation")

results = {}
sentence_count = 0
i = 0

# Process sentences until we have exactly 10 successful analyses
while sentence_count < 10 and i < len(ds):
    prompt = ds[i]["sentence"]
    
    # Calculate number of tokens to estimate memory usage (aligned to graph nodes)
    tokens = splitter.split(prompt)
    num_tokens = len(tokens)
    
    # Skip sentences with more than 21 tokens to mirror graph cap
    if num_tokens > 21:
        print(f"Skipping sentence {i+1} (too long: {num_tokens} tokens): {prompt[:50]}...")
        i += 1
        continue
    
    max_combinations = 2 ** num_tokens
    print(f"\nSentence {sentence_count+1}: {prompt}")
    print(f"Number of tokens: {num_tokens}, Max combinations: {max_combinations}")
    
    # Use adaptive sampling ratio based on sentence length
    if num_tokens <= 8:
        sampling_ratio = 0.5  # 50% for short sentences
    elif num_tokens <= 12:
        sampling_ratio = 0.1  # 10% for medium sentences
    elif num_tokens <= 16:
        sampling_ratio = 0.01  # 1% for long sentences
    else:
        sampling_ratio = 0.001  # 0.1% for very long sentences
    
    print(f"Using sampling ratio: {sampling_ratio}")
    print("Starting TokenSHAP analysis...")
    
    try:
        start_time = time.time()
        
        # Force flush stdout to ensure progress bar displays correctly
        sys.stdout.flush()
        
        # Use adaptive sampling ratio to prevent memory explosion
        df = explainer.analyze(prompt, sampling_ratio=sampling_ratio, print_highlight_text=True)
        
        elapsed_time = time.time() - start_time
        print(f"Analysis completed in {elapsed_time:.2f} seconds")
        print(f"Generated {len(df)} combinations")
        print(df.head(10))  # Show only first 10 rows to avoid overwhelming output
        
        # Store results with metadata - use sentence_count as key to ensure exactly 10 samples
        results[f"sentence_{sentence_count+1}"] = {
            "prompt": prompt,
            "result": df, 
            "sampling_ratio": sampling_ratio, 
            "num_tokens": num_tokens, 
            "max_combinations": max_combinations,
            "elapsed_time": elapsed_time,
            "dataset_index": i
        }
        
        sentence_count += 1
        print(f"✓ Successfully analyzed sentence {sentence_count}/10")
        
    except ZeroDivisionError:
        print("Warning: All token combinations produced identical predictions. Skipping analysis.")
    except MemoryError:
        print(f"Memory error: Sentence too long ({num_tokens} tokens). Skipping analysis.")
    except Exception as e:
        print(f"Error during analysis: {e}")
    
    i += 1

# Check if we got exactly 10 samples
if sentence_count < 10:
    print(f"\n⚠️  Warning: Only processed {sentence_count} sentences out of 10 requested.")
    print(f"   Examined {i} sentences from dataset (dataset has {len(ds)} total sentences)")
else:
    print(f"\n✅ Successfully processed exactly 10 sentences!")

print(f"Analysis complete! Processed {sentence_count} sentences out of {i} examined.")

# Save results to file
output_file = "tokenSHAP_results.pkl"
print(f"Saving results to {output_file}...")
with open(output_file, 'wb') as f:
    pickle.dump(results, f)

print(f"Results saved successfully! File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")

# Also save a summary in JSON format
import json
summary = {}
for sentence_key, data in results.items():
    summary[sentence_key] = {
        "prompt": data["prompt"],
        "num_tokens": data["num_tokens"],
        "max_combinations": data["max_combinations"],
        "sampling_ratio": data["sampling_ratio"],
        "elapsed_time": data["elapsed_time"],
        "dataset_index": data["dataset_index"],
        "result_shape": data["result"].shape if data["result"] is not None else None
    }

with open("tokenSHAP_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print("Summary saved to tokenSHAP_summary.json")
print(f"Total samples saved: {len(results)}")
        
