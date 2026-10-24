import torch
from collections import Counter
import nemo.collections.asr as nemo_asr
from datasets import load_dataset
import os

def generate_idf():
    model_name = "nvidia/nemotron-3.5-asr-streaming-0.6b"
    print(f"Loading model tokenizer from {model_name}...")
    model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name)
    tokenizer = model.tokenizer
    vocab_size = tokenizer.vocab_size

    print("Loading Hindi text data for frequency counting...")
    # Load a text slice from Kathbath or Common Voice transcriptions
    dataset = load_dataset("google/fleurs", "hi_in", split="train", streaming=True)

    token_counts = Counter()
    total_docs = 0

    print("Counting token frequencies...")
    for idx, item in enumerate(dataset):
        text = item.get("text", item.get("sentence", ""))
        if not text.strip():
            continue
        
        # Tokenize text using model's tokenizer
        tokens = tokenizer.text_to_ids(text)
        token_counts.update(tokens)
        total_docs += 1

        if idx >= 50000:  # Sample enough sentences for stable frequency estimation
            print("Reached sample limit for IDF estimation.")
            break

    # Compute IDF: log(Total_Docs / (1 + Document_Frequency))
    idf_weights = torch.ones(vocab_size)
    for token_id, count in token_counts.items():
        if token_id < vocab_size:
            # Simple IDF formulation
            idf_weights[token_id] = torch.log(torch.tensor(float(total_docs) / (1.0 + count)))

    # Normalize/Scale weights nicely
    if idf_weights.max() > idf_weights.min():
        idf_weights = (idf_weights - idf_weights.min()) / (idf_weights.max() - idf_weights.min())

    os.makedirs(".", exist_ok=True)
    torch.save(idf_weights, "idf_weights.pt")
    print(f"Successfully generated 'idf_weights.pt' for vocab size {vocab_size}!")

if __name__ == "__main__":
    generate_idf()