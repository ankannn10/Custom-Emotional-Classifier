import csv
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import argparse

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():  # For Apple Silicon
        return 'mps'
    else:
        return 'cpu'

class Mish(torch.nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


def load_model(model_path, device):
    #tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    #base_model = AutoModel.from_pretrained('roberta-base')
    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
    base_model = AutoModel.from_pretrained('distilroberta-base')
    
    class EmoModelWithAttention(torch.nn.Module):
        def __init__(self, base_model, n_classes):
            super().__init__()
            self.base_model = base_model
            self.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.1),
                torch.nn.Linear(768, 768),
                Mish(), 
                torch.nn.Dropout(0.1),
                torch.nn.Linear(768, n_classes)
            )
            self.attention_weights = torch.nn.Linear(768, 1)  # Attention score per token

        def forward(self, input_ids, attention_mask):
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)
            
            # Compute attention scores
            attention_scores = self.attention_weights(hidden_states).squeeze(-1)
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))
            attention_weights = F.softmax(attention_scores, dim=1)
            
            # Weighted average of token embeddings
            pooled_output = torch.sum(hidden_states * attention_weights.unsqueeze(-1), dim=1)
            
            return self.classifier(pooled_output)
    
    model = EmoModelWithAttention(base_model=base_model, n_classes=6)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)
    model.eval()
    
    return model, tokenizer

label_mapping = {
    0: 'Sadness',
    1: 'Joy',
    2: 'Love',
    3: 'Anger',
    4: 'Fear',
    5: 'Surprise'
}

def chunk_text(text, max_length=64, overlap=16):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_length - overlap):
        chunk = " ".join(words[i:i + max_length])
        chunks.append(chunk)
    return chunks

def predict_top3_emotions(text, model, tokenizer, device):
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=64,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = F.softmax(outputs, dim=1).squeeze(0)
    
    top3_indices = torch.argsort(probabilities, descending=True)[:3]
    top3_labels = [label_mapping[idx.item()] for idx in top3_indices]
    top3_scores = [probabilities[idx].item() for idx in top3_indices]
    
    return list(zip(top3_labels, top3_scores))

def predict_top3_emotions_for_long_text(text, model, tokenizer, device):
    chunks = chunk_text(text, max_length=64)
    aggregated_scores = torch.zeros(6, device=device)  # Assuming 6 emotion classes
    
    for chunk in chunks:
        encoding = tokenizer(
            chunk,
            truncation=True,
            padding='max_length',
            max_length=64,
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask']
            )
            probabilities = F.softmax(outputs, dim=1).squeeze(0)
        
        aggregated_scores += probabilities
    
    aggregated_scores /= len(chunks)  # Average across all chunks
    
    # Extract top-3 predictions
    top3_indices = torch.argsort(aggregated_scores, descending=True)[:3]
    top3_labels = [label_mapping[idx.item()] for idx in top3_indices]
    top3_scores = [aggregated_scores[idx].item() for idx in top3_indices]
    
    return list(zip(top3_labels, top3_scores))

CHUNKS_FILE = "./output/chunks.csv"  # Hardcoded path for chunks.csv

def update_chunks_with_emotions(model, tokenizer, device):
    try:
        df = pd.read_csv(CHUNKS_FILE)
        
        if 'Transcript Chunk' not in df.columns:
            raise ValueError("chunks.csv must contain 'Transcript Chunk' column.")
        
        emotion_results = []
        for index, row in df.iterrows():
            chunk = row['Transcript Chunk']
            emotions = predict_top3_emotions(chunk, model, tokenizer, device)
            
            emotion_results.append({
                'row_index': index + 1,
                'Transcript Chunk': chunk,
                **{f'transcript_emotion{i+1}': emotion for i, (emotion, _) in enumerate(emotions)},
                **{f'transcript_conf{i+1}': score for i, (_, score) in enumerate(emotions)},
            })
        
        updated_df = pd.DataFrame(emotion_results)
        updated_df.to_csv(CHUNKS_FILE, index=False)
        print(f"✅ Transcript chunk predictions updated in {CHUNKS_FILE}")
    
    except FileNotFoundError:
        print(f"❌ Error: {CHUNKS_FILE} not found. Please ensure the file exists.")
    except Exception as e:
        print(f"❌ Error processing {CHUNKS_FILE}: {e}")

summary_file = "./output/summary.csv"  

def run_inference_on_summary(model, tokenizer, device):
    try:
        # Load the summary file
        df = pd.read_csv(summary_file)
        
        # Validate the presence of the 'summary' column
        if 'summary' not in df.columns:
            raise ValueError("summary.csv must contain a 'summary' column.")
        
        # Fetch the summary text
        summary = df['summary'].iloc[0] if not df.empty else None
        
        if summary:
            # Predict emotions for the summary
            emotions = predict_top3_emotions(summary, model, tokenizer, device)
            
            # Prepare the updated data
            summary_data = {
                'summary': summary,
                **{f'summary_emotion{i+1}': emotion for i, (emotion, _) in enumerate(emotions)},
                **{f'summary_conf{i+1}': score for i, (_, score) in enumerate(emotions)},
            }
            
            # Save the updated summary back to the CSV
            updated_df = pd.DataFrame([summary_data])
            updated_df.to_csv(summary_file, index=False, quoting=csv.QUOTE_ALL)
            
            print(f"✅ Summary predictions updated in {summary_file}")
        else:
            print("❌ No summary text found in the CSV file.")
    
    except FileNotFoundError:
        print(f"❌ Error: {summary_file} not found. Please ensure the file exists.")
    except Exception as e:
        print(f"❌ Error processing {summary_file}: {e}")


def run_inference(input_csv, output_csv, model_path):
    device = get_device()
    print(f"🖥️ Using device: {device}")
    
    # Load model and tokenizer
    model, tokenizer = load_model(model_path, device)
    print("✅ Model and tokenizer loaded successfully.")
    
    update_chunks_with_emotions(model, tokenizer, device)
    run_inference_on_summary(model, tokenizer, device)

    # Load data
    df = pd.read_csv(input_csv)
    
    # Validate required columns
    required_columns = ['Most Relevant Transcript Chunk', 'Comment']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Input CSV must contain columns: {required_columns}")
    
    results = []
    for index, row in df.iterrows():
        transcript_emotions = predict_top3_emotions(row['Most Relevant Transcript Chunk'], model, tokenizer, device)
        comment_emotions = predict_top3_emotions_for_long_text(row['Comment'], model, tokenizer, device)
        
        results.append({
            'row_index': index + 1,
            'comment': row['Comment'],
            **{f'comment_emotion{i+1}': emotion for i, (emotion, _) in enumerate(comment_emotions)},
            **{f'comment_conf{i+1}': score for i, (_, score) in enumerate(comment_emotions)},
            'transcript_chunk': row['Most Relevant Transcript Chunk'],
            **{f'transcript_emotion{i+1}': emotion for i, (emotion, _) in enumerate(transcript_emotions)},
            **{f'transcript_conf{i+1}': score for i, (_, score) in enumerate(transcript_emotions)},
        })
    
    # Save results
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_csv, index=False)
    print(f"✅ Predictions saved to {output_csv}")
    return output_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run emotion inference on a dataset.")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file.")
    parser.add_argument("--output", type=str, required=True, help="Path to output CSV file.")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights file.")
    parser = argparse.ArgumentParser(description="Run emotion inference on summary.")
    parser.add_argument("--json", type=str, required=True, help="Path to JSON file with summary.")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights file.")
    parser.add_argument("--output", type=str, default="summary.csv", help="Path to output summary CSV file.")

    args = parser.parse_args()
    run_inference(args.input, args.output, args.model)
