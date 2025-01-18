import csv
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import argparse
from typing import List, Tuple


def get_device() -> str:
    """
    Determines the best available device for computation.
    
    Returns:
        str: The device to use ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():  # For Apple Silicon
        return 'mps'
    else:
        return 'cpu'


class Mish(nn.Module):
    """Implements the Mish activation function."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


def load_model(model_path: str, device: str) -> Tuple[nn.Module, AutoTokenizer]:
    """
    Loads the emotion model and tokenizer.
    
    Args:
        model_path (str): Path to the saved model weights.
        device (str): Device to load the model onto.
    
    Returns:
        Tuple[nn.Module, AutoTokenizer]: The model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
    base_model = AutoModel.from_pretrained('distilroberta-base')
    
    class EmoModelWithAttention(nn.Module):
        """Defines the emotion model with attention mechanism."""
        def __init__(self, base_model: AutoModel, n_classes: int):
            super().__init__()
            self.base_model = base_model
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(768, 768),
                Mish(),
                nn.Dropout(0.2),
                nn.Linear(768, n_classes)
            )
            self.attention_weights = nn.Linear(768, 1)
            self.layer_norm = nn.LayerNorm(768)

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            attention_scores = self.attention_weights(hidden_states).squeeze(-1)
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))
            attention_weights = F.softmax(attention_scores, dim=1)
            pooled_output = torch.sum(hidden_states * attention_weights.unsqueeze(-1), dim=1)
            return self.classifier(self.layer_norm(pooled_output))
    
    model = EmoModelWithAttention(base_model=base_model, n_classes=6)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device).eval()
    return model, tokenizer


def chunk_text(text: str, max_length: int = 64, overlap: int = 16) -> List[str]:
    """
    Splits text into overlapping chunks.
    
    Args:
        text (str): The text to split.
        max_length (int): Maximum length of each chunk.
        overlap (int): Number of overlapping tokens.
    
    Returns:
        List[str]: List of text chunks.
    """
    words = text.split()
    return [" ".join(words[i:i + max_length]) for i in range(0, len(words), max_length - overlap)]


def predict_top3_emotions(text: str, model: nn.Module, tokenizer: AutoTokenizer, device: str) -> List[Tuple[str, float]]:
    """
    Predicts the top-3 emotions for a given text.
    
    Args:
        text (str): The input text.
        model (nn.Module): The emotion classification model.
        tokenizer (AutoTokenizer): The tokenizer for preprocessing.
        device (str): The device to use for computation.
    
    Returns:
        List[Tuple[str, float]]: List of top-3 emotions and their confidence scores.
    """
    encoding = tokenizer(text, truncation=True, padding='max_length', max_length=64, return_tensors='pt').to(device)
    with torch.no_grad():
        probabilities = F.softmax(model(**encoding), dim=1).squeeze(0)
    top3_indices = torch.argsort(probabilities, descending=True)[:3]
    label_mapping = {0: 'Sadness', 1: 'Joy', 2: 'Love', 3: 'Anger', 4: 'Fear', 5: 'Surprise'}
    return [(label_mapping[idx.item()], probabilities[idx].item()) for idx in top3_indices]


def update_chunks_with_emotions(chunks_file: str, model: nn.Module, tokenizer: AutoTokenizer, device: str) -> None:
    """
    Updates chunks.csv with emotion predictions for each transcript chunk.
    
    Args:
        chunks_file (str): Path to the chunks.csv file.
        model (nn.Module): The emotion classification model.
        tokenizer (AutoTokenizer): The tokenizer for preprocessing.
        device (str): The device to use for computation.
    """
    try:
        df = pd.read_csv(chunks_file)
        if 'Transcript Chunk' not in df.columns:
            raise ValueError("chunks.csv must contain 'Transcript Chunk' column.")
        
        emotion_results = []
        for index, chunk in enumerate(df['Transcript Chunk']):
            emotions = predict_top3_emotions(chunk, model, tokenizer, device)
            emotion_results.append({
                'row_index': index + 1,
                'Transcript Chunk': chunk,
                **{f'transcript_emotion{i+1}': emotion for i, (emotion, _) in enumerate(emotions)},
                **{f'transcript_conf{i+1}': score for i, (_, score) in enumerate(emotions)},
            })
        
        updated_df = pd.DataFrame(emotion_results)
        updated_df.to_csv(chunks_file, index=False)
        print(f" Transcript chunk predictions updated in {chunks_file}")
    except Exception as e:
        print(f" Error processing {chunks_file}: {e}")


def run_inference(input_csv: str, output_csv: str, model_path: str) -> str:
    """
    Runs emotion inference for each transcript-comment pair in the input CSV.
    
    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to save the predictions.
        model_path (str): Path to the saved model weights.
    
    Returns:
        str: Path to the output CSV file.
    """
    device = get_device()
    print(f" Using device: {device}")
    model, tokenizer = load_model(model_path, device)
    update_chunks_with_emotions('./output/chunks.csv', model, tokenizer, device)

    df = pd.read_csv(input_csv)
    results = []
    for _, row in df.iterrows():
        comment_emotions = predict_top3_emotions(row['Comment'], model, tokenizer, device)
        transcript_emotions = predict_top3_emotions(row['Most Relevant Transcript Chunk'], model, tokenizer, device)
        results.append({
            'comment': row['Comment'],
            **{f'comment_emotion{i+1}': e for i, (e, _) in enumerate(comment_emotions)},
            **{f'comment_conf{i+1}': s for i, (_, s) in enumerate(comment_emotions)},
            **{f'transcript_emotion{i+1}': e for i, (e, _) in enumerate(transcript_emotions)},
            **{f'transcript_conf{i+1}': s for i, (_, s) in enumerate(transcript_emotions)},
        })
    
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f" Predictions saved to {output_csv}")
    return output_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run emotion inference on a dataset.")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file.")
    parser.add_argument("--output", type=str, required=True, help="Path to output CSV file.")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights file.")
    args = parser.parse_args()
    run_inference(args.input, args.output, args.model)
