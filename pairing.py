import csv
from sentence_transformers import SentenceTransformer, util
import torch
import argparse

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():  # For Apple Silicon
        return 'mps'
    else:
        return 'cpu'


def validate_csv(input_file):
    with open(input_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        if 'Transcript Chunk' not in reader.fieldnames or 'Comment' not in reader.fieldnames:
            raise ValueError("Input CSV must contain 'Transcript Chunk' and 'Comment' columns.")


def load_data(pairs_csv):
    pairs = []
    with open(pairs_csv, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            pairs.append((row['Transcript Chunk'], row['Comment']))
    return pairs


def compute_embeddings(model, texts, device):
    return model.encode(texts, convert_to_tensor=True, device=device)


def find_most_relevant_chunk(pairs_csv, output_csv):
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    # Validate CSV
    validate_csv(pairs_csv)

    # Load data
    pairs = load_data(pairs_csv)
    transcript_chunks = list(set(pair[0] for pair in pairs))
    comments = list(set(pair[1] for pair in pairs))

    print("Computing embeddings...")
    transcript_embeddings = compute_embeddings(model, transcript_chunks, device)
    comment_embeddings = compute_embeddings(model, comments, device)

    print("Calculating cosine similarity...")
    similarity_matrix = util.pytorch_cos_sim(comment_embeddings, transcript_embeddings)

    # Find most relevant transcript chunk for each comment
    results = []
    for comment_idx, comment in enumerate(comments):
        most_relevant_idx = torch.argmax(similarity_matrix[comment_idx]).item()
        most_relevant_chunk = transcript_chunks[most_relevant_idx]
        results.append((comment, most_relevant_chunk))

    # Save results to CSV
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Comment", "Most Relevant Transcript Chunk"])
        writer.writerows(results)

    print(f"Relevant chunks saved to {output_csv}")
    return output_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the most relevant transcript chunk for each comment.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input CSV file with transcript-comment pairs.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output CSV file.")
    
    args = parser.parse_args()
    
    find_most_relevant_chunk(args.input, args.output)
