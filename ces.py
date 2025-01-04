import pandas as pd
import argparse

def get_relationship_matrix():
    """
    Returns the emotion relationship matrix and mapping.
    """
    label2int = {
        "Sadness": 0,
        "Joy": 1,
        "Love": 2,
        "Anger": 3,
        "Fear": 4,
        "Surprise": 5
    }

    relationship_matrix = [
        [1, -1, -0.5, 0.5, 0.5, 0.2],  # Sadness
        [-1, 1, 0.5, -0.5, -0.5, 0.7],  # Joy
        [-0.5, 0.5, 1, -0.5, -0.2, 0.3],  # Love
        [0.5, -0.5, -0.5, 1, 0.3, 0.2],  # Anger
        [0.5, -0.5, -0.2, 0.3, 1, 0.1],  # Fear
        [0.2, 0.7, 0.3, 0.2, 0.1, 1]     # Surprise
    ]
    
    return label2int, relationship_matrix

def relationship_function(comment_emotion, transcript_emotion, label2int, relationship_matrix):
    """
    Get the relationship score between comment emotion and transcript emotion.
    """
    return relationship_matrix[label2int[transcript_emotion]][label2int[comment_emotion]]


def calculate_ces(top3_comment, top3_transcript, lambda_weight=0.5):
    label2int, relationship_matrix = get_relationship_matrix()
    ces_scores = {}
    
    for c_emotion, c_conf in top3_comment:
        ces = c_conf
        for t_emotion, t_conf in top3_transcript:
            relationship_score = relationship_function(c_emotion, t_emotion, label2int, relationship_matrix)
            ces += lambda_weight * (t_conf * relationship_score)
        ces_scores[c_emotion] = ces
    
    return ces_scores

def calculate_wrs(top_comment_emotion, top_comment_conf, top_transcript_emotion, top_transcript_conf, ces_scores, beta=0.3):
    label2int, relationship_matrix = get_relationship_matrix()
    relationship_score = relationship_function(top_comment_emotion, top_transcript_emotion, label2int, relationship_matrix)
    
    wrs = top_transcript_conf * top_comment_conf * relationship_score
    wrs += beta * ces_scores.get(top_comment_emotion, 0)
    
    ecf = None
    if wrs > -0.2:
        ecf = None
    elif -0.5 < wrs <= -0.2:
        ecf = "moderate_mismatch"
    elif wrs <= -0.5 and top_transcript_conf > 0.7 and top_comment_conf > 0.7:
        if top_transcript_emotion in ["Joy", "Love"] and top_comment_emotion in ["Anger", "Sadness"]:
            ecf = "high_sarcasm"
        elif top_transcript_emotion in ["Sadness", "Anger"] and top_comment_emotion in ["Joy", "Love"]:
            ecf = "high_irony"
    
    return wrs, ecf


def process_csv(input_csv, output_csv):
    """
    Process CSV file and calculate CES and WRS.
    """
    df = pd.read_csv(input_csv)
    
    # Validate columns
    required_columns = ['transcript_emotion1', 'transcript_conf1', 'comment_emotion1', 'comment_conf1']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Input CSV must contain columns: {required_columns}")
    
    results = []
    for index, row in df.iterrows():
        top3_transcript = [(row[f'transcript_emotion{i+1}'], row[f'transcript_conf{i+1}']) for i in range(3)]
        top3_comment = [(row[f'comment_emotion{i+1}'], row[f'comment_conf{i+1}']) for i in range(3)]
        
        ces_scores = calculate_ces(top3_comment, top3_transcript)
        final_emotion = max(ces_scores, key=ces_scores.get)
        
        wrs, ecf = calculate_wrs(
            top3_comment[0][0], top3_comment[0][1],
            top3_transcript[0][0], top3_transcript[0][1],
            ces_scores
        )
        
        results.append({
            'row_index': index + 1,
            "final_emotion": final_emotion,
            "WRS": wrs,
            "ECF": ecf,
            **ces_scores
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    return output_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    process_csv(args.input, args.output)
