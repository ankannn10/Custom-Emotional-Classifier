import json
import csv
from itertools import product
import re
import argparse
import pandas as pd
import emoji
import os
from typing import List, Optional


# Slang Dictionary
SLANG_DICT = {
    "lol": "laugh out loud",
    "omg": "oh my god",
    "btw": "by the way",
    "idk": "i don't know",
    "lmao": "laughing my ass off",
    "smh": "shaking my head",
    "brb": "be right back",
    "gtg": "got to go",
    "tbh": "to be honest",
    "rofl": "rolling on the floor laughing",
    "thx": "thanks",
    "u": "you",
    "w": "with",
    "ur": "your",
    "r": "are",
    "k": "okay",
    "ikr": "i know right",
    "afk": "away from keyboard",
    "bff": "best friends forever",
    "gr8": "great",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "fyi": "for your information",
    "ttyl": "talk to you later",
    "np": "no problem",
    "wtf": "what the heck",
    "wth": "what the heck",
    "tmi": "too much information",
    "gg": "good game",
    "hmu": "hit me up",
    "wyd": "what are you doing",
    "wya": "where are you",
    "ily": "i love you",
    "ily2": "i love you too",
    "ikr": "i know right",
    "idc": "i don't care",
    "omw": "on my way",
    "irl": "in real life",
    "bday": "birthday",
    "bae": "baby",
    "bro": "brother",
    "sis": "sister",
    "yolo": "you only live once",
    "fomo": "fear of missing out",
    "sus": "suspicious",
    "lit": "amazing",
    "dope": "cool",
    "srsly": "seriously",
    "nah": "no",
    "yea": "yes",
    "tho": "though",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "lemme": "let me",
    "gimme": "give me",
    "ain't": "is not",
    "cuz": "because",
    "coz": "because",
    "tho": "though",
    "ya": "you",
    "nvm": "never mind",
    "omfg": "oh my god",
    "ffs": "for goodness sake",
    "pls": "please",
    "ppl": "people",
    "rn": "right now",
    "smh": "shaking my head",
    "tldr": "too long didn't read",
    "xoxo": "hugs and kisses",
    "asap": "as soon as possible",
    "jk": "just kidding",
    "b4": "before",
    "bc": "because",
    "thx": "thanks",
    "roflmao": "rolling on the floor laughing my ass off",
    "wtf": "what the heck",
    "thot": "that hoe over there",
    "clout": "influence or popularity",
    "cap": "lie",
    "no cap": "no lie",
    "bet": "okay",
    "fam": "family",
    "savage": "fierce",
    "lowkey": "a little",
    "highkey": "a lot",
    "dm": "direct message",
    "stan": "superfan",
    "salty": "bitter or upset",
    "snatched": "looking good",
    "fire": "amazing",
    "tea": "gossip",
    "slay": "do something amazingly well",
    "flex": "show off",
    "ghosted": "ignored",
    "shipping": "want two people to be in a relationship",
    "vibe": "atmosphere or mood"
}


def preprocess_text(text: str) -> str:
    """
    Preprocesses text by replacing emojis and slang terms with their meanings.
    
    Args:
        text (str): The input text to preprocess.
    
    Returns:
        str: The cleaned and preprocessed text.
    """
    text = emoji.demojize(text).lower()
    
    for slang, replacement in SLANG_DICT.items():
        text = re.sub(rf'\b{re.escape(slang)}\b', replacement, text)
    
    text = re.sub(r'[^a-zA-Z0-9\s:]', '', text)  # Remove unwanted characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text


def clean_comment(comment: str) -> Optional[str]:
    """
    Cleans a single comment by removing unwanted characters and preprocessing slang.
    
    Args:
        comment (str): The comment to clean.
    
    Returns:
        Optional[str]: The cleaned comment or None if invalid.
    """
    if not comment:
        return None

    comment = re.sub(r'\r\n|\n', ' ', comment)  # Replace newlines with space
    comment = re.sub(r'\s+', ' ', comment).strip()  # Remove excessive spaces

    if re.match(r'^\s*@', comment):  # Skip comments starting with '@'
        return None

    comment = re.sub(r'http\S+|www\.\S+', '', comment)  # Remove URLs
    comment = preprocess_text(comment)
    return comment if comment else None


def clean_transcript(transcript: str) -> str:
    """
    Cleans a transcript by removing timestamps and unwanted characters.
    
    Args:
        transcript (str): The transcript to clean.
    
    Returns:
        str: The cleaned transcript.
    """
    if not transcript:
        return ""

    transcript = re.sub(r'\[.*?\]', '', transcript)  # Remove bracketed content
    transcript = re.sub(r'\[\d{2}:\d{2}\]', '', transcript)  # Remove timestamps
    transcript = re.sub(r'\r\n|\n', ' ', transcript)  # Replace newlines with spaces
    transcript = re.sub(r'\s+', ' ', transcript)  # Remove multiple spaces
    transcript = re.sub(r'[^\w\s.,!?-]', '', transcript)  # Remove unwanted characters
    return transcript.strip()


def chunk_text(text: str, max_length: int = 64, overlap: int = 16) -> List[str]:
    """
    Splits text into overlapping chunks.
    
    Args:
        text (str): The text to split into chunks.
        max_length (int): The maximum length of each chunk.
        overlap (int): The number of overlapping words between chunks.
    
    Returns:
        List[str]: A list of text chunks.
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_length - overlap):
        chunk = " ".join(words[i:i + max_length])
        chunks.append(chunk)
    return chunks


def process_cleaning(input_file: str, output_file: str, max_chunk_length: int = 64, chunk_overlap: int = 16) -> Optional[str]:
    """
    Processes the input JSON file to clean transcript and comments and pairs them.
    
    Args:
        input_file (str): Path to the input JSON file.
        output_file (str): Path to the output CSV file.
        max_chunk_length (int): Maximum length of each transcript chunk.
        chunk_overlap (int): Overlap between transcript chunks.
    
    Returns:
        Optional[str]: Path to the output file if successful, None otherwise.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        transcript = clean_transcript(data.get('transcript', ''))
        if not transcript:
            raise ValueError("No valid transcript found in the file")

        comments = data.get('comments', [])
        cleaned_comments = [clean_comment(comment) for comment in comments if clean_comment(comment)]
        if not cleaned_comments:
            raise ValueError("No valid comments found in the file")

        chunks = chunk_text(transcript, max_length=max_chunk_length, overlap=chunk_overlap)
        if not chunks:
            raise ValueError("Transcript could not be chunked into meaningful parts")

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Save transcript chunks
        chunks_df = pd.DataFrame({'chunk_index': range(1, len(chunks) + 1), 'Transcript Chunk': chunks})
        chunks_df.to_csv('output/chunks.csv', index=False)
        print(f"Saved {len(chunks)} transcript chunks to output/chunks.csv")

        pairs = list(product(chunks, cleaned_comments))
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Transcript Chunk", "Comment"])
            writer.writerows(pairs)

        print(f"Processing complete! Generated {len(pairs)} pairs.")
        print(f"Output saved to {output_file}")
        return output_file

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and pair transcript and comments from a JSON file.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output CSV file.")
    parser.add_argument("--chunk_length", type=int, default=64, help="Maximum length of each transcript chunk.")
    parser.add_argument("--chunk_overlap", type=int, default=16, help="Overlap between transcript chunks.")

    args = parser.parse_args()

    process_cleaning(
        input_file=args.input,
        output_file=args.output,
        max_chunk_length=args.chunk_length,
        chunk_overlap=args.chunk_overlap
    )
