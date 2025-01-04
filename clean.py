import json
import csv
from itertools import product
import re
import argparse
import pandas as pd
import emoji
import re

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

# Preprocessing Function
def preprocess_text(text):
    text = emoji.demojize(text)
    text = text.lower()
    
    for slang, replacement in SLANG_DICT.items():
        text = re.sub(r'\b' + re.escape(slang) + r'\b', replacement, text)
    
    #Remove Unwanted Characters (Keep emoji tokens intact)
    text = re.sub(r'[^a-zA-Z0-9\s:]', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# Clean Comment Function
def clean_comment(comment):
    if not comment:
        return None
        
    comment = re.sub(r'\r\n|\n', ' ', comment)  # Replace newlines with space
    comment = re.sub(r'\s+', ' ', comment).strip()  # Remove excessive spaces
    
    if re.match(r'^\s*@', comment):
        return None  # Skip comments starting with '@'
    
    comment = re.sub(r'http\S+|www.\S+', '', comment)  # Remove URLs
    
    # Preprocess with emoji and slang replacement
    comment = preprocess_text(comment)
    
    # Final cleanup
    comment = re.sub(r'\s+', ' ', comment).strip()
    
    return comment if comment else None



def clean_transcript(transcript):
    if not transcript:
        return ""
    
    transcript = re.sub(r'\[.*?\]', '', transcript)  # Remove bracketed content
    transcript = re.sub(r'\[\d{2}:\d{2}\]', '', transcript)  # Remove timestamps
    transcript = re.sub(r'\r\n|\n', ' ', transcript)  # Replace newlines with spaces
    transcript = re.sub(r'\s+', ' ', transcript)  # Remove multiple spaces
    transcript = re.sub(r'[^\w\s.,!?-]', '', transcript)  # Remove unwanted characters
    
    return transcript.strip()


def chunk_text(text, max_length=64, overlap=16):
    words = text.split()
    chunks = []
    
    if not words:
        return chunks
        
    for i in range(0, len(words), max_length - overlap):
        chunk = " ".join(words[i:i + max_length])
        if chunk:
            chunks.append(chunk)
    return chunks


def process_cleaning(input_file, output_file, max_chunk_length=64, chunk_overlap=16):
    try:
        # Read JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Clean transcript
        transcript = clean_transcript(data.get('transcript', ''))
        if not transcript:
            raise ValueError("No valid transcript found in the file")
        
        # Clean comments
        comments = data.get('comments', [])
        cleaned_comments = [clean_comment(comment) for comment in comments if clean_comment(comment)]
        
        if not cleaned_comments:
            raise ValueError("No valid comments found in the file")
        
        # Create chunks from transcript
        chunks = chunk_text(transcript, max_length=max_chunk_length, overlap=chunk_overlap)
        # Save transcript chunks to chunks.csv
        chunks_df = pd.DataFrame({
            'chunk_index': range(1, len(chunks) + 1),
            'Transcript Chunk': chunks
        })
        chunks_df.to_csv('output/chunks.csv', index=False)
        print(f"Saved {len(chunks)} transcript chunks to output2/chunks.csv")
        if not chunks:
            raise ValueError("Transcript could not be chunked into meaningful parts")
        
        # Create pairs
        pairs = list(product(chunks, cleaned_comments))
        
        # Save to CSV
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
