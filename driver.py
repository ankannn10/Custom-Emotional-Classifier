import os
import json
import argparse
import pandas as pd
from scraper import setup_driver, scrape_video_data, save_data_to_json
from clean import process_cleaning
from pairing import find_most_relevant_chunk
from inference import run_inference
from ces import process_csv
import google.generativeai as genai # type: ignore

# Configure the API key
genai.configure(api_key='API KEY') 

model = genai.GenerativeModel('gemini-1.0-pro')

# Load JSON Transcript
def load_transcript_from_json(file_path):
    """Load transcript text from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print("Error: JSON file not found.")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON file.")
        return None

# Summarize Transcript Using Gemini API
def summarize_transcript(transcript):
    """Summarize the provided transcript using Gemini API."""
    if not transcript:
        print("Error: Empty transcript. Cannot summarize.")
        return ""
    try:
        response = model.generate_content(
            f"Summarize the following transcript in 3-5 sentences:\n\n{transcript}"
        )
        return response.text.strip()
    except Exception as e:
        print(f"API Error: {e}")
        return ""

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def cleanup_files(files):
    for file in files:
        if os.path.exists(file):
            os.remove(file)


def main(url, output_dir, cleanup=False):
    ensure_directory(output_dir)
    
    try:
        # Step 1: Scrape Data
        print("\n🔍 Step 1: Scraping YouTube Video Data...")
        scraper_output = os.path.join(output_dir, "video_data.json")
        driver = setup_driver()
        try:
            scraped_data = scrape_video_data(url, driver)
        finally:
            driver.quit()
        
        if scraped_data:
            save_data_to_json(scraped_data, scraper_output)
        else:
            print("❌ Error: Scraping failed. Exiting.")
            return
        json_file_path = scraper_output
        # Step 1: Load JSON data
        data = load_transcript_from_json(json_file_path)
    
        if data and "transcript" in data:
            transcript = data["transcript"]
        
        # Step 2: Summarize transcript
            summary = summarize_transcript(transcript)
        
            if summary:
                # Step 3: Save summary to csv
                summary_data = {
                    "summary": summary
                }
                summary_df = pd.DataFrame([summary_data])
                summary_output = os.path.join(output_dir, "summary.csv")
                summary_df.to_csv(summary_output, index=False, quoting=1)  # Ensure quoting
                print(f"✅ Summary saved to {summary_output}")

                print("\n📚 Transcript Summary Successful:\n")
                #print(summary)
                print("✅ Summary saved to summary.csv")
            else:
                print("❌ Failed to generate a summary.")
        else:
            print("❌ Transcript not found in the JSON file.")


        # Step 2: Clean Data
        print("\n🧹 Step 2: Cleaning Transcript and Comments...")
        cleaned_pairs_output = os.path.join(output_dir, "cleaned_pairs.csv")
        process_cleaning(scraper_output, cleaned_pairs_output, max_chunk_length=64, chunk_overlap=16)
        
        # Step 3: Pair Relevant Chunks
        print("\n🔗 Step 3: Pairing Comments with Relevant Transcript Chunks...")
        paired_output = os.path.join(output_dir, "relevant_chunks.csv")
        find_most_relevant_chunk(cleaned_pairs_output, paired_output)
        
        # Step 4: Run Emotion Inference
        print("\n🤖 Step 4: Running Emotion Inference...")
        inference_output = os.path.join(output_dir, "emotion_predictions.csv")
        run_inference(paired_output, inference_output, model_path='balancedai_emotion_classification_model.pth')
        
        # Step 5: Calculate CES and WRS
        print("\n📊 Step 5: Calculating CES and WRS...")
        ces_output = os.path.join(output_dir, "dependent_emotion_classification.csv")
        process_csv(inference_output, ces_output)
        
        print("\n✅ Workflow completed successfully!")
        print(f"📁 Independent Emotion Classification: {inference_output}")
        print(f"📁 Dependent Emotion Classification: {ces_output}")
        
        # Cleanup Intermediate Files
        if cleanup:
            print("\n🗑️ Cleaning up intermediate files...")
            intermediate_files = [scraper_output, cleaned_pairs_output, paired_output, inference_output]
            cleanup_files(intermediate_files)
            print("✅ Intermediate files removed.")
        
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
    finally:
        print("\n🧹 Performing final cleanup...")
        try:
            if driver:
                driver.quit()
        except Exception as e:
            print(f"❌ Final cleanup error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YouTube Video Emotion Analysis Workflow")
    parser.add_argument("--url", type=str, help="YouTube video URL")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save outputs")
    parser.add_argument("--cleanup", action="store_true", help="Remove intermediate files after processing")
    
    args = parser.parse_args()
    
    # Interactive Prompt for URL if not provided
    if not args.url:
        print("\n📝 No URL provided via arguments. Please enter a YouTube URL below:")
        args.url = input("Enter YouTube URL: ").strip()
        if not args.url:
            print("❌ Error: No URL provided. Exiting.")
            exit(1)
    
    main(args.url, args.output_dir, args.cleanup)
