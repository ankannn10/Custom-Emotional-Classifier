import os
import argparse
from scraper_updated import setup_driver, scrape_video_data, save_data_to_json
from clean_new import process_cleaning
from pairing_updated import find_most_relevant_chunk
from inf_attn import run_inference
from ces_updated import process_csv

# -------------------------------
# 🚀 File Path Utility Functions
# -------------------------------
def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def cleanup_files(files):
    for file in files:
        if os.path.exists(file):
            os.remove(file)


# -------------------------------
# 🚀 Workflow Pipeline
# -------------------------------
def main(url, output_dir, cleanup=False):
    ensure_directory(output_dir)
    
    try:
        # Step 1: Scrape Data
        print("\n🔍 Step 1: Scraping YouTube Video Data...")
        scraper_output = os.path.join(output_dir, "video_data.json")
        driver = setup_driver()
        scraped_data = scrape_video_data(url, driver)
        driver.quit()
        
        if scraped_data:
            save_data_to_json(scraped_data, scraper_output)
        else:
            print("❌ Error: Scraping failed. Exiting.")
            return
        
        # Step 2: Clean Data
        print("\n🧹 Step 2: Cleaning Transcript and Comments...")
        cleaned_pairs_output = os.path.join(output_dir, "cleaned_pairs.csv")
        process_cleaning(scraper_output, cleaned_pairs_output, max_chunk_length=128, chunk_overlap=32)
        
        # Step 3: Pair Relevant Chunks
        print("\n🔗 Step 3: Pairing Comments with Relevant Transcript Chunks...")
        paired_output = os.path.join(output_dir, "relevant_chunks.csv")
        find_most_relevant_chunk(cleaned_pairs_output, paired_output)
        
        # Step 4: Run Emotion Inference
        print("\n🤖 Step 4: Running Emotion Inference...")
        inference_output = os.path.join(output_dir, "emotion_predictions.csv")
        run_inference(paired_output, inference_output, model_path='ai_emotion_classification_model.pth')
        
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


# -------------------------------
# 🚀 Main Entry Point
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YouTube Video Emotion Analysis Workflow")
    parser.add_argument("--url", type=str, help="YouTube video URL")
    parser.add_argument("--output_dir", type=str, default="output2", help="Directory to save outputs")
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
