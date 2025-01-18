import os
import argparse
from scraper import setup_driver, scrape_video_data, save_data_to_json
from clean import process_cleaning
from pairing import find_most_relevant_chunk
from inference import run_inference
from ces import process_csv


def ensure_directory(path):
    """Creates the directory if it does not already exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def cleanup_files(files):
    """Removes the specified files if they exist."""
    for file in files:
        if os.path.exists(file):
            os.remove(file)


def main(url, output_dir, cleanup=False):
    ensure_directory(output_dir)
    driver = None

    try:
        scraper_output = os.path.join(output_dir, "video_data.json")
        driver = setup_driver()
        try:
            scraped_data = scrape_video_data(url, driver)
        finally:
            driver.quit()

        if scraped_data:
            save_data_to_json(scraped_data, scraper_output)
        else:
            print("Error: Scraping failed.")
            return

        cleaned_pairs_output = os.path.join(output_dir, "cleaned_pairs.csv")
        process_cleaning(scraper_output, cleaned_pairs_output, max_chunk_length=64, chunk_overlap=16)

        paired_output = os.path.join(output_dir, "relevant_chunks.csv")
        find_most_relevant_chunk(cleaned_pairs_output, paired_output)

        inference_output = os.path.join(output_dir, "emotion_predictions.csv")
        run_inference(
            paired_output,
            inference_output,
            model_path='balancedai_emotion_classification_model.pth'
        )

        ces_output = os.path.join(output_dir, "dependent_emotion_classification.csv")
        process_csv(inference_output, ces_output)

        if cleanup:
            intermediate_files = [
                scraper_output,
                cleaned_pairs_output,
                paired_output,
                inference_output
            ]
            cleanup_files(intermediate_files)

    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        if driver:
            try:
                driver.quit()
            except Exception as e:
                print(f"Cleanup error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YouTube Video Emotion Analysis Workflow")
    parser.add_argument("--url", type=str, help="YouTube video URL")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save outputs")
    parser.add_argument("--cleanup", action="store_true", help="Remove intermediate files after processing")

    args = parser.parse_args()
    if not args.url:
        args.url = input("Enter YouTube URL: ").strip()
        if not args.url:
            print("Error: No URL provided.")
            exit(1)

    main(args.url, args.output_dir, args.cleanup)
