from flask import Flask, render_template, request, jsonify  # type: ignore
import os
import json
import pandas as pd
import traceback

# Importing pipeline orchestrator
from driver import main as run_pipeline

app = Flask(__name__)

def assign_sentiment(top_emotion, confidence):
    if confidence < 0.7:
        return "Neutral"
    elif top_emotion in ["Anger", "Sadness", "Fear"]:
        return "Negative"
    else:
        return "Positive"

def ratio_as_percentage(part, whole):
    if whole == 0:
        return 0.0
    return round((part / whole) * 100, 2)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        # Extracting the YouTube link
        youtube_link = request.form.get("yt_link", "").strip()
        if not youtube_link:
            return jsonify({"error": "No YouTube link provided"}), 400

        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        run_pipeline(youtube_link, output_dir, cleanup=False)

        emotion_csv_path = os.path.join(output_dir, "emotion_predictions.csv")
        if not os.path.exists(emotion_csv_path):
            return jsonify({"error": "emotion_predictions.csv not found"}), 500

        df_emotions = pd.read_csv(emotion_csv_path)

        required_cols = ["comment", "comment_emotion1", "comment_conf1"]
        if not all(col in df_emotions.columns for col in required_cols):
            return jsonify(
                {"error": f"Missing one or more required columns: {required_cols} in emotion_predictions.csv"}
            ), 500

        # Assigning sentiment for each comment
        df_emotions["sentiment"] = df_emotions.apply(
            lambda row: assign_sentiment(row["comment_emotion1"], row["comment_conf1"]),
            axis=1
        )

        # Converting to a list of dicts for Jinja
        report_data = df_emotions.to_dict(orient="records")

        # Loading overall video summary from summary.csv
        summary_csv_path = os.path.join(output_dir, "summary.csv")
        video_summary = None
        if os.path.exists(summary_csv_path):
            df_summary = pd.read_csv(summary_csv_path)
            if "summary" in df_summary.columns and not df_summary.empty:
                video_summary = df_summary["summary"].iloc[0]
            else:
                video_summary = "No 'summary' column found in summary.csv"
        else:
            video_summary = "summary.csv not found"

        # Loading video_data.json for engagement metrics
        video_json_path = os.path.join(output_dir, "video_data.json")
        title = ""
        channel_name = ""
        upload_date = ""
        tags = []
        likes_to_views_perc = 0.0
        comments_to_views_perc = 0.0
        views_to_subs_perc = 0.0

        if os.path.exists(video_json_path):
            with open(video_json_path, "r", encoding="utf-8") as f:
                video_data = json.load(f)
                
            title = video_data.get("title", "No Title")
            channel_name = video_data.get("channel_name", "Unknown Channel")
            tags = video_data.get("tags", [])
            views = video_data.get("views", 0)
            likes = video_data.get("likes", 0)
            upload_date = video_data.get("upload_date", "Unknown Date")
            comment_count = video_data.get("comment_count", 0)
            subscriber_count = video_data.get("subscriber_count", 0)

            likes_to_views_perc = ratio_as_percentage(likes, views)
            comments_to_views_perc = ratio_as_percentage(comment_count, views)
            views_to_subs_perc = ratio_as_percentage(views, subscriber_count)
        else:
            print("video_data.json not found. Skipping engagement metrics.")

        # Computing Overall Public Opinion (sentiment & top emotion)
        overall_sentiment = "N/A"
        overall_emotion = "N/A"
        if not df_emotions.empty:
            # Overall Sentiment
            sentiment_counts = df_emotions["sentiment"].value_counts(dropna=False)
            if not sentiment_counts.empty:
                overall_sentiment = sentiment_counts.idxmax()

            # Overall Emotion
            emotion_counts = df_emotions["comment_emotion1"].value_counts(dropna=False)
            if not emotion_counts.empty:
                overall_emotion = emotion_counts.idxmax()

        return render_template(
            "results.html",
            video_link=youtube_link,
            report_data=report_data,
            video_summary=video_summary,
            overall_sentiment=overall_sentiment,
            overall_emotion=overall_emotion,

            # Engagement fields
            video_title=title,
            channel_name=channel_name,
            tags=tags,
            upload_date = upload_date,
            likes_to_views_perc=likes_to_views_perc,
            comments_to_views_perc=comments_to_views_perc,
            views_to_subs_perc=views_to_subs_perc
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # For local testing
    app.run(debug=True, host="0.0.0.0", port=5000)
