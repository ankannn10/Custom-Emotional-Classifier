# 📊 YouTube Emotion Analysis

## 🚀 Overview
YouTube Emotion Analysis is a comprehensive pipeline designed to extract, process, and analyze emotional insights from YouTube video transcripts and comments. This project combines natural language processing, machine learning, and a user-friendly web interface to provide meaningful emotional metrics, sentiment scores, and contextual insights.

## 🌟 Key Features
- **YouTube Data Scraper:** Extracts video metadata, comments, views, likes, and transcripts.
- **Data Cleaning:** Removes noise, handles slang, and preprocesses text.
- **Transcript & Comment Pairing:** Aligns comments with relevant transcript chunks.
- **Emotion Classification Model:** Fine-tuned Transformer model with attention mechanism and Mish activation.
- **CES (Contextual Emotion Score) & WRS (Weighted Relationship Score):** Measures contextual emotional alignment.
- **Summarization:** AI-powered transcript summarization using Google Gemini API.
- **Visualization Dashboard:** Intuitive web interface for presenting results.

## 🛠️ Tech Stack
- **Python** (Backend Processing)
- **Transformers** (Hugging Face Library)
- **Flask** (Web Framework)
- **Pandas** (Data Processing)
- **Sentence Transformers** (Semantic Text Matching)
- **Selenium** (Web Scraping)
- **Google Gemini API** (Summarization)
- **Bootstrap 5** (Frontend Styling)

## 📁 Project Structure
```
📂 project-root/
├── app.py                # Flask web application
├── driver.py             # Pipeline orchestrator
├── scraper.py            # Data scraping module
├── clean.py              # Data cleaning module
├── pairing.py            # Comment-transcript pairing
├── inference.py          # Emotion classification
├── ces.py                # CES and WRS calculation
├── templates/
│   ├── index.html        # Input form for video link
│   ├── results.html      # Display results
├── static/
│   ├── emotion_model_architecture.png  # Model architecture diagram
└── requirements.txt      # Dependencies
```

## 🧠 Model Architecture
The emotion classification model is built on a fine-tuned **DistilRoBERTa** architecture with an attention mechanism and Mish activation function.

### 📊 **Model Workflow**
1. **Input Text (Tokenized)**
2. **Pre-trained RoBERTa (Hidden States)**
3. **Attention Mechanism (Token-level Attention Scores)**
4. **Weighted Average Pooling (Attention-Weighted Hidden States)**
5. **Fully Connected Layers with Mish Activation & Dropout**
6. **Softmax Activation (Emotion Probabilities)**
7. **Top-3 Emotions with Confidence Scores**

![Model Architecture](/static/emotion_model_architecture.png)
### Metrics
**Test Results:**
- **Test Loss:** 0.1021
- **Test Accuracy:** 0.9428
- **Precision:** 0.9520
- **Recall:** 0.9428
- **F1-Score:** 0.9444

**Classification Report:**
```
               precision    recall  f1-score   support

     sadness       1.00      0.95      0.98     18178
         joy       1.00      0.92      0.96     21160
        love       0.77      1.00      0.87      5183
       anger       0.91      1.00      0.95      8598
        fear       0.94      0.87      0.90      7157
    surprise       0.75      1.00      0.86      2246

    accuracy                           0.94     62522
   macro avg       0.90      0.96      0.92     62522
weighted avg       0.95      0.94      0.94     62522
```

## 📦 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ankannn10/Custom-Emotional-Classifier.git
   cd Custom-Emotional-Classifier
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # For MacOS/Linux
   .\.venv\Scripts\activate  # For Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   - `GOOGLE_API_KEY` for Gemini API

## ▶️ Usage
### **Run Flask App**
```bash
python app.py
```
Navigate to `http://localhost:5000` in your browser.

### **Analyze a YouTube Video**
1. Enter the YouTube video URL.
2. Click **Analyze**.
3. View results including:
   - Engagement metrics
   - Emotional insights
   - Transcript summary

## 📊 Output Files
- **video_data.json:** Raw scraped data.
- **cleaned_pairs.csv:** Preprocessed transcript-comment pairs.
- **relevant_chunks.csv:** Paired comments and transcript chunks.
- **emotion_predictions.csv:** Top-3 emotions for each comment and transcript chunk.
- **dependent_emotion_classification.csv:** CES and WRS scores.
- **summary.csv:** Transcript summary with predicted emotions.

## 🤝 Contribution
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Make changes and commit:
   ```bash
   git commit -m 'Add new feature'
   ```
4. Push changes:
   ```bash
   git push origin feature-branch
   ```
5. Open a pull request.

## 📜 License
This project is licensed under the **MIT License**.

## 📧 Contact
For inquiries or feedback:
- **Email:** ankan10.edu@gmail.com
- **GitHub:** [Ankan](https://github.com/ankannn10)

---
_This project aims to bridge the gap between raw data and actionable emotional insights for YouTube content creators and marketers._
