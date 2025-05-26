from flask import Flask, render_template, request
import praw
import re
import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from config import CLIENT_ID, CLIENT_SECRET, USER_AGENT
from tenacity import retry, stop_after_attempt, wait_fixed
from datetime import datetime
from collections import Counter
from itertools import zip_longest

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Reddit API and VADER
reddit = praw.Reddit(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, user_agent=USER_AGENT)
analyzer = SentimentIntensityAnalyzer()

# Add zip filter to Jinja2 environment
app.jinja_env.filters['zip'] = zip_longest

# Date formatting filter
app.jinja_env.filters['datetimeformat'] = lambda value: datetime.utcfromtimestamp(value).strftime('%Y-%m-%d %H:%M:%S')

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_post_data(post_url):
    try:
        logger.info(f"Fetching post: {post_url}")
        url_pattern = r'^https?://(www\.)?reddit\.com/r/[\w-]+/comments/([\w]+)/?.*$'
        match = re.match(url_pattern, post_url)
        if not match:
            logger.warning(f"Invalid URL format: {post_url}")
            return {"error": "Invalid Reddit URL format. Use: https://www.reddit.com/r/subreddit/comments/post_id/"}
        post_id = match.group(2)
        submission = reddit.submission(id=post_id)
        submission.comments.replace_more(limit=0)  # Fetch all comments
        comments = [comment.body for comment in submission.comments[:5] if not comment.is_submitter]  # Exclude OP's comments
        logger.info(f"Successfully fetched post {post_id} with {len(comments)} comments")
        return {
            "title": submission.title,
            "text": submission.selftext or submission.title,
            "author": submission.author.name if submission.author else "Deleted",
            "created_at": submission.created_utc,
            "comments": comments
        }
    except praw.exceptions.RedditAPIException as e:
        logger.error(f"Reddit API error: {str(e)}")
        return {"error": f"Cannot fetch post: {str(e)} (possibly deleted or private)"}
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {"error": f"Failed to fetch post: {str(e)}"}

def analyze_sentiment(text):
    try:
        scores = analyzer.polarity_scores(text)
        compound = scores['compound']
        logger.debug(f"Sentiment scores for '{text[:50]}...': {scores}")
        sentiment = "Positive" if compound > 0.05 else "Negative" if compound < -0.05 else "Neutral"
        return {"sentiment": sentiment, "score": compound}  # Raw compound score (-1 to 1)
    except Exception as e:
        logger.error(f"Sentiment analysis error: {str(e)}")
        return {"sentiment": "Neutral", "score": 0.0}  # Fallback

@app.route("/", methods=["GET", "POST"])  # Fixed syntax error
def home():
    if request.method == "POST":
        post_url = request.form.get("post_url")
        logger.info(f"Processing POST request for: {post_url}")
        post_data = get_post_data(post_url)
        if "error" in post_data:
            logger.warning(f"Error fetching post: {post_data['error']}")
            return render_template("index.html", error=post_data["error"])
        text = f"{post_data['title']} {post_data['text']}"
        sentiment_data = analyze_sentiment(text)
        comment_sentiments = [analyze_sentiment(comment)['sentiment'] for comment in post_data.get("comments", [])]
        # Prepare chart data
        sentiments = [sentiment_data['sentiment']] + comment_sentiments
        sentiment_counts = Counter(sentiments)
        chart_data = {
            "labels": ["Positive", "Negative", "Neutral"],
            "data": [sentiment_counts.get("Positive", 0), sentiment_counts.get("Negative", 0), sentiment_counts.get("Neutral", 0)]
        }
        logger.info(f"Post sentiment: {sentiment_data['sentiment']}, Comment sentiments: {comment_sentiments}")
        return render_template("index.html", post=post_data, sentiment=sentiment_data['sentiment'], 
                              sentiment_score=sentiment_data['score'], comment_sentiments=comment_sentiments, chart_data=chart_data)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)