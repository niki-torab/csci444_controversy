import praw
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import models
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load environment variables
load_dotenv()

client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')
user_agent = os.getenv('USER_AGENT')

reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent
)

analyzer = SentimentIntensityAnalyzer()

data_file = "posts_data.json"

def fetch_posts(subreddit, limit=500):
    # Check if data file exists and has data
    if os.path.exists(data_file):
        with open(data_file, "r") as f:
            try:
                existing_data = json.load(f)
                if existing_data:  # If there's existing data, return it
                    print(f"Loaded {len(existing_data)} posts from {data_file}")
                    return existing_data
            except json.JSONDecodeError:
                pass  # If file is empty or invalid JSON, we'll fetch fresh data

    # If we reach here, we need to fetch from Reddit
    print(f"No local data found or file is empty. Fetching {limit} posts from Reddit...")
    posts = []
    for submission in subreddit.top(limit=limit):
        submission.comments.replace_more(limit=0)
        comments = [comment.body for comment in submission.comments.list()]
        full_text = f"{submission.title} {submission.selftext} {' '.join(comments)}"
        posts.append({
            "title": submission.title,
            "body": submission.selftext,
            "comments": comments,
            "full_text": full_text,
            "comment_count": len(submission.comments.list())
        })
    print(f"Total posts fetched: {len(posts)}")

    # Save fetched data to file
    with open(data_file, "w") as f:
        json.dump(posts, f, indent=4)

    return posts

def preprocess_posts(posts):
    stop_words = set(stopwords.words('english'))
    custom_stop_words = {'ai', 'like', 'would', 'one', 'much', 'could'}
    stop_words.update(custom_stop_words)

    processed_texts = []
    empty_count = 0  # Counter for empty processed texts

    for idx, post in enumerate(posts):
        raw_text = post["full_text"]
        tokens = word_tokenize(raw_text.lower())
        tokens = [word for word in tokens if word.isalpha()]
        tokens = [word for word in tokens if word not in stop_words]

        processed_text = " ".join(tokens)
        if processed_text.strip():  # Ensure the text is not empty
            processed_texts.append(processed_text)
        else:
            empty_count += 1
            print(f"Post {idx + 1} resulted in empty text after preprocessing.")

    print(f"\nTotal posts processed: {len(posts)}")
    print(f"Posts with empty processed texts: {empty_count}")
    return processed_texts

def process_batches(posts, batch_size=50, num_topics=7):
    total_batches = (len(posts) // batch_size) + (1 if len(posts) % batch_size != 0 else 0)
    if total_batches == 0:
        print("No posts to process.")
        return []

    batch_logs = []

    for batch_num in range(total_batches):
        start = batch_num * batch_size
        end = min(start + batch_size, len(posts))

        print("Start: ", start)
        print("End: ", end)

        batch_posts = posts[start:end]

        # Preprocess posts
        processed_texts = preprocess_posts(batch_posts)

        # TF-IDF Vectorization
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(processed_texts)

        # Convert TF-IDF to Gensim corpus format
        dictionary = {i: word for i, word in enumerate(tfidf_vectorizer.get_feature_names_out())}
        corpus = [list(zip(tfidf_matrix[i].indices, tfidf_matrix[i].data)) for i in range(tfidf_matrix.shape[0])]

        # LDA Topic Modeling
        lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=20)
        topics = lda_model.print_topics(num_words=5)

        # Assign posts to topics
        topic_posts = {i: [] for i in range(num_topics)}
        for i, bow in enumerate(corpus):
            topic_probabilities = lda_model[bow]
            top_topic = max(topic_probabilities, key=lambda x: x[1])[0]
            topic_posts[top_topic].append(batch_posts[i])

        # Sentiment Analysis
        sentiment_data = {}
        for topic, post_thing in topic_posts.items():
            combined_scores = []
            for post in post_thing:
                title_sentiment = analyzer.polarity_scores(post["title"])
                body_sentiment = analyzer.polarity_scores(post["body"])
                comments_sentiments = [analyzer.polarity_scores(comment) for comment in post["comments"]]
                combined_sentiments = [title_sentiment["compound"], body_sentiment["compound"]] + [
                    comment["compound"] for comment in comments_sentiments
                ]
                combined_scores.extend(combined_sentiments)
            sentiment_data[topic] = np.var(combined_scores) if combined_scores else 0

        # Log batch details
        batch_log = {
            "batch_num": batch_num + 1,
            "topics": {idx: topic for idx, topic in enumerate(topics)},
            "sentiment_variance": sentiment_data,
            "posts_per_topic": {topic: [post["title"] for post in posts_list] for topic, posts_list in topic_posts.items()}
        }
        batch_logs.append(batch_log)

        # Print batch summary
        print(f"\nBatch {batch_num + 1}/{total_batches} Processed:")
        print(json.dumps(batch_log, indent=4))

    # Save logs to file
    with open("batch_logs.json", "w") as log_file:
        json.dump(batch_logs, log_file, indent=4)

    return batch_logs

# Main processing
print("Fetching posts...")
subreddit = reddit.subreddit('artificial')
posts = fetch_posts(subreddit, limit=500)

print("Processing batches...")
batch_logs = process_batches(posts, batch_size=50)
