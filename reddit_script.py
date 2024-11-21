import praw
import os
from gensim import corpora, models
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

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

subreddit = reddit.subreddit('artificial')
organized_data = []

for submission in subreddit.top(limit=50):  # Adjusted to fetch 50 posts
    submission.comments.replace_more(limit=0)
    comments = [comment.body for comment in submission.comments.list()[:10]]
    full_text = f"{submission.title} {submission.selftext} {' '.join(comments)}"
    organized_data.append({
        "title": submission.title,
        "body": submission.selftext,
        "comments": comments,
        "full_text": full_text
    })

stop_words = set(stopwords.words('english'))
processed_posts = []
for post in organized_data:
    tokens = word_tokenize(post["full_text"].lower())
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    processed_posts.append(tokens)

dictionary = corpora.Dictionary(processed_posts)
corpus = [dictionary.doc2bow(post) for post in processed_posts]

lda_model = models.LdaModel(corpus, num_topics=6, id2word=dictionary, passes=10)

print("\nTopics:")
for idx, topic in lda_model.print_topics(num_words=5):
    print(f"Topic {idx + 1}: {topic}")

topic_posts = {i: [] for i in range(6)}
for i, bow in enumerate(corpus):
    topic_probabilities = lda_model[bow]
    top_3_topics = sorted(topic_probabilities, key=lambda x: x[1], reverse=True)[:3]
    for topic_id, _ in top_3_topics:
        topic_posts[topic_id].append(organized_data[i])

print("\nPosts categorized by top 3 topics:")
for topic, posts in topic_posts.items():
    print(f"\nTopic {topic + 1}:")
    for post in posts:
        print(f" - Title: {post['title']}")

print("\nCalculating sentiment scores...")

sentiment_data = {}
for topic, posts in topic_posts.items():
    sentiment_data[topic] = []
    for post in posts:
        title_sentiment = analyzer.polarity_scores(post["title"])
        body_sentiment = analyzer.polarity_scores(post["body"])
        comments_sentiments = [analyzer.polarity_scores(comment) for comment in post["comments"]]
        combined_sentiments = [title_sentiment["compound"], body_sentiment["compound"]] + [
            comment["compound"] for comment in comments_sentiments
        ]
        sentiment_data[topic].append({
            "combined_sentiments": combined_sentiments
        })

print("\nCalculating variance for combined distributions across topics...")

for topic, posts in sentiment_data.items():
    combined_scores = []
    for post in posts:
        combined_scores.extend(post["combined_sentiments"])
    combined_variance = np.var(combined_scores) if combined_scores else 0
    print(f"\nTopic {topic + 1}:")
    print(f"  Combined Sentiment Variance: {combined_variance}")
