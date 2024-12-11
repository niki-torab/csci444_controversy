import praw
import os
from gensim import corpora, models
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt

# Download NLTK data
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

# Fetch top 50 posts
for submission in subreddit.top(limit=50):
    submission.comments.replace_more(limit=0)
    comments = [comment.body for comment in submission.comments.list()[:10]]
    full_text = f"{submission.title} {submission.selftext} {' '.join(comments)}"
    organized_data.append({
        "title": submission.title,
        "body": submission.selftext,
        "comments": comments,
        "full_text": full_text
    })

# Text preprocessing
stop_words = set(stopwords.words('english'))
custom_stop_words = {'ai', 'like', 'would', 'one', 'much', 'could'}
stop_words.update(custom_stop_words)

processed_posts = []
for post in organized_data:
    tokens = word_tokenize(post["full_text"].lower())
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    processed_posts.append(tokens)

dictionary = corpora.Dictionary(processed_posts)
corpus = [dictionary.doc2bow(post) for post in processed_posts]

lda_model = models.LdaModel(corpus, num_topics=7, id2word=dictionary, passes=20)

print("\nTopics:")
for idx, topic in lda_model.print_topics(num_words=5):
    print(f"Topic {idx + 1}: {topic}")

# Categorize posts by their top topic
topic_posts = {i: [] for i in range(7)}
for i, bow in enumerate(corpus):
    topic_probabilities = lda_model[bow]
    top_topic = max(topic_probabilities, key=lambda x: x[1])[0]  # Get the most probable topic
    topic_posts[top_topic].append(organized_data[i])

print("\nPosts categorized by top topic:")
for topic, posts in topic_posts.items():
    print(f"\nTopic {topic + 1}:")
    for post in posts:
        print(f" - Title: {post['title']}")

# Calculate sentiment scores, variance, and average sentiment
sentiment_data = {}
variance_data = []
average_sentiments = []

for topic, posts in topic_posts.items():
    sentiment_data[topic] = []
    topic_sentiments = []
    for post in posts:
        title_sentiment = analyzer.polarity_scores(post["title"])
        body_sentiment = analyzer.polarity_scores(post["body"])
        comments_sentiments = [analyzer.polarity_scores(comment) for comment in post["comments"]]
        combined_sentiments = [title_sentiment["compound"], body_sentiment["compound"]] + [
            comment["compound"] for comment in comments_sentiments
        ]
        sentiment_data[topic].append(combined_sentiments)
        topic_sentiments.extend(combined_sentiments)
    
    combined_scores = [score for post in sentiment_data[topic] for score in post]
    variance = np.var(combined_scores) if combined_scores else 0
    average_sentiment = np.mean(combined_scores) if combined_scores else 0
    variance_data.append(variance)
    average_sentiments.append(average_sentiment)
    print(f"\nTopic {topic + 1}:")
    print(f"  Combined Sentiment Variance: {variance}")
    print(f"  Average Sentiment: {average_sentiment}")



# Scatter plot: Sentiment Variance vs Average Sentiment
plt.figure(figsize=(10, 6))

# Linear regression setup
X = np.array(average_sentiments).reshape(-1, 1)  # Average sentiment (X-axis)
y = np.array(variance_data)  # Sentiment variance (Y-axis)

# Fit linear regression
regressor = LinearRegression()
regressor.fit(X, y)
y_pred = regressor.predict(X)

# Calculate R²
r2 = r2_score(y, y_pred)

# Plot the scatter points
plt.scatter(average_sentiments, variance_data, color='green', s=100, alpha=0.7, label='Data Points')

# Plot the regression line
plt.plot(average_sentiments, y_pred, color='red', label=f'Linear Fit (R² = {r2:.2f})')

# Add labels for each topic
for i, (x, y) in enumerate(zip(average_sentiments, variance_data)):
    plt.text(x, y, f"Topic {i + 1}", fontsize=9, ha='right')

# Add plot details
plt.title('Sentiment Variance vs. Average Sentiment', fontsize=14)
plt.xlabel('Average Sentiment', fontsize=12)
plt.ylabel('Sentiment Variance', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.show()

