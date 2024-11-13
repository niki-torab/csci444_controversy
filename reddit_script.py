import praw
import os
from gensim import corpora, models
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv

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

subreddit = reddit.subreddit('artificial')
top_posts = []
for submission in subreddit.top(limit=10):
    submission.comments.replace_more(limit=0)
    comments = " ".join([comment.body for comment in submission.comments.list()[:10]])
    full_text = f"{submission.title} {submission.selftext} {comments}"
    top_posts.append(full_text)

stop_words = set(stopwords.words('english'))
processed_posts = []
for post in top_posts:
    tokens = word_tokenize(post.lower())
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
        topic_posts[topic_id].append(top_posts[i])

print("\nPosts categorized by top 3 topics:")
for topic, posts in topic_posts.items():
    print(f"\nTopic {topic + 1}:")
    for post in posts:
        print(f" - {post}")
