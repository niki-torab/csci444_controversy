import praw
import os
from gensim import corpora, models
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv

nltk.download('punkt')

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
top_posts = [submission.title for submission in subreddit.top(limit=15)]
print(top_posts)

stop_words = set(stopwords.words('english'))
processed_posts = []
for post in top_posts:
    tokens = word_tokenize(post.lower())  
    tokens = [word for word in tokens if word.isalpha()]  
    tokens = [word for word in tokens if word not in stop_words]  
    processed_posts.append(tokens)

dictionary = corpora.Dictionary(processed_posts)
corpus = [dictionary.doc2bow(post) for post in processed_posts]

lda_model = models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=10)

for idx, topic in lda_model.print_topics(num_words=5):
    print(f"Topic {idx + 1}: {topic}")