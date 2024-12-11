import os
import praw
from openai import OpenAI
from gensim import corpora, models
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.stats import spearmanr
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
import re
import json
import sys

# Load environment variables
load_dotenv()

client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')
user_agent = os.getenv('USER_AGENT')

openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
analyzer = SentimentIntensityAnalyzer()

LOG_FILE = "batch_results_log.json"  # Path to the log file
DATA_FILE = "posts_data.json"        # Cached Reddit posts file

def log_batch_results(batch_num, topics, topic_posts, gpt_rankings, variance_rankings, comment_count_rankings, sentiment_rankings, correlations):
    """Logs the details of each batch to a JSON file."""
    batch_data = {
        "batch_num": batch_num,
        "topics": topics,
        "topic_posts": {k: [post['title'] for post in posts] for k, posts in topic_posts.items()},
        "rankings": {
            "GPT": gpt_rankings,
            "Variance": variance_rankings,
            "Comment_Count": comment_count_rankings,
            "Sentiment": sentiment_rankings
        },
        "correlations": correlations
    }

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(batch_data)

    with open(LOG_FILE, "w") as f:
        json.dump(data, f, indent=4)

def rank_topics_with_gpt(client, topics, topic_posts):
    topic_descriptions = []
    for idx, posts in topic_posts.items():
        if not posts:
            topic_descriptions.append(f"Topic {idx + 1}: (No posts)")
            continue
        examples = []
        for post in posts[:5]:  # Include up to 5 posts per topic
            body_preview = post['body'][:200] if len(post['body']) > 200 else post['body']
            comments_preview = " ".join(post['comments'][:3])  # Up to 3 comments
            examples.append(f"- Title: {post['title']}\n  Body: {body_preview}\n  Comments: {comments_preview}")
        topic_descriptions.append(f"Topic {idx + 1}:\n" + "\n".join(examples))

    prompt = (
        "You are given several topics with example posts. "
        "Please rank ALL of these topics from most to least controversial. "
        "Do not skip any topics. Make sure your final answer is exactly as follows:\n"
        "1. Topic X\n2. Topic Y\n...\n"
        "(with each line containing the rank and the word 'Topic' followed by the topic number). "
        "If a topic has no posts, still rank it at the bottom. Do Not EVER give me an explanation. Just the numbers and topics. That is it."
        "Here are the topics:\n\n" 
        + "\n\n".join(topic_descriptions)
    )
    print("Grouping of topics for this batch:")
    # print(prompt)

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7
    )

    content = completion.choices[0].message.content.strip()
    print("GPT ranking: ", content)
    rankings = content.split("\n")

    gpt_order = []
    for line in rankings:
        line = line.strip()
        if line and line[0].isdigit():
            try:
                rank_num = int(line.split('.')[0])
                match = re.search(r"Topic\s+(\d+)", line, re.IGNORECASE)
                if match:
                    topic_id = int(match.group(1)) - 1  # zero-based
                    gpt_order.append((rank_num, topic_id))
            except:
                continue
    
    gpt_order.sort(key=lambda x: x[0])  # sort by rank number
    if len(gpt_order) != len(topics):
        return None
    gpt_rankings = [topic_id for _, topic_id in gpt_order]
    gpt_rankings = gpt_rankings[::-1]

    return gpt_rankings

def load_posts_from_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} does not exist. Please ensure the file is present.")
        sys.exit(1)
    with open(file_path, "r") as f:
        data = json.load(f)
        if not data:
            print(f"Error: {file_path} is empty or invalid.")
            sys.exit(1)
    return data

def process_batches(client, all_submissions, num_batches, batch_size):
    all_results = []
    num_topics = 7

    for batch_num in range(num_batches):
        start = batch_num * batch_size
        end = start + batch_size
        batch = all_submissions[start:end]

        if not batch:
            print(f"Skipping batch {batch_num}: No posts in this batch.")
            continue

        # Limit comments to 5 per post
        processed_texts = []
        for post in batch:
            truncated_comments = post["comments"][:10]  # Only take top 5 comments
            full_text = f"{post['title']} {post['body']} {' '.join(truncated_comments)}"
            processed_texts.append(full_text)

        # TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(processed_texts)

        # Construct Gensim corpus
        gensim_corpus = [
            [(idx, value) for idx, value in enumerate(tfidf_matrix[i].toarray()[0]) if value > 0]
            for i in range(tfidf_matrix.shape[0])
        ]

        # Create dictionary from feature names
        dictionary = corpora.Dictionary([list(vectorizer.get_feature_names_out())])

        if len(dictionary) == 0:
            continue

        # LDA Topic Modeling
        lda_model = models.LdaModel(corpus=gensim_corpus, num_topics=num_topics, id2word=dictionary, passes=10)

        topics = [lda_model.print_topic(i, topn=5) for i in range(num_topics)]

        topic_posts = {i: [] for i in range(num_topics)}
        for i, bow in enumerate(gensim_corpus):
            topic_dist = lda_model[bow]
            if topic_dist:
                top_topic = max(topic_dist, key=lambda x: x[1])[0]
                topic_posts[top_topic].append(batch[i])

        variance_rankings = []
        comment_count_rankings = []
        sentiment_rankings = []

        for i in range(num_topics):
            posts = topic_posts[i]
            topic_sentiments = []
            comment_counts = []

            for post in posts:
                all_texts = [post["title"], post["body"]] + post["comments"]
                scores = [analyzer.polarity_scores(text)["compound"] for text in all_texts if text.strip()]
                topic_sentiments.extend(scores)
                comment_counts.append(len(post["comments"]))

            variance_rankings.append(np.var(topic_sentiments) if topic_sentiments else 0)
            comment_count_rankings.append(np.mean(comment_counts) if comment_counts else 0)
            sentiment_rankings.append(np.mean(topic_sentiments) if topic_sentiments else 0)

        gpt_rankings = rank_topics_with_gpt(client, topics, topic_posts)
        if gpt_rankings is None or len(gpt_rankings) != num_topics:
            continue

        gpt_topic_ranks = [0]*num_topics
        for rank_pos, topic_id in enumerate(gpt_rankings):
            gpt_topic_ranks[topic_id] = rank_pos + 1
        print(gpt_rankings)

        print(variance_rankings)
        sorted_indices = sorted(range(len(variance_rankings)), key=lambda i: variance_rankings[i])
        one_based_indices = [i+1 for i in sorted_indices]
        spearman_variance = spearmanr(gpt_topic_ranks, one_based_indices).correlation
        print(one_based_indices)


        print(comment_count_rankings)
        sorted_indices = sorted(range(len(comment_count_rankings)), key=lambda i: comment_count_rankings[i])
        one_based_indices = [i+1 for i in sorted_indices]
        spearman_comments = spearmanr(gpt_topic_ranks, one_based_indices).correlation
        print(one_based_indices)

        
        print(sentiment_rankings)
        sorted_indices = sorted(range(len(sentiment_rankings)), key=lambda i: sentiment_rankings[i])
        one_based_indices = [i+1 for i in sorted_indices]
        print(one_based_indices)


        spearman_sentiment = spearmanr(gpt_topic_ranks, one_based_indices).correlation

        correlations = {
            "variance_r": spearman_variance,
            "comments_r": spearman_comments,
            "sentiment_r": spearman_sentiment
        }

        log_batch_results(batch_num, topics, topic_posts, gpt_rankings, variance_rankings, comment_count_rankings, sentiment_rankings, correlations)
        
        all_results.append(correlations)

    return all_results

# Run analysis
subreddit_name = 'artificial'
num_batches = 10
batch_size = 50
total_posts = num_batches * batch_size

# Load data from the local file instead of fetching from the API
all_submissions = load_posts_from_file(DATA_FILE)

results = process_batches(openai_client, all_submissions, num_batches, batch_size)

variance_scores = [result["variance_r"] for result in results if result["variance_r"] is not None]
comments_scores = [result["comments_r"] for result in results if result["comments_r"] is not None]
sentiment_scores = [result["sentiment_r"] for result in results if result["sentiment_r"] is not None]

print(f"\nAverage Spearman's Correlation:")
if variance_scores:
    print(f"  Variance Ranking: {np.mean(variance_scores):.2f}")
if comments_scores:
    print(f"  Comment Count Ranking: {np.mean(comments_scores):.2f}")
if sentiment_scores:
    print(f"  Sentiment Ranking: {np.mean(sentiment_scores):.2f}")

plt.figure(figsize=(10, 6))
metrics = []
values = []
if variance_scores:
    metrics.append("Variance")
    values.append(np.mean(variance_scores))
if comments_scores:
    metrics.append("Comment Count")
    values.append(np.mean(comments_scores))
if sentiment_scores:
    metrics.append("Sentiment")
    values.append(np.mean(sentiment_scores))

plt.bar(metrics, values)
plt.title("Average Spearman's Correlation Across Batches")
plt.ylabel("Spearman's Correlation")
plt.show()
