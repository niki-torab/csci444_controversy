import praw
import os
from dotenv import load_dotenv

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
all_text = ""

for submission in subreddit.top(limit=50):
    submission.comments.replace_more(limit=0)
    comments = "\n".join([f"Comment: {comment.body}" for comment in submission.comments.list()[:10]])
    full_text = (
        f"Title: {submission.title}\n"
        f"Body: {submission.selftext}\n"
        f"{comments}\n\n"
    )
    all_text += full_text

with open("output.txt", "w", encoding="utf-8") as file:
    file.write(all_text)

print("Output saved to 'output.txt'")
