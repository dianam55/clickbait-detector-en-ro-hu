import praw
import csv
import time

#Reddit data collection (Hot posts)
reddit = praw.Reddit( #Reddit API client (use your own id/secret/username)
    client_id="...", 
    client_secret="...",
    user_agent="..."
)

def collect_headlines(subreddits, total_posts=1000):
    all_headlines = []
    for subreddit_name in subreddits:
        print(f"Collecting from r/{subreddit_name}...")
        subreddit = reddit.subreddit(subreddit_name)
        headlines = []
        try:
            for submission in subreddit.new(limit=total_posts):
                headlines.append([submission.title, ""])
            all_headlines.extend(headlines[:total_posts])
            print(f"Collected {len(headlines)} headlines from r/{subreddit_name}")
        except Exception as e:
            print(f"Error collecting from r/{subreddit_name}: {e}")
    return all_headlines

subreddits = ["Romania", "Bucuresti", "Cluj", "Iasi", "Timisoara", "StiriRomania", "Romemes", "rojokes", "RoPolitica"]
headlines = collect_headlines(subreddits, total_posts=1000)

with open("romanian_reddit_headlines.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["headline", "label"])
    writer.writerows(headlines)

print(f"Total collected: {len(headlines)} headlines from {len(subreddits)} subreddits")


#Reddit data collection (Top posts of all time)
def collect_headlines(subreddits, total_posts=1000):
    all_headlines = []
    for subreddit_name in subreddits:
        print(f"Collecting top posts from r/{subreddit_name}...")
        subreddit = reddit.subreddit(subreddit_name)
        headlines = []
        try:
            for submission in subreddit.top(time_filter="all", limit=total_posts):
                headlines.append([submission.title, ""])
                time.sleep(0.1) 
            all_headlines.extend(headlines[:total_posts])
            print(f"Collected {len(headlines)} headlines from r/{subreddit_name}")
        except Exception as e:
            print(f"Error collecting from r/{subreddit_name}: {e}")
    return all_headlines

subreddits = ["Romania", "Bucuresti", "Cluj", "Iasi", "Timisoara", "StiriRomania", "Romemes", "rojokes", "RoPolitica"]
headlines = collect_headlines(subreddits, total_posts=1000)

with open("romanian_reddit_top_headlines.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["headline", "label"])
    writer.writerows(headlines)

print(f"Total collected: {len(headlines)} headlines from {len(subreddits)} subreddits")


#Clean-up
import pandas as pd
from google.colab import files

#uploaded = files.upload()
df_reddit_hot = pd.read_csv('romanian_reddit_hot.csv', encoding='utf-8')
df_reddit_top = pd.read_csv('romanian_reddit_top.csv', encoding='utf-8')


df_reddit= pd.concat([df_reddit_hot, df_reddit_top], ignore_index=True)
df_reddit = df_reddit[df_reddit["headline"].apply(lambda x: len(str(x).split()) > 3)]

num_rows = len(df_reddit)
print(f"Number of rows left: {num_rows}")
df_reddit.to_csv('romanian_reddit_over_3_words.csv', index=False)

#English text removal
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

def is_english(text):
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

english_titles = df_reddit[df_reddit['headline'].apply(is_english)]['headline']
print("English titles found:")
print(english_titles)

df_reddit = df_reddit[~df_reddit['headline'].apply(is_english)]

num_rows = len(df_reddit)
print(f"Number of rows left: {num_rows}")

df_reddit.to_csv('romanian_reddit.csv', index=False)
