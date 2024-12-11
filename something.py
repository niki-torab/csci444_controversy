from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Sentences to analyze
sentence1 = "someone killed apple with knife"
sentence2 = "somebody was killed with knife"

# Analyze the sentiment for both sentences
score1 = analyzer.polarity_scores(sentence1)
score2 = analyzer.polarity_scores(sentence2)

# Print the results
print(f"Sentence 1: \"{sentence1}\"")
print(f"Scores: {score1}\n")

print(f"Sentence 2: \"{sentence2}\"")
print(f"Scores: {score2}\n")
