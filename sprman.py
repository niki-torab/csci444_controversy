from scipy.stats import spearmanr

manual_rankings = [1, 2, 3, 4, 5,6,7]  
lda_rankings = [2, 1, 4, 3, 5,6,7]    

correlation, _ = spearmanr(manual_rankings, lda_rankings)
print(f"Spearman’s Rank Correlation: {correlation}")
