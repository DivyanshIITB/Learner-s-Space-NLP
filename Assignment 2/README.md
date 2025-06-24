# Assignment 2.1 Solution: Text Vectorization Implementation

## Understanding and Summary

This assignment involved implementing the TF-IDF algorithm manually and comparing the results with scikit-learn’s `CountVectorizer` and `TfidfVectorizer` on a simple corpus.

### Comparison of Word Scores

1. **CountVectorizer** simply counts word occurrences in each document. As a result, common words like "the" receive high scores, even though they are not particularly informative. It does not consider how frequently a word appears across all documents.

2. **TfidfVectorizer** reduces the importance of common words by assigning lower scores to words that appear in many documents. This is achieved using Inverse Document Frequency (IDF), which penalizes high-frequency words.

3. **Manual TF-IDF Implementation** closely mirrors the behavior of `TfidfVectorizer`. Words that appear in only one or a few documents (like "celestial" or "satellite") receive high TF-IDF scores, while common words (like "the" or "is") receive low scores.

### Explanation of Score Differences for Common Words

Common words such as "the" appear in nearly every document. In TF-IDF, the IDF component becomes very small for such words, reducing their overall TF-IDF score. This helps the model focus on more informative and distinguishing words. On the other hand, `CountVectorizer` does not account for document frequency, so these common words are still counted with high weight, which may introduce noise in feature representation.

### Conclusion

TF-IDF provides a more meaningful representation of text data by emphasizing unique and informative terms while down-weighting common, non-discriminative words. The manual TF-IDF implementation produced similar trends to scikit-learn’s `TfidfVectorizer`, confirming the correctness of the logic.
