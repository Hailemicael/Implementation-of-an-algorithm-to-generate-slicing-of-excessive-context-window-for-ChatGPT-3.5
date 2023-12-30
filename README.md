# NLP Second Assignment
Implementation of an algorithm to generate slicing of excessive context window for ChatGPT 3.5

## Overview

This project implements an algorithm for generating slicing of an excessive context window for ChatGPT 3.5. The slicing process is based on a specified pipeline to handle inputs that exceed the standard size of the context window.

## Context Window Slicing Algorithm

The method is based on the following pipeline:

- When the input is below the standard size of the context window (128 MB), it is passed "as it is" to the Language Model (LM).
- When the input is above the standard size, it is subdivided into a finite number of slices, each of a size that fits the context window. These slices sum to a number N greater than or equal to the size of the input length.

## Text Preprocessing

The algorithm utilizes natural language processing techniques, including:

- Tokenization
```python
    words = nltk.word_tokenize(text)
    words = [word.lower() for word in words if word.isalnum()]
```
- Stopword elimination
```python
  stop_words = set(stopwords.words('english'))
  words = [word for word in words if word not in stop_words]
```
- Lemmatization
```python
  lemmatizer = WordNetLemmatizer()
  words = [lemmatizer.lemmatize(word) for word in words]
```
- Cosine similarity based on TF-IDF representation of text slices

## Cosine Similarity and Threshold Logic

The cosine similarity is employed to compare two adjacent slices based on the bag of words constructed by the usual pipeline of stopword elimination, stemming/lemmatization, and count of occurrences weighted on the length of the document after the preprocessing steps.

### Cosine Similarity and Threshold Example

Here's how cosine similarity and threshold logic are implemented in Python:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Example slices
slice1 = "This is the first slice."
slice2 = "Another slice with some different content."

# Tokenize and preprocess slices
processed_slice1 = preprocess_text(slice1)
processed_slice2 = preprocess_text(slice2)

# Convert slices to TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([processed_slice1, processed_slice2])

# Calculate cosine similarity
cosine_distance = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

# Set threshold (e.g., 20%)
threshold = 0.2

# Check if slices are "different enough" based on the threshold
if cosine_distance > threshold:
    print("The slices are different enough.")
else:
    print("The slices are too similar.")
```


# Files
Input File: input.txt
This text file contains the input text to be processed.

Output Slices: slices_output.txt
The generated slices are saved in this text file./due to the size, the best option is to display by file/


# Results
The algorithm generates slices based on the specified criteria and provides coverage for inputs exceeding the context window size.
