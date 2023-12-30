# NLP SECOND ASSIGNEMTN 
implement an algorithm to generate slicing of excessive context window for ChatGPT 3.5

## Overview

This project implements an algorithm for generating slicing of an excessive context window for ChatGPT 3.5. The slicing process is based on a specified pipeline to handle inputs that exceed the standard size of the context window.

## Context Window Slicing Algorithm

The method is based on the following pipeline:

- When the input is below the standard size of the context window (128 MB), it is passed "as it is" to the Language Model (LM).
- When the input is above the standard size, it is subdivided into a finite number of slices, each of a size that fits the context window. These slices sum to a number N greater than or equal to the size of the input length.

## Text Preprocessing

The algorithm utilizes natural language processing techniques, including:

- Tokenization
- Stopword elimination
- Lemmatization
- Cosine similarity based on TF-IDF representation of text slices

## Cosine Similarity and Threshold Logic

The cosine similarity is employed to compare two adjacent slices based on the bag of words constructed by the usual pipeline of stopword elimination, stemming/lemmatization, and count of occurrences weighted on the length of the document after the preprocessing steps.

### Cosine Similarity and Threshold Example

Here's a simple example of how cosine similarity and threshold logic might be implemented in Python:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Example slices
slice1 = "This is the first slice."
slice2 = "Another slice with some different content."

# Tokenize and preprocess slices
processed_slice1 = preprocess_text(slice1)
processed_slice2 = preprocess_text(slice2)

