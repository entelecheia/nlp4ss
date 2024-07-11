# 2.1 Text Cleaning, Normalization, and Representation

## 1. Introduction to Text Preprocessing

Text preprocessing is a crucial step in the Natural Language Processing (NLP) pipeline, especially for social science applications. It involves transforming raw text data into a clean, standardized format that can be easily analyzed by machine learning algorithms.

The importance of preprocessing in NLP cannot be overstated:

- It helps remove noise and irrelevant information from the text
- It standardizes the text, making it easier for algorithms to process
- It can significantly improve the performance of downstream NLP tasks

Common preprocessing steps include:

1. Text cleaning
2. Lowercase conversion
3. Tokenization
4. Stop word removal
5. Stemming or lemmatization
6. Text representation

Let's explore each of these steps in detail.

## 2. Text Cleaning

Text cleaning involves removing or replacing elements in the text that are not relevant to the analysis. This step is particularly important when dealing with web-scraped data or social media content.

### Removing HTML tags and special characters

HTML tags and special characters can interfere with text analysis. We can use regular expressions to remove them.

```python
import re

def clean_html(text):
    clean_text = re.sub('<.*?>', '', text)  # Remove HTML tags
    clean_text = re.sub('[^a-zA-Z0-9\s]', '', clean_text)  # Remove special characters
    return clean_text

html_text = "<p>This is a <b>sample</b> text with HTML &amp; special characters!</p>"
cleaned_text = clean_html(html_text)
print(cleaned_text)
# Output: This is a sample text with HTML  special characters
```

### Handling URLs and email addresses

For many NLP tasks, URLs and email addresses can be considered noise. We can remove or replace them with placeholders.

```python
def remove_urls_emails(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '[URL]', text)
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    return text

sample_text = "Check out our website at https://example.com or email us at info@example.com"
processed_text = remove_urls_emails(sample_text)
print(processed_text)
# Output: Check out our website at [URL] or email us at [EMAIL]
```

### Dealing with numbers and dates

Depending on the analysis, numbers and dates might need to be standardized or removed.

```python
def standardize_numbers(text):
    text = re.sub(r'\d+', '[NUM]', text)
    return text

sample_text = "The event took place on 15/04/2023 and 1500 people attended."
processed_text = standardize_numbers(sample_text)
print(processed_text)
# Output: The event took place on [NUM]/[NUM]/[NUM] and [NUM] people attended.
```

### Removing or replacing emojis and emoticons

Emojis and emoticons can carry sentiment information but may need to be standardized for consistent analysis.

```python
import emoji

def replace_emojis(text):
    return emoji.demojize(text)

sample_text = "I love this movie! 😍👍"
processed_text = replace_emojis(sample_text)
print(processed_text)
# Output: I love this movie! :smiling_face_with_heart_eyes::thumbs_up:
```

## 3. Lowercase Conversion

Converting text to lowercase helps standardize the text and reduce the vocabulary size. However, it's important to consider when case information might be relevant (e.g., for named entity recognition).

```python
def to_lowercase(text):
    return text.lower()

sample_text = "The Quick Brown Fox Jumps Over The Lazy Dog"
lowercased_text = to_lowercase(sample_text)
print(lowercased_text)
# Output: the quick brown fox jumps over the lazy dog
```

## 4. Tokenization

Tokenization is the process of breaking text into smaller units, typically words or sentences. It's a fundamental step in many NLP tasks.

### Word tokenization

```python
from nltk.tokenize import word_tokenize

def tokenize_words(text):
    return word_tokenize(text)

sample_text = "This is a sample sentence for word tokenization."
tokens = tokenize_words(sample_text)
print(tokens)
# Output: ['This', 'is', 'a', 'sample', 'sentence', 'for', 'word', 'tokenization', '.']
```

### Sentence tokenization

```python
from nltk.tokenize import sent_tokenize

def tokenize_sentences(text):
    return sent_tokenize(text)

sample_text = "This is the first sentence. And here's another one. What about a question?"
sentences = tokenize_sentences(sample_text)
print(sentences)
# Output: ['This is the first sentence.', "And here's another one.", 'What about a question?']
```

## 5. Stop Word Removal

Stop words are common words that usually don't contribute much to the meaning of a text. Removing them can help reduce noise in the data.

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)

sample_text = "This is a sample sentence with stop words."
processed_text = remove_stopwords(sample_text)
print(processed_text)
# Output: This sample sentence stop words .
```

## 6. Stemming

Stemming reduces words to their root form, which can help in reducing vocabulary size and grouping similar words.

```python
from nltk.stem import PorterStemmer

def stem_words(text):
    ps = PorterStemmer()
    words = word_tokenize(text)
    return ' '.join([ps.stem(word) for word in words])

sample_text = "The runner runs quickly through the running track"
stemmed_text = stem_words(sample_text)
print(stemmed_text)
# Output: The runner run quickli through the run track
```

## 7. Lemmatization

Lemmatization is similar to stemming but produces more linguistically correct root forms.

```python
from nltk.stem import WordNetLemmatizer

def lemmatize_words(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(word) for word in words])

sample_text = "The children are playing with their toys"
lemmatized_text = lemmatize_words(sample_text)
print(lemmatized_text)
# Output: The child are playing with their toy
```

## 8. Text Representation Techniques

After preprocessing, we need to represent the text in a format that machine learning algorithms can understand.

### Bag-of-Words (BoW) model

```python
from sklearn.feature_extraction.text import CountVectorizer

def bow_representation(texts):
    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform(texts)
    return bow_matrix, vectorizer.get_feature_names_out()

texts = ["This is the first document.", "This document is the second document.", "And this is the third one."]
bow_matrix, feature_names = bow_representation(texts)
print(feature_names)
print(bow_matrix.toarray())
```

### Term Frequency-Inverse Document Frequency (TF-IDF)

```python
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_representation(texts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer.get_feature_names_out()

texts = ["This is the first document.", "This document is the second document.", "And this is the third one."]
tfidf_matrix, feature_names = tfidf_representation(texts)
print(feature_names)
print(tfidf_matrix.toarray())
```

### Word Embeddings

Word embeddings provide dense vector representations of words, capturing semantic relationships.

```python
import gensim.downloader as api

def get_word_embedding(word, model_name='glove-wiki-gigaword-100'):
    model = api.load(model_name)
    return model[word]

word = "sociology"
embedding = get_word_embedding(word)
print(f"Embedding for '{word}':")
print(embedding[:10])  # Printing first 10 dimensions
```

This comprehensive overview covers the main aspects of text cleaning, normalization, and representation, with code examples for each step. These techniques form the foundation for more advanced NLP tasks in social science research, such as sentiment analysis, topic modeling, and text classification.
