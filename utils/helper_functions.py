import re
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from typing import Dict
from sklearn.metrics import (precision_score, accuracy_score,
                             recall_score, f1_score,
                             make_scorer)
import pandas as pd
import numpy as np
from typing import List
from sklearn.base import BaseEstimator
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt_tab')
from nltk.stem import WordNetLemmatizer
from transformers import PreTrainedTokenizer, TFBertForSequenceClassification

def customize_bar(position: str, axes, 
                  values_font=12, pct=False, round_to=0) -> None:
    """
    Function, which customizes bar chart
    Takes axes object and:
        - gets rid of spines
        - modifies ticks
        - adds value above each bar
    Parameters:
        - position(str): modify the bar depending on how the
        bars are positioned: vertically or horizontally
    Return: None
    """
    # get rid of spines
    for spine in axes.spines.values():
        spine.set_visible(False)
    # modify ticklabels
    if position == "v":
        axes.set_yticks([])
        for tick in axes.get_xticklabels():
            tick.set_rotation(0)
    if position == "h":
        axes.set_xticks([])
        for tick in axes.get_yticklabels():
            tick.set_rotation(0)
    # add height value above each bar
    for bar in axes.patches:
        if bar.get_width() == 0:
            continue
        if position == "v":
            text_location = (bar.get_x() + bar.get_width()/2,
                             bar.get_height() + 1/100*bar.get_height())
            value = bar.get_height()
            location = "center"
        elif position == "h":
            text_location = (bar.get_width(),
                             bar.get_y() + bar.get_height() / 2)
            value = bar.get_width()
            location = "left"
        if pct:
            value = f"{round(value * 100, round_to)}%"
        elif round_to == 0:
            value = str(int(value))
        else:
            value = str(round(value, round_to))
        axes.text(text_location[0],
                text_location[1],
                str(value),
                fontsize=values_font,
                ha=location)
        

def box_hist_comparison(column: str, target: str, data: pd.DataFrame, title: str = None) -> None:
    """
    Generates a side-by-side comparison of a histogram and a boxen plot for a specified column, 
    grouped by the target column in the given DataFrame.

    Args:
    column (str): The column name to plot.
    target (str): The target column used for grouping.
    data (pd.DataFrame): The DataFrame containing the data to plot.
    title (str, optional): The title of the plots. Defaults to None.
    """
    _, ax = plt.subplots(1, 2, figsize=(15, 5))
    sns.histplot(x=column, kde=True, ax=ax[0],
                 hue=target, data=data)
    sns.boxenplot(x=target, y=column, data=data, ax=ax[1])

def number_of_numbers(text: str) -> int:
    """
    Counts the number of digits in the given text.

    Args:
    text (str): The input text to analyze.

    Returns:
    int: The number of digits found in the text.
    """
    return len(re.findall(r'\d', text))

def numbers_ratio(text: str) -> float:
    """
    Calculates the ratio of digits to total characters in the given text.

    Args:
    text (str): The input text to analyze.

    Returns:
    float: The ratio of digits to total characters in the text.
    """
    numbers = number_of_numbers(text)
    return numbers/len(text)

def preprocess_text(text: str) -> List[str]:
    """
    Preprocesses the input text by performing several text cleaning operations, 
    including converting to lowercase, removing URLs, usernames, non-alphanumeric characters, 
    stopwords, and lemmatizing words.

    Args:
    text (str): The input text to preprocess.

    Returns:
    List[str]: A list of processed tokens (words).
    """
    wordLemm = WordNetLemmatizer()
    text = text.lower()

    text = re.sub(r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)", "url", text)

    text = re.sub('@[^\s]+', "user", text)
    text = re.sub(r'#\S+', ' hashtag', text)

    text = re.sub("[^a-zA-Z0-9]", " ", text)

    text = re.sub(r"(.)\1\1+", r"\1\1", text)

    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]

    tokens = [wordLemm.lemmatize(word) for word in tokens]
    return tokens

def preprocess_text_bert(text: str) -> List[str]:
    """
    Preprocesses the input text by performing several text cleaning operations, 
    including converting to lowercase, removing URLs, usernames, non-alphanumeric characters, 
    stopwords, and lemmatizing words.

    Args:
    text (str): The input text to preprocess.

    Returns:
    List[str]: A list of processed tokens (words).
    """
    wordLemm = WordNetLemmatizer()
    text = text.lower()

    text = re.sub(r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)", "", text)

    text = re.sub('@[^\s]+', "", text)

    text = re.sub("[^a-zA-Z0-9]", " ", text)

    text = re.sub(r"(.)\1\1+", r"\1\1", text)

    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]

    tokens = [wordLemm.lemmatize(word) for word in tokens]
    return tokens

def plot_wordcloud(data: pd.DataFrame, sentiment: str) -> None:
    """
    Generates and displays a word cloud for the specified sentiment from the input DataFrame.

    Args:
    data (pd.DataFrame): The DataFrame containing the text data.
    sentiment (str): The sentiment to filter by.
    """
    text = ' '.join(data[data['sentiment'] == sentiment]['text'])
    wordcloud = WordCloud(width=800, height=400, max_font_size=100).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {sentiment} Sentiment')
    plt.show()

def avg_word(text: str) -> float:
    """
    Calculates the average word length in the given text.

    Args:
    text (str): The input text to analyze.

    Returns:
    float: The average word length in the text.
    """
    words = text.split()
    return (sum(len(word) for word in words)/len(words))

def contains_special_characters(text: str) -> bool:
    """
    Checks if the text contains any special characters (non-alphanumeric characters).

    Args:
    text (str): The input text to analyze.

    Returns:
    bool: True if the text contains special characters, False otherwise.
    """
    return bool(re.search(r'[^a-zA-Z0-9\s]', text)) 

def contains_url(text: str) -> bool:
    """
    Checks if the text contains a URL.

    Args:
    text (str): The input text to analyze.

    Returns:
    bool: True if the text contains a URL, False otherwise.
    """
    return bool(re.search(r'https?://[^\s]+', text))

def contains_hashtag(text: str) -> bool:
    """
    Checks if the text contains a hashtag.

    Args:
    text (str): The input text to analyze.

    Returns:
    bool: True if the text contains a hashtag, False otherwise.
    """
    return bool(re.search(r'#\S+', text))

def exclamations_ratio(text: str) -> float:
    """
    Calculates the ratio of exclamation marks to total characters in the text.

    Args:
    text (str): The input text to analyze.

    Returns:
    float: The ratio of exclamation marks to total characters in the text.
    """
    return text.count('!') / len(text)

def consecutive_exclamations(text: str) -> int:
    """
    Counts the number of consecutive exclamation marks in the text.

    Args:
    text (str): The input text to analyze.

    Returns:
    int: The number of occurrences of consecutive exclamation marks.
    """
    return len(re.findall(r'!+', text))

def get_scores(model: BaseEstimator, X: np.ndarray,
               y: np.ndarray, fit: bool = True) -> Dict[str, float]:
    """
    Compute performance scores on the data.

    Parameters:
    model (BaseEstimator): The machine learning model
    X (np.ndarray): The feature matrix used.
    y (np.ndarray): The target vector used.
    fit (bool): If True, the model will be fitted to the data. Default is True.

    Returns:
    Dict[str, float]: A dictionary containing accuracy, recall, precision, and f1 scores.
    """
    if fit:
        model.fit(X, y)

    model_predict = model.predict(X)

    recall = make_scorer(recall_score, average="weighted")
    precision = make_scorer(precision_score, average="weighted")
    f1 = make_scorer(f1_score, average="weighted")

    scores = {
        "accuracy": accuracy_score(y, model_predict),
        "recall": recall_score(y, model_predict, average="weighted"),
        "precision": precision_score(y, model_predict, average="weighted"),
        "f1": f1_score(y, model_predict, average="weighted")
    }

    return scores


def load_glove_embeddings(file_path: str) -> Dict[str, np.ndarray]:
    """
    Loads GloVe word embeddings from a specified file and returns them as a dictionary.

    Args:
    file_path (str): The path to the GloVe embeddings file.

    Returns:
    Dict[str, np.ndarray]: A dictionary where keys are words and values are their corresponding 
                            word embeddings as numpy arrays.
    """
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def get_predictions(data: pd.DataFrame, 
                    model: TFBertForSequenceClassification, 
                    tokenizer: PreTrainedTokenizer) -> List[np.ndarray]:
    batch_size = 32
    num_batches = len(data) // batch_size
    preds = []
    encodings = tokenizer(list(data["preprocessed_text"]), 
                          max_length=128, 
                          padding='max_length', 
                          truncation=True, return_tensors="tf")
    for i in range(num_batches):
        batch = encodings[i*batch_size:(i+1)*batch_size]
        batch_preds = model(batch)
        preds.extend(batch_preds.logits.numpy())
    return preds