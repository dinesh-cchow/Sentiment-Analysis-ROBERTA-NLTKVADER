
# Sentiment Analysis with NLTK Vader and RoBERTa

## Introduction
This notebook demonstrates sentiment analysis on a dataset of reviews using NLTK's Vader tool and the RoBERTa language model.

## Setup
First, we import necessary libraries:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
```

## Data Loading
Load and preview the dataset:
```python
df = pd.read_csv("data/reviews.csv").head(500)
df.head()
```

## Sentiment Analysis with Vader
Initialize Vader and analyze sentiment scores:
```python
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
```
Example analysis:
```python
sia.polarity_scores("Sample review text")
```

## Sentiment Analysis with RoBERTa
Utilize RoBERTa for sentiment analysis:
```python
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
```
Example encoding and analysis:
```python
encoded_input = tokenizer("Sample review text", return_tensors='pt')
output = model(**encoded_input)
```

## Results and Visualization
![image](https://github.com/Tanvik-VP/Sentiment-Analysis-with-NLTK-Vader-and-RoBERTa/assets/77459265/38464eb3-65e7-4f25-86ce-fab7715680a0)


## Conclusion
We can see that the Roberat LLM model has high confidence in finding sentiment than regular NLTK model
