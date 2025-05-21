# AI Chat Log Summarizer

**AI Chat Log Summarizer** is a Python-based tool that reads `.txt` chat logs between a user and an AI, parses the conversation, and generates a clean summary including:
- Number of messages from the user and AI
- Total number of exchanges
- Most common keywords extracted using **TF-IDF**
- Nature of the conversation based on keyword topics

This project demonstrates basic **Natural Language Processing (NLP)** and keyword extraction using `scikit-learn` and `nltk`.

---

## Features

-  Parses chat logs into user and AI messages
-  Cleans messages (removes punctuation, digits, stopwords)
-  Extracts top keywords using TF-IDF
-  Generates a text summary
-  Saves cleaned logs and summary to files

Install the required libraries:

- pip install scikit-learn nltk
- 
## Run:

-python summarizer.py

## Summary:
- Total exchanges: 4
- User messages: 2
- AI messages: 2
- Conversation focus: python and related topics
- Top keywords (TF-IDF): python, data, science, ai, web




