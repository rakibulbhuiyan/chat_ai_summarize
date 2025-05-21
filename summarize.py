import os
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

def parsing_chat(path):
    user_msg = []
    ai_msg = []
    with open(path,"r",encoding='utf-8') as file:
        for line in file:
            if line.startswith("User:"):
                user_msg.append(line[5:].strip())
            elif line.startswith("AI:"):
                ai_msg.append(line[3:].strip())
    return user_msg, ai_msg



def clean_txt(text):
    text = re.sub(r'\d+','',text) # digit remove
    text = re.sub(r'\s+',' ',text).strip() # digit remove
    text = text.translate(str.maketrans('','',string.punctuation)) # remove punctuations
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text

def extract_words(clean_msg):
    tf_idf = TfidfVectorizer()
    tf_metrix = tf_idf.fit_transform(clean_msg)
    summed_scores  = tf_metrix.sum(axis=0)
    word_score = []
    for word, indx in tf_idf.vocabulary_.items():
        score = summed_scores[0, indx]
        word_score.append((word, score))

    sorted_keywords = sorted(word_score, key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_keywords]

def generate_summary(user_msg, ai_msg, keywords):
    return f"""
            Summary:
            - Total exchanges: {len(user_msg) + len(ai_msg)}
            - User messages: {len(user_msg)}
            - AI messages: {len(ai_msg)}
            - Conversation focus: {keywords[0]} and related topics
            - Top keywords (TF-IDF): {', '.join(keywords)}
            """.strip()



if __name__ == "__main__":
    path = "A.txt"
    user_msg, ai_msg = parsing_chat(path)
    print(user_msg)
    print(ai_msg)
    all_msg = user_msg + ai_msg

    # clean the message(like am,is are)
    clean_msg = [ clean_txt(msg) for msg in all_msg ]
    print(clean_msg)

    # save in a file after clean msg
    with open("cleaned_messages.txt", "w", encoding="utf-8") as f:
        for line in clean_msg:
            f.write(line + "\n")
    
    keywords = extract_words(clean_msg)
    summary = generate_summary(user_msg, ai_msg, keywords)
    # print(summary)

    with open("summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)