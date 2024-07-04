#!/usr/bin/env python
# coding: utf-8

# In[42]:


# Importing necessary libraries
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


# In[43]:


# Step 1: Loading dataset with proper encoding
file_path = 'spam.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1')


# In[44]:


# Step 2: Inspecting the dataset
print(data.head())


# In[45]:


# Step 3: Assigning column names
data.columns = ['label', 'message', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']


# In[46]:


# Step 4: Dropping unnecessary columns
data = data[['label', 'message']]


# In[47]:


# Step 5: Checking for NaN values
nan_counts = data.isnull().sum()
print("NaN counts per column:\n", nan_counts)

# Dropping rows with NaN values
data = data.dropna()


# In[49]:


# Step 6: Preprocessign the text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

data['message'] = data['message'].apply(preprocess_text)


# In[50]:


# Step 7: Converting labels to binary values
data['label'] = data['label'].map({'ham': 0, 'spam': 1})


# In[51]:


# Step 8: Splitting the data into training and testing sets
X = data['message']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[52]:


# Step 9: Vectorizing the text data
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# In[53]:


# Step 10: Training an SVM classifier
svm_clf = SVC(kernel='linear', probability=True)
svm_clf.fit(X_train_tfidf, y_train)


# In[54]:


# Step 11: Making predictions on the test set
y_pred = svm_clf.predict(X_test_tfidf)


# In[55]:


# Step 12: Evaluating the model
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))


# In[56]:


# Step 13: Real-time spam message detector function
def detect_spam(message):
    # Preprocessing the message
    preprocessed_message = preprocess_text(message)
    # Vectorizing the message
    message_tfidf = vectorizer.transform([preprocessed_message])
    # Predicting using the trained model
    prediction = svm_clf.predict(message_tfidf)
    # Mapping the prediction to the label
    label = 'spam' if prediction[0] == 1 else 'ham'
    return label

# Example usage of real-time spam message detector
new_message = "England v Macedonia - dont miss the goals/team news. Txt ur national team to 87077 eg ENGLAND to 87077 Try:WALES, SCOTLAND 4txt/Ì¼1.20 POBOXox36504W45WQ 16+"
print(f"The message '{new_message}' is classified as: {detect_spam(new_message)}")

