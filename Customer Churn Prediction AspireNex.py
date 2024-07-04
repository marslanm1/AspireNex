#!/usr/bin/env python
# coding: utf-8

# In[52]:


# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# In[53]:


# Loading dataset
file_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(file_path)


# In[54]:


# Display Ooverview of dataset
print("Dataset Head:")
print(df.head())


# In[55]:


print("\nDataset Description:")
print(df.describe())


# In[56]:


# Handling missing values in TotalCharges column
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)


# In[57]:


# Droping unnecessary columns for churn prediction
df.drop(['customerID'], axis=1, inplace=True)


# In[58]:


# Exploratory Data Analysis (EDA)


# In[59]:


# Distribution plots for numeric variables
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
sns.histplot(df['tenure'], bins=30, kde=True, color='blue', alpha=0.7)
plt.title('Distribution of Tenure')
plt.xlabel('Tenure (months)')
plt.ylabel('Count')

plt.subplot(1, 3, 2)
sns.histplot(df['MonthlyCharges'], bins=30, kde=True, color='red', alpha=0.7)
plt.title('Distribution of Monthly Charges')
plt.xlabel('Monthly Charges')
plt.ylabel('Count')

plt.subplot(1, 3, 3)
sns.histplot(df['TotalCharges'], bins=30, kde=True, color='green', alpha=0.7)
plt.title('Distribution of Total Charges')
plt.xlabel('Total Charges')
plt.ylabel('Count')

plt.tight_layout()
plt.show()


# In[60]:


# Boxplots for categorical variables vs churn
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
sns.boxplot(x='Contract', y='tenure', hue='Churn', data=df)
plt.title('Contract vs Tenure by Churn')

plt.subplot(1, 3, 2)
sns.boxplot(x='InternetService', y='MonthlyCharges', hue='Churn', data=df)
plt.title('Internet Service vs Monthly Charges by Churn')

plt.subplot(1, 3, 3)
sns.boxplot(x='PaymentMethod', y='TotalCharges', hue='Churn', data=df)
plt.title('Payment Method vs Total Charges by Churn')

plt.tight_layout()
plt.show()


# In[61]:


# Count plots for categorical variables
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
sns.countplot(x='gender', hue='Churn', data=df)
plt.title('Gender Count by Churn')

plt.subplot(2, 2, 2)
sns.countplot(x='SeniorCitizen', hue='Churn', data=df)
plt.title('Senior Citizen Count by Churn')

plt.subplot(2, 2, 3)
sns.countplot(x='Dependents', hue='Churn', data=df)
plt.title('Dependents Count by Churn')

plt.subplot(2, 2, 4)
sns.countplot(x='Partner', hue='Churn', data=df)
plt.title('Partner Count by Churn')

plt.tight_layout()
plt.show()


# In[71]:


# Select numeric columns for correlation calculation
numeric_cols = df.select_dtypes(include=np.number)

# Visualize correlations
plt.figure(figsize=(8, 5))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# In[63]:


# Feature Engineering
df['has_internet_service'] = (df['InternetService'] != 'No').astype(int)
df['multiple_services'] = (df['PhoneService'] == 'Yes') & (df['InternetService'] != 'No')


# In[64]:


# Encode categorical variables
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].apply(LabelEncoder().fit_transform)


# In[65]:


# Model Training and Evaluation
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[66]:


# Scaling numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[67]:


# Defining models
models = [
    ('RandomForest', RandomForestClassifier(random_state=42)),
    ('GradientBoosting', GradientBoostingClassifier(random_state=42)),
    ('XGBoost', XGBClassifier(random_state=42)),
    ('LogisticRegression', LogisticRegression(max_iter=1000, random_state=42))
]


# In[68]:


# Training and evaluating each model
for name, model in models:
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name} Accuracy: {accuracy}')


# In[69]:


# Displaying classification report for each model
print(f'Classification Report for {name}:')
print(classification_report(y_test, y_pred))


# In[70]:


# Voting Classifier
voting_clf = VotingClassifier(estimators=models, voting='soft')
voting_clf.fit(X_train_scaled, y_train)
voting_pred = voting_clf.predict(X_test_scaled)
voting_accuracy = accuracy_score(y_test, voting_pred)
print(f'Voting Classifier Accuracy: {voting_accuracy}')

