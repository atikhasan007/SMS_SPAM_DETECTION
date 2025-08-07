# 📩 SMS Spam Detection using Machine Learning

A machine learning project to classify SMS messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) and multiple supervised learning algorithms.

## 📁 Dataset

- **Name**: SMS Spam Collection
- **Source**: [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Samples**: 5,572 SMS messages labeled as `ham` or `spam`.

## ✅ Key Features

- Data cleaning: Lowercasing, punctuation & number removal, stopword removal
- Text preprocessing using NLTK
- Feature extraction:
  - Bag of Words (CountVectorizer)
  - TF-IDF (TfidfVectorizer)
- Tried multiple classification models:
  - ✅ Multinomial Naive Bayes
  - ✅ Logistic Regression
  - ✅ Support Vector Machine (SVM)
  - ✅ Random Forest Classifier
  - ✅ K-Nearest Neighbors (KNN)
  - ✅ Decision Tree Classifier
- Model evaluation using:
  - Accuracy, Precision
  - Confusion Matrix
- Visualization:
  - WordCloud for spam and ham messages
  - Barplot of most common spam words

## 🔧 Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud
