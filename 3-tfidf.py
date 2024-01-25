import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('filtered_dataset.csv')
df_row_count = df.shape[0] 
limit = int((3*df_row_count) / 5)

print('Number of reviews:', df_row_count)

# splitting data for training (60%)
train_text = df.review[:limit] 
train_sentiments = df.sentiment[:limit]
test_text = df.review[limit:]
test_sentiments = df.sentiment[limit:]

# TfidfVectorizer converts to a matrix
tv_obj = TfidfVectorizer(max_df = 0.5, ngram_range = (1,3))
train_text_tv = tv_obj.fit_transform(train_text)
test_text_tv = tv_obj.transform(test_text)

print("Number of documents/reviews/lines:", train_text_tv.shape[0])
print("Number of words in the vocabulary:", train_text_tv.shape[1])

# Logistic Regression
lr_obj = LogisticRegression(max_iter = 500, random_state = 42)
#fitting model and prediction
tfidf_lr = lr_obj.fit(train_text_tv, train_sentiments)
tfidf_predict = lr_obj.predict(test_text_tv)

# accuracy of model
tfidf_accur = accuracy_score(test_sentiments, tfidf_predict)
print('Accuracy:', tfidf_accur)

# report
tfidf_report = classification_report(test_sentiments,tfidf_predict, target_names = ['Positive', 'Negative'], output_dict = True)
tfidf_report['no. of reviews'] = df_row_count
tfidf_report['no. of docs'] = train_text_tv.shape[0]
tfidf_report['no. of words in vocabulary'] = train_text_tv.shape[1]
tfidf_df = pd.DataFrame(tfidf_report).transpose()

# saving the report
tfidf_df.to_csv('TF-IDF-report.csv')







