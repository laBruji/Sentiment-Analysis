import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("filtered_dataset.csv")
df_row_count = df.shape[0]
limit = int((3 * df_row_count) / 5)

print("Number of reviews:", df_row_count)

# splitting data for training (60%)
train_text = df.review[:limit]
train_sentiments = df.sentiment[:limit]
test_text = df.review[limit:]
test_sentiments = df.sentiment[limit:]

# CountVectorizer converts to bag of words
cv_obj = CountVectorizer(max_df=0.5, ngram_range=(1, 3))
# transforming train text into vector
train_text_cv = cv_obj.fit_transform(train_text)
test_text_cv = cv_obj.transform(test_text)

features = cv_obj.get_feature_names_out()
print(features)

print("Number of documents/reviews/lines:", train_text_cv.shape[0])
print("Number of words in the vocabulary:", train_text_cv.shape[1])

# Logistic Regression
lr_obj = LogisticRegression(max_iter = 500, random_state = 42)

#fitting model for bag of words and prediction
bow_lr = lr_obj.fit(train_text_cv, train_sentiments)
bow_predict = lr_obj.predict(test_text_cv)

# Accuracy
bow_accur = accuracy_score(test_sentiments, bow_predict)
print("Accuracy:", bow_accur)

bow_report = classification_report(test_sentiments, bow_predict, target_names = ['Positive', 'Negative'], output_dict = True)
bow_report['no. of reviews'] = df_row_count
bow_report['no. of docs'] = train_text_cv.shape[0]
bow_report['no. of words in vocabulary'] = train_text_cv.shape[1]
bow_df = pd.DataFrame(bow_report).transpose()


bow_df.to_csv('BoW-report.csv')
