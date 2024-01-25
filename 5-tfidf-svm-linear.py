import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
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
tv_obj = TfidfVectorizer(max_df = 1, ngram_range = (1,3))
train_text_tv = tv_obj.fit_transform(train_text)
test_text_tv = tv_obj.transform(test_text)

print("Number of documents/reviews/lines:", train_text_tv.shape[0])
print("Number of words in the vocabulary:", train_text_tv.shape[1])

# SVM Machine
svm_obj = LinearSVC(loss = 'hinge', C = 1.5)

#fitting model and prediction
tfidf_svm = svm_obj.fit(train_text_tv, train_sentiments)
tfidf_svm_predict = svm_obj.predict(test_text_tv)

# accuracy of model
tfidf_svm_accur = accuracy_score(test_sentiments, tfidf_svm_predict)
print("Accuracy:", tfidf_svm_accur)


# report
tfidf_svm_report = classification_report(test_sentiments,tfidf_svm_predict, target_names = ['Positive', 'Negative'], output_dict = True)
tfidf_svm_report['no. of reviews'] = df_row_count
tfidf_svm_report['no. of docs'] = train_text_tv.shape[0]
tfidf_svm_report['no. of words in vocabulary'] = train_text_tv.shape[1]
tfidf_df = pd.DataFrame(tfidf_svm_report).transpose()

# saving the report
tfidf_df.to_csv('TF-IDF-svm-report.csv')







