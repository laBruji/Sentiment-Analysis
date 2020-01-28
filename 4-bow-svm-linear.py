import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
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

# CountVectorizer converts to bag of words
cv_obj = CountVectorizer(max_df = 1, ngram_range = (1,3))
#transforming train text into vector
train_text_cv = cv_obj.fit_transform(train_text)
test_text_cv = cv_obj.transform(test_text)

print("Number of documents/reviews/lines:", train_text_cv.shape[0])
print("Number of words in the vocabulary:", train_text_cv.shape[1])

# SVM Machine
svm_obj = LinearSVC(loss = 'hinge', C = 0.01)

#fitting model for bag of words and prediction
bow_svm = svm_obj.fit(train_text_cv, train_sentiments)
bow_svm_predict = svm_obj.predict(test_text_cv)

# Accuracy
bow_svm_accur = accuracy_score(test_sentiments, bow_svm_predict)
print("Accuracy:", bow_svm_accur)

bow_svm_report = classification_report(test_sentiments, bow_svm_predict, target_names = ['Positive', 'Negative'], output_dict = True)
bow_svm_report['no. of reviews'] = df_row_count
bow_svm_report['no. of docs'] = train_text_cv.shape[0]
bow_svm_report['no. of words in vocabulary'] = train_text_cv.shape[1]
bow_df = pd.DataFrame(bow_svm_report).transpose()

bow_df.to_csv('BoW-SVM-report.csv')







