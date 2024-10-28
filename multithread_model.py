import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from textblob import TextBlob
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
trainingSet = pd.read_csv("./data/train.csv")
testingSet = pd.read_csv("./data/test.csv")

# Parallelized Sentiment Analysis
def compute_sentiment(text):
    blob = TextBlob(str(text))
    return blob.sentiment.polarity, blob.sentiment.subjectivity

def add_sentiment_features(df):
    with ProcessPoolExecutor() as executor:
        # Map sentiment features across the data
        text_sentiment = list(executor.map(compute_sentiment, df['Text'].fillna('')))
        summary_sentiment = list(executor.map(compute_sentiment, df['Summary'].fillna('')))
    
    # Unpack and add to DataFrame
    df['Text_Polarity'], df['Text_Subjectivity'] = zip(*text_sentiment)
    df['Summary_Polarity'], df['Summary_Subjectivity'] = zip(*summary_sentiment)
    return df

# Process with multiprocessing
trainingSet = add_sentiment_features(trainingSet)
testingSet = add_sentiment_features(testingSet)

# Fill missing values in Helpfulness and Text features
trainingSet['Helpfulness'] = trainingSet['HelpfulnessNumerator'] / trainingSet['HelpfulnessDenominator']
trainingSet['Helpfulness'] = trainingSet['Helpfulness'].fillna(0)
testingSet['Helpfulness'] = testingSet['HelpfulnessNumerator'] / testingSet['HelpfulnessDenominator']
testingSet['Helpfulness'] = testingSet['Helpfulness'].fillna(0)

# Vectorize the text columns with parallel processing
vectorizer = TfidfVectorizer(max_features=500, n_jobs=-1)  # n_jobs=-1 uses all available cores
train_text_features = vectorizer.fit_transform(trainingSet['Text'].fillna(''))
test_text_features = vectorizer.transform(testingSet['Text'].fillna(''))

# Combine all features
train_features = pd.concat([trainingSet[['Helpfulness', 'Text_Polarity', 'Text_Subjectivity', 'Summary_Polarity', 'Summary_Subjectivity']], 
                            pd.DataFrame(train_text_features.toarray())], axis=1)
test_features = pd.concat([testingSet[['Helpfulness', 'Text_Polarity', 'Text_Subjectivity', 'Summary_Polarity', 'Summary_Subjectivity']], 
                           pd.DataFrame(test_text_features.toarray())], axis=1)

# Split training data for model evaluation
X_train, X_val, y_train, y_val = train_test_split(train_features, trainingSet['Score'], test_size=0.25, random_state=42)

# Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions and evaluation
y_val_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
print("Accuracy on validation set:", accuracy)

# # Confusion matrix
# cm = confusion_matrix(y_val, y_val_pred)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.title("Confusion Matrix")
# plt.show()

# Make predictions on test set and check row count before saving
testingSet['Score'] = model.predict(test_features)

# Confirm row count
if len(testingSet) == 212192:
    print("Row count matches. Saving to submission.csv.")
    submission = testingSet[['Id', 'Score']]
    submission.to_csv("./data/submission.csv", index=False)
else:
    print(f"Row count mismatch: Expected 212192, got {len(testingSet)}.")
