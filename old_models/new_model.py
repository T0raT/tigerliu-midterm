import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
trainingSet = pd.read_csv("./data/train.csv")
testingSet = pd.read_csv("./data/test.csv")

# Sentiment feature extraction for training data only
def add_sentiment_features(df):
    df['Text_Polarity'] = df['Text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df['Text_Subjectivity'] = df['Text'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
    df['Summary_Polarity'] = df['Summary'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df['Summary_Subjectivity'] = df['Summary'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
    return df

trainingSet = add_sentiment_features(trainingSet)

# Fill missing values in Helpfulness features
trainingSet['Helpfulness'] = trainingSet['HelpfulnessNumerator'] / trainingSet['HelpfulnessDenominator']
trainingSet['Helpfulness'] = trainingSet['Helpfulness'].fillna(0)

# Since `test.csv` lacks the 'Text' and 'Summary' columns, we'll only use common features
testingSet['Helpfulness'] = testingSet['HelpfulnessNumerator'] / testingSet['HelpfulnessDenominator']
testingSet['Helpfulness'] = testingSet['Helpfulness'].fillna(0)

# Vectorize the text columns in training set only
vectorizer = TfidfVectorizer(max_features=500)  # Limit features for efficiency
train_text_features = vectorizer.fit_transform(trainingSet['Text'].fillna(''))

# Combine all features for training set
train_features = pd.concat([trainingSet[['Helpfulness', 'Text_Polarity', 'Text_Subjectivity', 'Summary_Polarity', 'Summary_Subjectivity']], 
                            pd.DataFrame(train_text_features.toarray())], axis=1)

# Define testing features with only available non-text columns
test_features = testingSet[['Helpfulness']].copy()

# Split training data for model evaluation
X_train, X_val, y_train, y_val = train_test_split(train_features, trainingSet['Score'], test_size=0.25, random_state=42)

# Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions and evaluation
y_val_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
print("Accuracy on validation set:", accuracy)

# Confusion matrix
cm = confusion_matrix(y_val, y_val_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Generate predictions for submission
# Using only 'Helpfulness' here for the test set since 'Text' is unavailable
testingSet['Score'] = model.predict(test_features)

# Confirm row count and save to submission
if len(testingSet) == 212192:
    print("Row count matches. Saving to submission.csv.")
    submission = testingSet[['Id', 'Score']]
    submission.to_csv("./data/submission.csv", index=False)
else:
    print(f"Row count mismatch: Expected 212192, got {len(testingSet)}.")
