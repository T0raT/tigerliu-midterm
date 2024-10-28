# Importing Libraries
import pickle
import pandas as pd
import seaborn as sns
from os.path import exists
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Loading the Files
trainingSet = pd.read_csv("./data/train.csv")
testingSet = pd.read_csv("./data/test.csv")

print("train.csv shape is ", trainingSet.shape)
print("test.csv shape is ", testingSet.shape)

# Add Features Function
def add_features_to(df):
    df['Helpfulness'] = df['HelpfulnessNumerator'] / df['HelpfulnessDenominator']
    df['Helpfulness'] = df['Helpfulness'].fillna(0)
    return df

# Load the feature extracted files if already generated
if exists('./data/X_train.csv'):
    X_train = pd.read_csv("./data/X_train.csv")
if exists('./data/X_submission.csv'):
    X_submission = pd.read_csv("./data/X_submission.csv")
else:
    # Process the DataFrame
    train = add_features_to(trainingSet)
    # Ensure that the merge only keeps rows from the test set without duplicating rows
    X_submission = pd.merge(testingSet[['Id']], train, on='Id', how='left')
    X_submission = X_submission.drop(columns=['Score_x'], errors='ignore')  # Keep only relevant columns
    X_submission = X_submission.rename(columns={'Score_y': 'Score'})  # Rename if necessary

    X_submission = X_submission.drop_duplicates(subset='Id')

    X_train = train[train['Score'].notnull()]
    X_submission.to_csv("./data/X_submission.csv", index=False)
    X_train.to_csv("./data/X_train.csv", index=False)

# Confirm the row count of X_submission
print("Rows in test.csv (expected 212192):", testingSet.shape[0])
print("Rows in X_submission after merging:", X_submission.shape[0])


# Sample + Split into training and testing set
X_train, X_test, Y_train, Y_test = train_test_split(
    X_train.drop(columns=['Score']),
    X_train['Score'],
    test_size=1/4.0,
    random_state=0
)

# Feature Selection
features = ['HelpfulnessNumerator', 'HelpfulnessDenominator', 'Time', 'Helpfulness']
X_train_select = X_train[features]
X_test_select = X_test[features]
X_submission_select = X_submission[features]

# Model Creation with Logistic Regression
# model = LogisticRegression(max_iter=1000).fit(X_train_select, Y_train)

# Define Pipeline for Scaling and Logistic Regression
pipe = Pipeline([
    ('scaler', StandardScaler()),  # Scaling for Logistic Regression
    ('logreg', LogisticRegression(max_iter=1000))  # Base model with max_iter
])

# Define the Parameter Grid for Hyperparameter Tuning
param_grid = {
    'logreg__C': [0.01, 0.1, 1, 10, 100],      # Regularization strength
    'logreg__penalty': ['l1', 'l2'],           # Regularization type
    'logreg__solver': ['liblinear', 'saga']    # Solver choice based on penalty
}

# Grid Search with Cross-Validation and Multithreading
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)  # n_jobs=-1 uses all available CPU cores
grid.fit(X_train_select, Y_train)

# Use the best-found parameters
model = grid.best_estimator_

# Display the best parameters and accuracy
print("Best Parameters:", grid.best_params_)
print("Best Cross-Validation Accuracy:", grid.best_score_)

# Predict the score using the model
Y_test_predictions = model.predict(X_test_select)

# Model Evaluation
print("Accuracy on testing set = ", accuracy_score(Y_test, Y_test_predictions))

# Confusion Matrix
cm = confusion_matrix(Y_test, Y_test_predictions, normalize='true')
sns.heatmap(cm, annot=True)
plt.title('Confusion matrix of the classifier')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Create submission file
X_submission['Score'] = model.predict(X_submission_select)
submission = X_submission[['Id', 'Score']]
# submission.to_csv("./data/submission.csv", index=False)
# Check row count and save if correct
if submission.shape[0] == 212192:
    submission.to_csv("./data/submission.csv", index=False)
    print("Submission file created successfully with 212192 rows!")
else:
    print(f"Error: Expected 212192 rows, but got {submission.shape[0]}")