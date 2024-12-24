from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd


# ------------------------------------------------------------------------------------------------------------------

df = pd.read_csv('train.csv')

# ------------------------------------------------------------------------------------------------------------------

x = df['sms']
y = df['label']

# ------------------------------------------------------------------------------------------------------------------

max_features = 10000
max_len = 200

# ------------------------------------------------------------------------------------------------------------------

vectorizer = TfidfVectorizer(stop_words = 'english', max_features = max_features)
x = vectorizer.fit_transform(x) # Transforms string data (like text) into numerical formats such as floats, vectors, or matrices.


# -------------------------------------------------------------------------------------------------------------------

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# -------------------------------------------------------------------------------------------------------------------

model = MLPClassifier(hidden_layer_sizes = (100,), max_iter = max_len, random_state = 42)

model.fit(x_train, y_train)

model_predictions = model.predict(x_test)

# --------------------------------------------------------------------------------------------------------------------

model_accuracy = accuracy_score(y_test, model_predictions)

print(f'Accuracy: {model_accuracy}')

classification_results = classification_report(y_test, model_predictions)
