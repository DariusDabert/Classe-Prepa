# -*- coding: utf-8 -*-

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import wavfile
import pywt
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import (f1_score, confusion_matrix, classification_report, 
                             recall_score, precision_recall_curve)
from sklearn.decomposition import PCA

# Function to process WAV files and generate a DataFrame for amplitude variation analysis
def process_wav_files_amplitude(output_file):
    dico = {"stegano_or_not": []}
    for i in range(150):
        dico[f"key_{i}"] = []

    for k in range(20):
        for l in range(20):
            samplerate, data = wavfile.read(f'G:\\les musiques pour danser\\stegano-{k}-{l}.wav')
            data = data / max(data)  # Normalize
            coefficients = pywt.wavedec(data, 'db1', level=3)

            if k < 5:
                dico["stegano_or_not"].append(0)
            else:
                dico["stegano_or_not"].append(1)

            for i in range(50):
                dico[f"key_{i + 50}"].append(coefficients[2][i])
                dico[f"key_{i}"].append(coefficients[1][i])
                dico[f"key_{i + 100}"].append(coefficients[3][i])

    df = pd.DataFrame(dico)
    df.to_excel(output_file, index=False)

# Function to process WAV files and generate a DataFrame for rhythm variation analysis
def process_wav_files_rhythm(output_file):
    dico = {"stegano_or_not": []}
    for i in range(150):
        dico[f"key_{i}"] = []

    for k in range(8, 13):
        for l in range(8, 13):
            samplerate, data = wavfile.read(f'G:\\les nouvelles musiques youhou\\music-{k}-{l}.wav')
            data = data / max(data)  # Normalize
            coefficients = pywt.wavedec(data, 'db1', level=3)

            if l == 10 or k == 10:
                dico["stegano_or_not"].append(0)
            else:
                dico["stegano_or_not"].append(1)

            for i in range(50):
                dico[f"key_{i + 50}"].append(coefficients[2][i])
                dico[f"key_{i}"].append(coefficients[1][i])
                dico[f"key_{i + 100}"].append(coefficients[3][i])

    df = pd.DataFrame(dico)
    df.to_excel(output_file, index=False)

# Preprocessing function for data preparation
def preprocessing(df):
    def encodage(df):
        code = {"1": 1, "0": 0}
        for col in df.select_dtypes('object'):
            df[col] = df[col].map(code)
        return df

    df = encodage(df)
    X = df.drop('stegano_or_not', axis=1)
    y = df['stegano_or_not']
    return X, y

# Load and preprocess data
data = pd.read_excel('G:\\TIPE CODE\\musiquebis4.xlsx')
X, y = preprocessing(data)

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=0)
X_train, y_train = preprocessing(trainset)
X_test, y_test = preprocessing(testset)

# Define models and pipelines
preprocessor = make_pipeline(PolynomialFeatures(2, include_bias=False), SelectKBest(f_classif, k=4))
list_of_models = {
    'RandomForest': make_pipeline(VarianceThreshold(1e-12), preprocessor, RandomForestClassifier(random_state=0)),
    'DecisionTree': make_pipeline(VarianceThreshold(1e-12), preprocessor, DecisionTreeClassifier(random_state=0)),
    'AdaBoost': make_pipeline(VarianceThreshold(1e-12), preprocessor, AdaBoostClassifier(random_state=0)),
    'SVM': make_pipeline(VarianceThreshold(1e-12), preprocessor, StandardScaler(), SVC(random_state=0)),
    'KNN': make_pipeline(VarianceThreshold(1e-12), preprocessor, StandardScaler(), KNeighborsClassifier())
}

# Evaluation function
def evaluate_model(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    N, train_score, val_score = learning_curve(model, X_train, y_train, cv=4, train_sizes=np.linspace(0.1, 1, 10))
    plt.figure(figsize=(12, 6))
    plt.plot(N, train_score.mean(axis=1), label='Train Score')
    plt.plot(N, val_score.mean(axis=1), label='Validation Score')
    plt.legend()
    plt.show()

# Evaluate each model
for name, model in list_of_models.items():
    print(f"Evaluating {name}")
    evaluate_model(model)

# Hyperparameter tuning
grid_params = {
    'svc__gamma': [1e-3, 1e-4],
    'svc__C': [100, 1000],
    'pipeline__polynomialfeatures__degree': [2],
    'pipeline__selectkbest__k': range(4, 6)
}
grid = GridSearchCV(list_of_models['SVM'], grid_params, scoring='recall', cv=4)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
y_pred = grid.predict(X_test)
print(classification_report(y_test, y_pred))

# Precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, grid.best_estimator_.decision_function(X_test))
plt.plot(thresholds, precision[:-1], label='Precision')
plt.plot(thresholds, recall[:-1], label='Recall')
plt.legend()
plt.show()
