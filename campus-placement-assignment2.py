import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

def load_dataset(filepath="train.csv"):
    """
    Loads the dataset and prints its summary.
    """
    try:
        data = pd.read_csv(filepath)
        print(f"Loaded {filepath} successfully.")
        print(f"Dataset shape: {data.shape}")
        data.info()
        print("\nMissing values:")
        print(data.isnull().sum())
        return data
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return None

def preprocess_dataset(data):
    """
    Preprocesses the dataset: encoding, scaling, splitting.
    """
    if data is None:
        return None, None, None, None, None, None

    print("\n--- Starting Data Preprocessing ---")

    # Drop irrelevant columns
    data = data.drop(columns=['sl_no', 'salary'], errors='ignore')

    # Separate features and target
    features = data.drop('status', axis=1)
    target_raw = data['status']

    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(target_raw)

    num_features = features.select_dtypes(include=np.number).columns.tolist()
    cat_features = features.select_dtypes(include='object').columns.tolist()

    num_pipeline = Pipeline([('scaler', StandardScaler())])
    cat_pipeline = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, stratify=target, random_state=42
    )

    return X_train, X_test, y_train, y_test, preprocessor, label_encoder

def evaluate_model(name, pipeline, X_test, y_test, label_names):
    """
    Evaluates a model and prints performance metrics.
    """
    print(f"\n--- Evaluating: {name} ---")
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, predictions, target_names=label_names))

    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrBr',
            xticklabels=label_names, yticklabels=label_names)
    plt.title(f'Confusion Matrix: {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    return accuracy, precision, recall, f1

def main():
    data = load_dataset()
    if data is None:
        return

    X_train, X_test, y_train, y_test, preprocessor, label_encoder = preprocess_dataset(data)
    label_names = label_encoder.classes_

    classifiers = {
        "Logistic Regression": LogisticRegression(solver='liblinear', random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest (Default)": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "k-NN": KNeighborsClassifier()
    }

    performance_summary = {}
    trained_pipelines = {}

    for model_name, clf in classifiers.items():
        pipeline = Pipeline([
            ('preprocessing', preprocessor),
            ('classifier', clf)
        ])
        print(f"\n--- Training: {model_name} ---")
        pipeline.fit(X_train, y_train)
        acc, prec, rec, f1 = evaluate_model(model_name, pipeline, X_test, y_test, label_names)
        performance_summary[model_name] = {
            'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-score': f1
        }
        trained_pipelines[model_name] = pipeline

    print("\n--- Hyperparameter Tuning: Random Forest ---")
    rf_param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [None, 10],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }
    rf_pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    grid_search = GridSearchCV(
        rf_pipeline, rf_param_grid, cv=3,
        scoring='accuracy', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    print(f"Best Parameters: {grid_search.best_params_}")

    best_rf_pipeline = grid_search.best_estimator_
    acc, prec, rec, f1 = evaluate_model(
        "Random Forest (Tuned)", best_rf_pipeline, X_test, y_test, label_names
    )
    performance_summary["Random Forest (Tuned)"] = {
        'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-score': f1
    }

    print("\n--- Training: Voting Classifier ---")
    voting_pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('voting_classifier', VotingClassifier(
            estimators=[
                ('lr', LogisticRegression(solver='liblinear', random_state=42)),
                ('dt', DecisionTreeClassifier(random_state=42)),
                ('rf', best_rf_pipeline.named_steps['classifier'])
            ],
            voting='hard'
        ))
    ])
    voting_pipeline.fit(X_train, y_train)
    acc, prec, rec, f1 = evaluate_model(
        "Voting Classifier", voting_pipeline, X_test, y_test, label_names
    )
    performance_summary["Voting Classifier"] = {
        'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-score': f1
    }

    print("\n--- Model Performance Summary ---")
    summary_df = pd.DataFrame(performance_summary).T.sort_values(by='F1-score', ascending=False)
    print(summary_df)

if __name__ == '__main__':
    main()
