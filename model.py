#Import Packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.semi_supervised import LabelSpreading, LabelPropagation
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

import os
import time

# Classifier
classifiers = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Neural Network": MLPClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Bagging": BaggingClassifier(random_state=42),
    "Extra Trees": ExtraTreesClassifier(random_state=42),
    "Voting Classifier": VotingClassifier(estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svm', SVC(random_state=42)),
    ]),
    "Stacking Classifier": StackingClassifier(estimators=[
        ('dt', DecisionTreeClassifier(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svm', SVC(random_state=42)),
    ], final_estimator=LogisticRegression()),
    "Gaussian Process": GaussianProcessClassifier(random_state=42),
    "QDA": QuadraticDiscriminantAnalysis(),
    "LDA": LinearDiscriminantAnalysis(),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=42),
    "Label Spreading": LabelSpreading(),
    "Label Propagation": LabelPropagation(),
    "Linear SVC": make_pipeline(FunctionTransformer(lambda x: x.astype(float)), LinearSVC(random_state=42))
}

#Evaluasi Model
def evaluate_model(classifier, X_train, X_test, y_train, y_test):
    start_time = time.time()
    classifier.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Prediksi
    train_preds = classifier.predict(X_train)
    test_preds = classifier.predict(X_test)
    
    # Metriks Evaluasi
    accuracy_train = accuracy_score(y_train, train_preds)
    accuracy_test = accuracy_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds, average='weighted')
    precision = precision_score(y_test, test_preds, average='weighted')
    recall = recall_score(y_test, test_preds, average='weighted')
    
    return {
        "Accuracy Training": accuracy_train,
        "Accuracy Testing": accuracy_test,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall,
        "Train Time (s)": train_time
    }

def create_output_folder(output_folder):
    # Membuat folder apabila tidak ada
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

def main(input_file, output_folder):
    # absolut path
    input_file = os.path.abspath(input_file)
    output_folder = os.path.abspath(output_folder)
    
    # membaca data
    df = pd.read_csv(input_file, delimiter=';')

    # menentukan target dan input
    X = df[['Kedalaman_qu', 'Kedalaman_m']]
    y = df['Konsistensi']
    
    # Encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inisialisasi hasil
    results = []

    # Evaluasi setiap model yang sudah didefinisikan di atas
    for clf_name, clf in classifiers.items():
        print(f"Training {clf_name}...")
        clf_results = evaluate_model(clf, X_train, X_test, y_train, y_test)
        results.append({
            "Method": clf_name,
            **clf_results
        })
        print(f"{clf_name} trained.")

    # Membuat dataframe untuk hasil evaluasi
    results_df = pd.DataFrame(results)

    # Membuat output folder
    create_output_folder(output_folder)

    # Menyimpan ke csv
    output_file = os.path.join(output_folder, 'metadata-model.csv')
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Machine Learning Model Trainer")
    parser.add_argument('--input', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Path to the output folder')

    args = parser.parse_args()

    main(args.input, args.output)
