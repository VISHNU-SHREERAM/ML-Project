from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

def classify_svc(X_train, y_train, X_test, y_test, **kwargs):
    clf = SVC(**kwargs)
    
    # training set in x, y axis
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy_clf = accuracy_score(y_test, y_pred)
    print(f"Classification Accuracy: {accuracy_clf}")

    
    print('Classification Report',classification_report(y_test, y_pred))
    return ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))

def classify_rf(X_train, y_train, X_test, y_test):
    # 3. Train a Random Forest Classifier
    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(X_train, y_train)

    # 4. Predict on the test set
    y_pred_clf = rf_classifier.predict(X_test)

    # 5. Calculate classification metrics
    accuracy_clf = accuracy_score(y_test, y_pred_clf)
    print(f"Classification Accuracy: {accuracy_clf}")

    # 6. Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred_clf)
    print('Classification Report',classification_report(y_test, y_pred_clf))
    return ConfusionMatrixDisplay(conf_matrix)

def compare_clf(X_train, y_train, X_test, y_test, classifiers):
    for name, model in classifiers.items():
        model.fit(X_train, y_train)
        y_pred_clf = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred_clf)
        print(f"{name} Classifier - Accuracy: {accuracy:.4f}")

        # Confusion Matrix for each classifier
        conf_matrix = confusion_matrix(y_test, y_pred_clf)
        ConfusionMatrixDisplay(conf_matrix).plot(cmap="Blues")
        plt.title(f"Confusion Matrix for {name} Classifier")
        plt.show()