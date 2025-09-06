import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score, precision_recall_curve

from itertools import cycle

def distr_tgt_feat(label_enc_dict, df):
    count = df['Sleep Disorder'].value_counts()
    plt.bar(label_enc_dict['Sleep Disorder'].inverse_transform(count.keys()),count)
    plt.xlabel('Sleep Disorder')
    plt.ylabel('Number of people')
    plt.show()

def boxplot(df):
    columns = ['Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'BMI Category', 'Heart Rate', 'Daily Steps', 'Diastolic Pressure', 'Systolic Pressure']
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))

    for i, column in enumerate(columns):
        ax = axes[i // 4, i % 4]
        ax.boxplot(df[column])
        ax.set_title(column)

    plt.tight_layout()
    plt.show()

def correlation_analysis(df):
    # Calculate the correlation matrix
    corr_matrix = df.corr()

    # Plot the correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap of Features')
    plt.show()

def regression_results(scoreList):
    xList = [i for i in range(1, 5)]

    plt.xlabel("Degree of polynomial feature")
    plt.ylabel("R2 Score")
    plt.title("Graph of Degree vs R2 Score")
    plt.plot(xList, scoreList)
    plt.show()

def plot_pca(df_full_pca, df, label_encoding_dict):
    labels = label_encoding_dict['Sleep Disorder'].classes_
    colors = ['r', 'g', 'b']

    for i, label in enumerate(labels):
        plt.scatter(df_full_pca[df['Sleep Disorder'] == i, 0],
                    df_full_pca[df['Sleep Disorder'] == i, 1],
                    label=label,
                    color=colors[i])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Sleep Disorder')
    plt.legend()
    plt.show()

def plot_variance_ratio(pca):
    # explained variance ratio graph
    plt.plot([i+1 for i in range(len(pca.explained_variance_ratio_)) ],np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio vs Number of Components')
    # show all the ticks in x-axis
    plt.xticks([i+1 for i in range(len(pca.explained_variance_ratio_))])
    # show the points in the graph with x,y values
    for i in range(len(pca.explained_variance_ratio_)):
        plt.text(i+1, np.cumsum(pca.explained_variance_ratio_)[i], f'({i+1}, {np.cumsum(pca.explained_variance_ratio_)[i]:.2f})', fontsize=6
                , verticalalignment='bottom', horizontalalignment='right')
        plt.grid()
    plt.show()

def plot_residual(y_test, y_pred_reg, residuals):
    # Plot the residuals
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(residuals)), y_test, label="Actual", color='blue', marker='o', alpha=0.7)
    plt.scatter(range(len(residuals)), y_pred_reg, label="Predicted", color='red', marker='o', alpha=0.7)

    plt.axhline(0, color='black', linestyle='--', lw=2)  # Add a horizontal line at y=0 for reference
    plt.xlabel("Index")
    plt.ylabel("Difference in Stress Level")
    plt.title("Difference between Actual and Predicted Stress Level")
    plt.legend()
    plt.show()

def plot_rfc(X_train, y_train, X_test, y_test):
    l2 = []
    for  i in range(1,80):
        rf_classifier = RandomForestClassifier(n_estimators=i,random_state=42)
        rf_classifier.fit(X_train, y_train)

        # 4. Predict on the test set
        y_pred_clf = rf_classifier.predict(X_test)

        # 5. Calculate classification metrics
        accuracy_clf = accuracy_score(y_test, y_pred_clf)
        l2.append(accuracy_clf)

    plt.plot(range(1,80),l2)
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Estimators')
    plt.show()

def prec_rec_avg(X_test, y_test, classifier):
    y_score = classifier.decision_function(X_test)

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(3):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i] , y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i] , y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_test.ravel(), y_score.ravel()
    )
    average_precision["micro"] = average_precision_score(y_test , y_score, average="micro")

    return precision, recall, average_precision

def precision_recall_1(X_train, y_train, X_test, y_test, classifier):
    precision, recall, average_precision = prec_rec_avg(X_test, y_test, classifier)

    display = PrecisionRecallDisplay(
        recall=recall["micro"],
        precision=precision["micro"],
        average_precision=average_precision["micro"],
        prevalence_pos_label=Counter(y_test.ravel())[1] / y_test.size,
    )
    display.plot(plot_chance_level=True)
    _ = display.ax_.set_title("Micro-averaged over all classes")

def precision_recall_2(X_test, y_test, classifier, n_classes):
    # setup plot details
    precision, recall, average_precision = prec_rec_avg(X_test, y_test, classifier)

    colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])

    _, ax = plt.subplots(figsize=(7, 8))

    f_scores = np.linspace(0.2, 0.8, num=4)
    _lines, labels = [], []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    display = PrecisionRecallDisplay(
        recall=recall["micro"],
        precision=precision["micro"],
        average_precision=average_precision["micro"],
    )
    display.plot(ax=ax, name="Micro-average precision-recall", color="gold")

    for i, color in zip(range(n_classes), colors):
        display = PrecisionRecallDisplay(
            recall=recall[i],
            precision=precision[i],
            average_precision=average_precision[i],
        )
        display.plot(ax=ax, name=f"Precision-recall for class {i}", color=color)

    # add the legend for the iso-f1 curves
    handles, labels = display.ax_.get_legend_handles_labels()
    handles.extend([l])
    labels.extend(["iso-f1 curves"])
    # set the legend and the axes
    ax.legend(handles=handles, labels=labels, loc="best")
    ax.set_title("Extension of Precision-Recall curve to multi-class")

    plt.show()