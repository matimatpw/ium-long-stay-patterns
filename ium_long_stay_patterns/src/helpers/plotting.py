import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def plot_confusion_matrix(y_true, y_pred, title="Confusion matrix"):
    # Obliczanie macierzy
    cm = confusion_matrix(y_true, y_pred)

    # Tworzenie wykresu
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Short Stay (0)", "Long Stay (1)"],
        yticklabels=["Short Stay (0)", "Long Stay (1)"],
    )

    plt.title(title)
    plt.ylabel("Real")
    plt.xlabel("Predicted")
    plt.show()

    # Dodatkowo warto wyświetlić raport tekstowy (Precision, Recall, F1)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
