import numpy as np
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def run_knn():
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    start_time = time.time()
    cls_knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
    cls_knn.fit(X_train, y_train)
    y_pred_knn = cls_knn.predict(X_test)
    end_time = time.time()
    elapsed_time = end_time - start_time
    acc = accuracy_score(y_test, y_pred_knn)
    return elapsed_time, acc

# Number of times to run the code
num_runs = 20
total_time = 0
accuracies = []

for _ in range(num_runs):
    elapsed_time, acc = run_knn()
    total_time += elapsed_time
    accuracies.append(acc)

average_time = total_time / num_runs
average_accuracy = np.mean(accuracies)

print(f"Average elapsed time over {num_runs} runs: {average_time} seconds")
print(f"Average accuracy over {num_runs} runs: {average_accuracy}")
