from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Giả sử X là đặc trưng và y là nhãn cho dữ liệu y tế
X = np.random.rand(1000, 100)  # Thay thế bằng dữ liệu thực tế
y = np.random.randint(0, 2, 1000)  # Nhãn nhị phân cho ví dụ

# Các tỷ lệ chia tập dữ liệu
split_ratios = [(0.8, 0.2), (0.7, 0.3), (0.6, 0.4), (0.4, 0.6)]

# Kết quả sẽ lưu trữ độ chính xác trên mỗi tỷ lệ chia
results = []

for train_size, test_size in split_ratios:
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size)

    # SVM
    svm_model = SVC(kernel='linear')  # Hoặc kernel khác phù hợp với dữ liệu
    svm_model.fit(X_train, y_train)
    svm_predictions = svm_model.predict(X_test)

    # KNN
    knn_model = KNeighborsClassifier(n_neighbors=5)  # Số láng giềng có thể thử thay đổi
    knn_model.fit(X_train, y_train)
    knn_predictions = knn_model.predict(X_test)

    # Đánh giá kết quả SVM
    svm_accuracy = accuracy_score(y_test, svm_predictions)
    svm_precision = precision_score(y_test, svm_predictions)
    svm_recall = recall_score(y_test, svm_predictions)
    svm_f1 = f1_score(y_test, svm_predictions)

    # Đánh giá kết quả KNN
    knn_accuracy = accuracy_score(y_test, knn_predictions)
    knn_precision = precision_score(y_test, knn_predictions)
    knn_recall = recall_score(y_test, knn_predictions)
    knn_f1 = f1_score(y_test, knn_predictions)

    # Lưu kết quả
    results.append({
        'train-test ratio': f'{int(train_size*100)}-{int(test_size*100)}',
        'SVM': {'accuracy': svm_accuracy, 'precision': svm_precision, 'recall': svm_recall, 'f1_score': svm_f1},
        'KNN': {'accuracy': knn_accuracy, 'precision': knn_precision, 'recall': knn_recall, 'f1_score': knn_f1},
    })

# In kết quả
for result in results:
    print(result)
