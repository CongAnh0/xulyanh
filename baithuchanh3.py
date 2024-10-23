import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import time

# 1. Đường dẫn tới thư mục chứa ảnh
dataset_path = "dataset/flowers"

# 2. Khởi tạo biến để lưu ảnh và nhãn (labels)
images = []
labels = []

# 3. Duyệt qua các thư mục con (tên thư mục là nhãn của ảnh)
for category in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category)

    if os.path.isdir(category_path):
        for img_file in os.listdir(category_path):
            img_path = os.path.join(category_path, img_file)
            img = cv2.imread(img_path)

            # Resize ảnh về kích thước cố định (ví dụ 64x64)
            img = cv2.resize(img, (64, 64))

            # Thêm ảnh và nhãn vào danh sách
            images.append(img)
            labels.append(category)

# 4. Chuyển đổi dữ liệu thành numpy array
X = np.array(images)
y = np.array(labels)

# 5. Chuyển đổi nhãn từ chuỗi thành số (sử dụng LabelEncoder)
le = LabelEncoder()
y = le.fit_transform(y)

# 6. Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 7. Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform([i.flatten() for i in X_train])
X_test = scaler.transform([i.flatten() for i in X_test])


# Hàm đánh giá và đo thời gian thực thi
def evaluate_model(model, X_train, X_test, y_train, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    elapsed_time = end_time - start_time

    return accuracy, precision, recall, elapsed_time


# 8. SVM
svm_model = SVC()
svm_results = evaluate_model(svm_model, X_train, X_test, y_train, y_test)

# 9. KNN
knn_model = KNeighborsClassifier()
knn_results = evaluate_model(knn_model, X_train, X_test, y_train, y_test)

# 10. Decision Tree
tree_model = DecisionTreeClassifier()
tree_results = evaluate_model(tree_model, X_train, X_test, y_train, y_test)

# 11. So sánh kết quả
print("SVM Results: Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, Time: {:.4f} seconds".format(*svm_results))
print("KNN Results: Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, Time: {:.4f} seconds".format(*knn_results))
print("Decision Tree Results: Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, Time: {:.4f} seconds".format(
    *tree_results))
