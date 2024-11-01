import random
import numpy as np

data = [
    [5.1, 3.5, 1.4, 0.2, 0],  # setosa
    [4.9, 3.0, 1.4, 0.2, 0],  # setosa
    [4.7, 3.2, 1.3, 0.2, 0],  # setosa
    [4.6, 3.1, 1.5, 0.2, 0],  # setosa
    [5.0, 3.6, 1.4, 0.2, 0],  # setosa
    [5.4, 3.9, 1.7, 0.4, 0],  # setosa
    [4.6, 3.4, 1.4, 0.3, 0],  # setosa
    [5.0, 3.4, 1.5, 0.2, 0],  # setosa
    [4.4, 2.9, 1.4, 0.2, 0],  # setosa
    [4.9, 3.1, 1.5, 0.1, 0],  # setosa
    [5.7, 2.8, 4.1, 1.3, 1],  # versicolor
    [6.3, 2.5, 4.9, 1.5, 1],  # versicolor
    [5.5, 2.4, 3.8, 1.1, 1],  # versicolor
    [5.5, 2.8, 4.2, 1.5, 1],  # versicolor
    [6.0, 2.2, 4.0, 1.0, 1],  # versicolor
    [6.1, 2.9, 4.7, 1.4, 1],  # versicolor
    [5.8, 2.7, 4.1, 1.0, 1],  # versicolor
    [5.0, 2.0, 3.5, 1.0, 1],  # versicolor
    [5.6, 2.9, 3.6, 1.3, 1],  # versicolor
    [6.7, 3.1, 4.4, 1.4, 1],  # versicolor
    [6.3, 3.3, 6.0, 2.5, 2],  # virginica
    [5.8, 2.7, 5.1, 1.9, 2],  # virginica
    [6.4, 3.2, 5.3, 2.3, 2],  # virginica
    [6.5, 3.0, 5.2, 2.0, 2],  # virginica
    [6.3, 2.5, 5.0, 1.9, 2],  # virginica
    [6.1, 2.8, 4.7, 1.2, 2],  # virginica
    [6.4, 2.9, 4.3, 1.3, 2],  # virginica
    [6.6, 3.0, 4.4, 1.4, 2],  # virginica
    [6.8, 2.8, 4.8, 1.4, 2],  # virginica
    [6.7, 3.1, 5.6, 2.4, 2],  # virginica
]

# Số lượng cụm
k = 3


# Bước 1: Khởi tạo các tâm cụm ngẫu nhiên
def initialize_centroids(data, k):
    return random.sample(data, k)


# Bước 2: Phân cụm
def assign_clusters(data, centroids):
    clusters = [[] for _ in range(k)]
    for point in data:
        distances = [euclidean_distance(point[:-1], centroid[:-1]) for centroid in centroids]
        cluster_index = distances.index(min(distances))
        clusters[cluster_index].append(point)
    return clusters


# Tính khoảng cách Euclidean
def euclidean_distance(point1, point2):
    return sum((x - y) ** 2 for x, y in zip(point1, point2)) ** 0.5


# Bước 3: Cập nhật các tâm cụm
def update_centroids(clusters):
    new_centroids = []
    for cluster in clusters:
        new_centroid = [sum(point[i] for point in cluster) / len(cluster) for i in range(len(cluster[0]) - 1)]
        new_centroids.append(new_centroid + [0])  # Thêm 0 cho label
    return new_centroids


# Bước 4: Lặp lại cho đến khi không còn sự thay đổi
def kmeans(data, k):
    centroids = initialize_centroids(data, k)
    while True:
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(clusters)
        if new_centroids == centroids:
            break
        centroids = new_centroids
    return clusters


clusters = kmeans(data, k)


# Bước 5: Đánh giá kết quả

# Tính F1-score
def f1_score(true_labels, predicted_labels):
    tp = sum(1 for t, p in zip(true_labels, predicted_labels) if t == p and t != -1)
    fp = sum(1 for t, p in zip(true_labels, predicted_labels) if t != p and p != -1)
    fn = sum(1 for t, p in zip(true_labels, predicted_labels) if t != p and t != -1)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0


true_labels = [point[-1] for point in data]  # Nhãn thực tế
predicted_labels = []  # Nhãn dự đoán từ cụm

# Gán nhãn cho các điểm
for i, cluster in enumerate(clusters):
    for point in cluster:
        predicted_labels.append(i)  # Gán nhãn của cụm cho các điểm

f1 = f1_score(true_labels, predicted_labels)
print("F1-score:", f1)


# Tính RAND index
def rand_index(true_labels, predicted_labels):
    n = len(true_labels)
    a = sum(1 for i in range(n) for j in range(i + 1, n) if
            true_labels[i] == true_labels[j] == predicted_labels[i] == predicted_labels[j])
    b = sum(1 for i in range(n) for j in range(i + 1, n) if
            true_labels[i] != true_labels[j] and predicted_labels[i] != predicted_labels[j])
    return (a + b) / (n * (n - 1) / 2)


rand_idx = rand_index(true_labels, predicted_labels)
print("Rand Index:", rand_idx)


# Tính NMI (Normalized Mutual Information)
def mutual_information(true_labels, predicted_labels):
    contingency_table = np.zeros((k, k))
    for i in range(len(true_labels)):
        contingency_table[true_labels[i]][predicted_labels[i]] += 1
    pxy = contingency_table / np.sum(contingency_table)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    mi = 0
    for i in range(k):
        for j in range(k):
            if pxy[i][j] > 0:
                mi += pxy[i][j] * np.log(pxy[i][j] / (px[i] * py[j]))
    return mi


nmi = mutual_information(true_labels, predicted_labels) / (np.log(k) if np.log(k) != 0 else 1)
print("NMI:", nmi)


# Tính DB index
def davies_bouldin_index(clusters):
    s = []
    for cluster in clusters:
        center = np.mean(cluster, axis=0)
        scatter = sum(euclidean_distance(point[:-1], center[:-1]) for point in cluster) / len(cluster)
        s.append(scatter)

    db_index = 0
    for i in range(len(clusters)):
        max_ratio = 0
        for j in range(len(clusters)):
            if i != j:
                d = euclidean_distance(np.mean(clusters[i], axis=0), np.mean(clusters[j], axis=0))
                ratio = (s[i] + s[j]) / d if d > 0 else 0
                max_ratio = max(max_ratio, ratio)
        db_index += max_ratio

    return db_index / len(clusters)


db = davies_bouldin_index(clusters)
print("Davies Bouldin Index:", db)
