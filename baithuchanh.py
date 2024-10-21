import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh dưới dạng ảnh xám (grayscale)
image = cv2.imread('images.jpg', cv2.IMREAD_GRAYSCALE)

# Hiển thị ảnh gốc
plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Ảnh gốc')
plt.axis('off')

# 1. Toán tử Sobel (gradient theo X và Y)
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Gradient theo trục X
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Gradient theo trục Y

# Tổng hợp hai gradient Sobel theo X và Y
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

# Hiển thị ảnh với toán tử Sobel
plt.subplot(1, 3, 2)
plt.imshow(sobel_combined, cmap='gray')
plt.title('Sobel Edge Detection')
plt.axis('off')

# 2. Toán tử Laplacian of Gaussian (LoG)
# Làm mịn ảnh bằng Gaussian trước
blurred_image = cv2.GaussianBlur(image, (3, 3), 0)

# Áp dụng toán tử Laplace
laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)

# Hiển thị ảnh với toán tử Laplace
plt.subplot(1, 3, 3)
plt.imshow(np.abs(laplacian), cmap='gray')
plt.title('Laplacian of Gaussian')
plt.axis('off')

# Hiển thị tất cả ảnh
plt.show()
