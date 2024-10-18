import cv2
import numpy as np
import matplotlib.pyplot as plt

# Hàm hiển thị ảnh
def show_image(title, image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# Đọc ảnh
image = cv2.imread('"D:\Ảnh\71e1f591dc15901a1cf61617bb7565ae.jpg"')  # Thay thế bằng đường dẫn ảnh của bạn

# 1. Ánh âm tính (Negative Image)
def negative_image(image):
    return 255 - image  # Đảo ngược giá trị pixel

negative_img = negative_image(image)
show_image("Negative Image", negative_img)

# 2. Tăng độ tương phản bằng CLAHE (Contrast Limited Adaptive Histogram Equalization)
def contrast_enhancement(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # Chuyển đổi ảnh sang không gian màu LAB
    l, a, b = cv2.split(lab)  # Tách các kênh màu
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # Tạo đối tượng CLAHE
    cl = clahe.apply(l)  # Áp dụng CLAHE lên kênh L (độ sáng)
    enhanced_lab = cv2.merge((cl, a, b))  # Kết hợp lại các kênh
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)  # Chuyển đổi lại sang không gian màu BGR

contrast_img = contrast_enhancement(image)
show_image("Contrast Enhanced", contrast_img)

# 3. Biến đổi log (Log Transformation)
def log_transform(image):
    c = 255 / (np.log(1 + np.max(image)))  # Hệ số c
    log_image = c * (np.log(1 + image.astype(np.float64)))  # Áp dụng biến đổi log
    return np.array(log_image, dtype=np.uint8)  # Chuyển về kiểu uint8

log_img = log_transform(image)
show_image("Log Transformed", log_img)

# 4. Cân bằng Histogram (Histogram Equalization)
def histogram_equalization(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)  # Chuyển đổi sang không gian màu YUV
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])  # Cân bằng kênh độ sáng (Y channel)
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)  # Chuyển lại sang không gian màu BGR

hist_eq_img = histogram_equalization(image)
show_image("Histogram Equalized", hist_eq_img)
