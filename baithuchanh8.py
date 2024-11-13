# Import các thư viện cần thiết
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Bước 1: Tiền xử lý dữ liệu
# Khởi tạo ImageDataGenerator để tăng cường dữ liệu cho tập huấn luyện
train_datagen = ImageDataGenerator(
    rescale=1./255,              # chuẩn hóa ảnh
    rotation_range=20,           # xoay ảnh ngẫu nhiên 20 độ
    width_shift_range=0.2,       # dịch chuyển ngang ngẫu nhiên 20%
    height_shift_range=0.2,      # dịch chuyển dọc ngẫu nhiên 20%
    shear_range=0.2,             # biến dạng ảnh
    zoom_range=0.2,              # phóng to/thu nhỏ ngẫu nhiên
    horizontal_flip=True         # lật ảnh ngẫu nhiên theo chiều ngang
)

# Không cần tăng cường cho tập kiểm tra, chỉ cần chuẩn hóa
validation_datagen = ImageDataGenerator(rescale=1./255)

# Đọc dữ liệu từ thư mục Train và Validation
train_generator = train_datagen.flow_from_directory(
    'D:/pythonProject/dataset/Train',             # Đường dẫn đến thư mục Train
    target_size=(150, 150),      # Kích thước ảnh
    batch_size=32,               # Kích thước lô (batch size)
    class_mode='binary'          # Chế độ phân loại nhị phân
)

validation_generator = validation_datagen.flow_from_directory(
    'D:\pythonProject\dataset\Validation',        # Đường dẫn đến thư mục Validation
    target_size=(150, 150),      # Kích thước ảnh
    batch_size=32,               # Kích thước lô (batch size)
    class_mode='binary'          # Chế độ phân loại nhị phân
)

# Bước 2: Xây dựng mô hình CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)), # Lớp tích chập đầu tiên
    MaxPooling2D(2, 2),  # Lớp gộp

    Conv2D(64, (3, 3), activation='relu'),  # Lớp tích chập thứ hai
    MaxPooling2D(2, 2),  # Lớp gộp

    Conv2D(128, (3, 3), activation='relu'), # Lớp tích chập thứ ba
    MaxPooling2D(2, 2),  # Lớp gộp

    Conv2D(128, (3, 3), activation='relu'), # Lớp tích chập thứ tư
    MaxPooling2D(2, 2),  # Lớp gộp

    Flatten(),            # Chuyển dữ liệu thành vector
    Dense(512, activation='relu'),  # Lớp kết nối dày (Dense layer)
    Dropout(0.5),         # Dropout để tránh overfitting
    Dense(1, activation='sigmoid')  # Lớp đầu ra với sigmoid cho phân loại nhị phân
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Bước 3: Huấn luyện mô hình
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=15,                       # Số lượng epochs
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

# Bước 4: Sử dụng mô hình để dự đoán ảnh mới
def predict_image(model, img_path):
    img = image.load_img(img_path, target_size=(150, 150))     # Đọc ảnh và resize
    img_array = image.img_to_array(img) / 255.0                # Chuyển đổi sang array và chuẩn hóa
    img_array = np.expand_dims(img_array, axis=0)              # Thêm chiều batch
    prediction = model.predict(img_array)                      # Dự đoán ảnh
    if prediction[0] > 0.5:
        print("Dự đoán: Chó")
    else:
        print("Dự đoán: Mèo")

# Ví dụ sử dụng dự đoán ảnh
# Thay 'path_to_your_image.jpg' bằng đường dẫn thực tế của ảnh cần dự đoán
predict_image(model, 'Image_21')

# Bước 5: Đánh giá kết quả
# Vẽ biểu đồ độ chính xác và độ mất mát
# Độ chính xác
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Độ mất mát
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
