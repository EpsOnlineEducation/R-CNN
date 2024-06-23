# import the necessary packages

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

BASE_PATH= "dataset"
INPUT_DIMS = (224, 224)
MODEL_PATH = "raccoon_detector.h5"
ENCODER_PATH = "label_encoder.pickle"

INIT_LR = 1e-4  # Khởi tạo tỷ lệ học
EPOCHS = 30   # Khởi tạo số epoch
BS = 32      # Kích thước Batch

# grab danhsacahs ảnh trong folder dataset và khởi tạo
# danh sách data chứa dữ liệu ảnh và danh sách labels chứa nhãn lớp
print("[INFO] loading images...")
imagePaths = list(paths.list_images(BASE_PATH))  #Định nghĩa đường dẫn đến dataset
data = []  # Khởi tạo danh sách dữ liệu ảnh
labels = []  # Khởi tạo danh sách nhãn
# Lặp qua Dataset
for imagePath in imagePaths:
	label = imagePath.split(os.path.sep)[-2] # Trích xuất tên nhãn từ tên folder
	image = load_img(imagePath, target_size=INPUT_DIMS) # Nạp ảnh với kích thước (224x224)
	image = img_to_array(image) #Chuyển anh sang mảng
	image = preprocess_input(image) #Tiền xử lý ảnh theo định dạng của mobilenet_v2
	# Cập nhật danh sách data và labels
	data.append(image)
	labels.append(label)

# Chuyển data và labels sang dạng mảng
data = np.array(data, dtype="float32")
labels = np.array(labels)
# Mã hóa nhãn
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Chia tách Dataset train và test theo tỷ lệ 75%: 25%
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)
# Tăng cường dữ liệu ảnh
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# Nạp MobileNetV2 loại bỏ các lớp Top (Head)
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Định nghĩa lớp Top mới để thay thế lớp Top đã loại bỏ của MobileNetV2
NewTop = baseModel.output
NewTop = AveragePooling2D(pool_size=(7, 7))(NewTop)
NewTop = Flatten(name="flatten")(NewTop)
NewTop = Dense(128, activation="relu")(NewTop)
NewTop = Dropout(0.5)(NewTop)
NewTop = Dense(2, activation="sigmoid")(NewTop)

# Tạo model bằng thay thế Top từ Top mới đã định nghĩa ở trên
model = Model(inputs=baseModel.input, outputs=NewTop)

# Đóng băng lớp cơ sở (base)
for layer in baseModel.layers:
	layer.trainable = False

# Biên dịch model
print("[INFO] Biên dịch model...")
opt = Adam(lr=INIT_LR)  #Chọn Thuật toán tối ưu Adam
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
# train phần Top mới của network
print("[INFO] Đang huấn luyện phần Top (head) của mạng...")
H = model.fit(aug.flow(trainX, trainY, batch_size=BS), steps_per_epoch=len(trainX) // BS, validation_data=(testX, testY),validation_steps=len(testX) // BS,	epochs=EPOCHS)

# Chú ý, có thể train lại một số lớp cơ sở, bằng cách ở đóng băng như trong bài
# Học chuyển tiếp bằng kỹ thuật tinh chỉnh

# Đánh giá model sau train
print("[INFO] Đánh giá model...")
predIdxs = model.predict(testX, batch_size=BS)

# In kết quả dự đoán
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testY.argmax(axis=1), predIdxs,	target_names=lb.classes_))

# Lưu model
print("[INFO] lưu model bộ phát hiện mặc nạ (mask detector) ...")
model.save(MODEL_PATH, save_format="h5")

# Lưu nhãn mã hóa
print("[INFO] Lưu bộ mã hóa nhãn (label encoder)...")
f = open(ENCODER_PATH, "wb")
f.write(pickle.dumps(lb))
f.close()

# Vẽ biểu đồ quá trình train, gồm: loss và accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.jpg")