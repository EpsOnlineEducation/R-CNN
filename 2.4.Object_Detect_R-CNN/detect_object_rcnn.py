# import the necessary packages
from nms import nms
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import pickle
import cv2

MODEL_PATH = "raccoon_detector.h5"
ENCODER_PATH = "label_encoder.pickle"
MAX_PROPOSALS_INFER=200 # xác định số lượng đề xuất tối đa Vùng để xuất (R-CNN nguyên thủy đề xuất 2000)
MIN_PROBA = 0.99
INPUT_DIMS = (224, 224)

# Nạp ảnh đầu vào từ đĩa để gán nhãn
image = cv2.imread("images/o.jpg")
image = imutils.resize(image, width=300)

# Bước 1: Thực thi tìm kiếm có chọn lọc trên hình ảnh để tạo các vùng đề xuất/hộp giới hạn
print("[INFO] Thực thi thuật toán tìm kiếm chọn lọc...")
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation() #Phân đoạn ảnh
ss.setBaseImage(image)
#ss.switchToSelectiveSearchFast() # Tìm kiếm chọn lọc
ss.switchToSelectiveSearchQuality() #Hoặc dùng phương thức này
rects = ss.process() # Tìm kiếm và trả về các Vùng đề xuất, theo dạng [x,y,w,h]

# khởi tạo danh sách vùng đề xuất mà sẽ phân loại cùng với giá trị để vẽ
# các hộp giới hạn
proposals = []
boxes = []

# lặp qua tọa độ hộp giới hạn được tạo bởi khi thực thi tìm kiếm có chọn lọc
for (x, y, w, h) in rects[:MAX_PROPOSALS_INFER]:
	# Từ giá trị (x, y, w, h) => trích xuất MAX_PROPOSALS_INFER vùng quan tâm
	# rồi chuyển đổi nó từ BGR sang RGB,

	roi = image[y:y + h, x:x + w]
	roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

	# Bước 2: Resize các vùng đề xuất về kích thước giống nhau (theo giá trị INPUT_DIMS
	# phù hợp với kích thước đầu vào của CNN được đào tạo trước)
	roi = cv2.resize(roi, INPUT_DIMS, interpolation=cv2.INTER_CUBIC)

    # Chuyển dữ liệu Vùng quan tâm ROI vào mảng để xử lý
	roi = img_to_array(roi)
	roi = preprocess_input(roi)
	# cập nhật danh sách các vùng đề xuất và danh sách chứa giá trị để vẽ
	# hộp giới hạn
	proposals.append(roi)
	boxes.append((x, y, x + w, y + h))

# chuyển đổi các Vùng đề xuất và giá trị để vẽ Hộp giới hạn vào mảng
proposals = np.array(proposals, dtype="float32")
boxes = np.array(boxes, dtype="int32")
print("[INFO] shape của Vùng đề xuất sau khi resize: {}".format(proposals.shape))

# Bước 3: Trích xuất và phân loại (lớp) từng ROI (Vùng đề xuất đã resize kích thước)

print("[INFO] Nạp mô hình tinh chỉnh và file mã hóa nhãn, và gán nhãn cho Vùng đề xuất")
model = load_model(MODEL_PATH) #Nạp model đã được hấn luyện tinh chỉnh từ file
lb = pickle.loads(open(ENCODER_PATH, "rb").read()) #Na bộ mã hóa nhãn
proba = model.predict(proposals)    #Dự đoán đặc trưng vùng đề xuất

# tìm chỉ số của tất cả các dự đoán cho lớp "raccoon"
labels = lb.classes_[np.argmax(proba, axis=1)]
idxs = np.where(labels == "raccoon")[0]

# sử dụng các chỉ mục để trích xuất tất cả giá trị để vẽ các hộp giới hạn
# và lớp liên quan với xác suất gắn nhãn "raccoon"
boxes = boxes[idxs]
proba = proba[idxs][:, 1]

# lọc các chỉ mục lớn hơn hoặc bằng xác suất dự đoán tối thiểu
idxs = np.where(proba >= MIN_PROBA)
boxes = boxes[idxs]
proba = proba[idxs]

# sao chép hình ảnh gốc (tức là tạo bản sao) để vẽ hộp giới hạn, nhãn,xác suất dự đoán lên nó
clone = image.copy()

# Trường hợp 1: Không dùng thuật toán NMS
# lặp qua các vùng đề xuất (vùng để vẽ hộp giới hạn) và xác suất liên quan
# và vẽ hộp giới hạn, nhãn, xác xuất dự đoán
# Không sử dụng thuật toán NMS
for (box, prob) in zip(boxes, proba):
	# Lấy giá trị tọa độ để vẽ hộp giới hạn
	(startX, startY, endX, endY) = box
	# Vẽ hộp giới hạn trên ảnh
	cv2.rectangle(clone, (startX, startY), (endX, endY),(0, 255, 0), 2)
	# Vẽ nhãn và xác suất dự đoán trên ảnh
	y = startY - 10 if startY - 10 > 10 else startY + 10
	text= "Raccoon:{:.1f}%".format(prob * 100)
	cv2.putText(clone, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

# Hiển thị ảnh
cv2.imshow("Before NMS", clone)
cv2.waitKey(0)

# Trường hợp 2: Gọi hàm thực thi thuật toán nms
print("[INFO] Thực thi thuật toán NMS...")
picked_boxes, picked_score = nms(boxes, proba, 0.5) #Gọi hàm NMS

# lặp qua các vùng đề xuất (vùng để vẽ hộp giới hạn) và xác suất liên quan
# và vẽ hộp giới hạn, nhãn, xác xuất dự đoán
for (start_x, start_y, end_x, end_y), confidence in zip(picked_boxes, picked_score):
	# vẽ hộp giới hạn trên ảnh
	cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
	# vẽ nhãn và xác suất dự đoán trên ảnh
	y = startY - 10 if startY - 10 > 10 else startY + 10
	text = "Raccoon:{:.1f}%".format(prob * 100)
	cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

# Hiển thị ảnh
cv2.imshow("After NMS", image)
cv2.waitKey(0)

