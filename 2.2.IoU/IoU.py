# import the necessary packages
from collections import namedtuple
import numpy as np
import cv2
# Định nghĩa đối tượng giả phát hiện Detection
# image_path: Folder chứa ảnh thử nghiệm

def Tinh_IoU(boxA, boxB):
	# Xác định tọa độ (x, y) của vùng hợp
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# Tính vùng chữ nhật hợp
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	# Tính vùng ch nhật của hộp thực tế và hộp dự đoán
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	# Tính IoU
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# Trả về giá trị IoU
	return iou



# Giả sử Đọc file ảnh
image = cv2.imread("img1.jpg")

# Giả sử danh sách chứa tọa độ hộp thực tế
# Trong bài toán phát hiện đối tượng thì giá trị này có được tập dữ liệu train
ground_truth = [39, 63, 203, 112]

# Giả sử danh sách chứa tọa độ mà bộ phát hiện đối tượng dự đoán
# Trong bài toán phát hiện đối tượng thì giá trị này có được từ thuật toán
# Tìm kiếm chọn lọc
predicted = [54, 66, 198, 114]

# Dựa số liệu như trên, vẽ hộp thực tế và hộp dự đoán
cv2.rectangle(image, ground_truth[:2],ground_truth[2:], (0, 255, 0), 2)
cv2.rectangle(image, predicted[:2],predicted[2:], (0, 0, 255), 2)

# Tính IoU và vẽ giá trị IoU trên ảnh
iou = Tinh_IoU(ground_truth, predicted)
cv2.putText(image, "IoU: {:.4f}".format(iou), (10, 30),	cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Hiển thị ảnh
cv2.imshow("Image", image)
cv2.waitKey(0)