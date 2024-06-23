import cv2
import numpy as np

# Định nghĩa hàm NMS có 3 tham số: Số hộp giới hạn, % dự đoán, giá trị ngưỡng
def nms(bounding_boxes, confidence_score, threshold):
    # Nếu không có hộp nào (bounding_boxes) trả về danh sách rỗng
    if len(bounding_boxes) == 0:
        return []

    # Tạo mảng danh sách boxes chứa các hộp giới hạn
    boxes = np.array(bounding_boxes)

    # xác định tọa độ các hộp giới hạn
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Tạo mảng danh sách score chứa Confidence scores (% dự đoán) của hộp giới hạn
    score = np.array(confidence_score)

    # Tạo danh sách chứa các hộp giới hạn được chọn
    picked_boxes = []
    picked_score = []

    # Tính diện tích (vùng) của hộp giới hạn
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Sắp xếp confidence score (% dự đoán) tăng dần của hộp giới hạn
    order = np.argsort(score)

    # lặp qua mảng danh sách chứa hộp giới hạn
    while order.size > 0:
        # The chỉ số index hộp giới hạn có confidence score (% dự đoán) lớn nhất
        index = order[-1]

        # Lấy hộp giới hạn có confidence score (% dự đoán) lớn nhất
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        # Tính tọa độ của intersection-over-union(IOU)
        x1 = np.maximum(x1[index], x1[order[:-1]])
        x2 = np.minimum(x2[index], x2[order[:-1]])
        y1 = np.maximum(y1[index], y1[order[:-1]])
        y2 = np.minimum(y2[index], y2[order[:-1]])

        # Tính diện tích của intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Tính tỷ lệ giữa phép giao (intersection) và phép hợp (union)
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        # Nếu nhỏ hơn ngưỡng cho trước thì lấy
        left = np.where(ratio < threshold)
        order = order[left]
    # Trả  về các  hộp giới hạn tốt nhất và giá trị % confidence score tương ứng
    return picked_boxes, picked_score



# xây dựng danh sách chứa 6 hộp giới hạn (giả sử bộ phát hiện tìm ra)
bounding_boxes = [(12, 84, 140, 212),
				  (24, 84, 152, 212),
				  (36, 84, 164, 212),
		    	  (12, 96, 140, 224),
				  (24, 96, 152, 224),
				  (24, 108, 152, 236)]
confidence_score = [0.9, 0.75, 0.8, 0.2,0.3,0.9]

# Read image
image = cv2.imread("img1.jpg")

# Tạo ảnh từ ảnh gốc
org = image.copy()

# Giả sử cho ngưỡng IoU
threshold = 0.5

# Vẽ các hộp  giới hạn và confidence score
for (x1, y1, x2, y2), confidence in zip(bounding_boxes, confidence_score):
    (w, h), baseline = cv2.getTextSize(str(confidence), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.rectangle(org, (x1, y1 - (2 * baseline + 5)), (x1 + w, y1), (0, 255, 255), -1)
    cv2.rectangle(org, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(org, str(confidence), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# Thực thi thuật toán non-max suppression
picked_boxes, picked_score = nms(bounding_boxes, confidence_score, threshold)

# Vẽ các hộp  giới hạn và confidence score sau khi thực thi non-maximum supression
for (x1, y1, x2, y2), confidence in zip(picked_boxes, picked_score):
    (w, h), baseline = cv2.getTextSize(str(confidence), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.rectangle(image, (x1, y1 - (2 * baseline + 5)), (x1 + w, y1), (0, 255, 255), -1)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(image, str(confidence), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# Hiển thị ảnh
cv2.imshow('Original', org)
cv2.imshow('NMS', image)
cv2.waitKey(0)