import cv2
import numpy as np

# Định nghĩa hàm NMS có 3 tham số: Số hộp giới hạn, (% dự đoán), giá trị ngưỡng
def nms(bounding_boxes, confidence_score, threshold):
    # Nếu không có hộp nào trả về danh sách rỗng
    if len(bounding_boxes) == 0:
        return []

    # Tạo mảng danh sách boxes chứa các hộp giới hạn
    boxes = np.array(bounding_boxes)


    # xác định tọa độ các hộp giới hạn
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Tạo mảng danh sách score chứa Confidence scores (% dự đoán) của hộp giới hạn
    score = np.array(confidence_score)

    # Tạo danh sách chứa các hộp giới hạn được chọn
    picked_boxes = []
    picked_score = []

    # Tính diện tích (vùng) của hộp giới hạn
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sắp xếp confidence score của hộp giới hạn
    order = np.argsort(score)

    # lặp qua mảng danh sách chứa hộp giới hạn
    while order.size > 0:
        # The ch số index hộp giới hạn có confidence score (% dự đoán) lớn nhất
        index = order[-1]

        # Lấy hộp giới hạn có confidence score (% dự đoán) lớn nhất
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        # Tính tọa độ của intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Tính diện tích của intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Tính tỷ lệ giữa phép giao (intersection)  và phép hợp (union)
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        # Nếu nhỏ hơn ngưỡng cho trước thì lấy
        left = np.where(ratio < threshold)
        order = order[left]
    # Trả  về các  hộp giới hạn tốt nhất và giá trị % confidence score tương ứng
    return picked_boxes, picked_score