# Cài đặt pip install opencv-python opencv-contrib-python để dùng ximgproc
# tiếp đến cài pip install opencv-contrib-python
import random
import cv2

# Mở file ảnh
image = cv2.imread("dog.jpg")
# Khởi tạo Selective Search bằng phân đoạn ảnh của OpenCV và thiết lập ảnh đầu vào
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)

#ss.switchToSelectiveSearchFast()  # Tìm kiếm chọn lọc, nhanh nhưng độ chính xác kém
# Hoặc dùng
ss.switchToSelectiveSearchQuality() # Tìm kiếm chọn lọc, chậm nhưng độ chính xác cao
rects = ss.process()  # Trả về các vùng đề xuất
print("[INFO] {} Tổng số vùng đề xuất".format(len(rects)))

# Lặp qua các vùng đề xuất và vẽ Hộp giới hạn trên ảnh
output = image.copy()
for (x, y, w, h) in rects:
	color = [random.randint(0, 255) for j in range(0, 3)]
	cv2.rectangle(output, (x, y), (x + w, y + h), color, 1)
# Hiển thị ảnh
cv2.imshow("Output", output)
key = cv2.waitKey(0) & 0xFF

