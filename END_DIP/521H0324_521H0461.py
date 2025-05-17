import cv2
import numpy as np
import os

# Biến toàn cục lưu các ảnh mẫu
TEMPLATE_SIGNS = {}

# Bản đồ ánh xạ tên biển báo
SIGN_LABELS = {
    "camnguocchieu": "Wrong way",
    "wrongway": "Wrong way",
    "loidichuyen": "Keep right",
    "keepright": "Keep right",
    "noleft": "No left turn",
    "noleftturn": "No left turn",
    "camdungxe": "No parking",
    "noparking": "No parking",
    "children": "Children",
    "slow": "Slow",
    "nostopandparking": "No stopping and parking",
    "camdungcamdoxe": "No parking",
    "road signs": "Direction sign",
    "bienbaochiduong": "Direction sign"
}

def initialize_templates(directory='sign_templates'):
    """Nạp các ảnh mẫu biển báo từ thư mục."""
    for file in os.listdir(directory):
        if file.lower().endswith(('.jpg', '.png')):
            label = os.path.splitext(file)[0]
            path = os.path.join(directory, file)
            img = cv2.imread(path)
            if img is not None:
                TEMPLATE_SIGNS[label] = img

def compare_roi_with_template(roi, template_img, size=(100, 100)):
    """So khớp ảnh ROI với template theo kích thước chuẩn."""
    roi_resized = cv2.resize(roi, size)
    template_resized = cv2.resize(template_img, size)
    
    roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_resized, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(roi_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    return np.max(result)

def identify_signs_in_frame(frame):
    """Nhận diện biển báo giao thông trong 1 khung hình."""
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    color_ranges = {
        "Red": (np.array([150, 50, 50]), np.array([180, 255, 255])),
        "Blue": (np.array([90, 120, 100]), np.array([130, 255, 255])),
        "Yellow": (np.array([0, 120, 0]), np.array([30, 255, 255]))
    }

    # Tạo mask kết hợp từ các màu
    mask_combined = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
    for lower, upper in color_ranges.values():
        mask_combined = cv2.bitwise_or(mask_combined, cv2.inRange(hsv_frame, lower, upper))

    # Làm sạch nhiễu
    kernel = np.ones((3, 3), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected = []

    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / float(h)
            if 0.8 <= aspect <= 1.5:
                roi = frame[y:y+h, x:x+w]
                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                pixel_counts = {
                    color: cv2.countNonZero(cv2.inRange(roi_hsv, *range_))
                    for color, range_ in color_ranges.items()
                }

                dominant_color = max(pixel_counts, key=pixel_counts.get)

                match_name = None
                top_score = 0

                for name, tmpl in TEMPLATE_SIGNS.items():
                    score = compare_roi_with_template(roi, tmpl)
                    if score > 0.55 and score > top_score:
                        top_score = score
                        match_name = name

                if match_name:
                    label = SIGN_LABELS.get(match_name, match_name)
                    detected.append((x, y, w, h, f"{label} ({dominant_color})"))

    return detected

def annotate_and_save_video(input_video_path, output_name='output_video1.mp4', student_id='521H0324_521H0461'):
    """Xử lý video: nhận diện và gán nhãn biển báo, sau đó xuất ra video mới."""
    initialize_templates()

    cap = cv2.VideoCapture(input_video_path)
    ret, frame = cap.read()
    if not ret:
        print("Không đọc được video.")
        return

    height, width = frame.shape[:2]
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

    if os.path.exists(output_name):
        os.remove(output_name)

    writer = cv2.VideoWriter(output_name,
                             cv2.VideoWriter_fourcc(*'avc1'),
                             fps, (width, height))

    def annotate_frame(frame):
        signs = identify_signs_in_frame(frame)
        for (x, y, w, h, label) in signs:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(frame, f"Student ID: {student_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame

    frame = annotate_frame(frame)
    cv2.imshow("Video Output", frame)
    writer.write(frame)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = annotate_frame(frame)
        writer.write(frame)
        cv2.imshow("Video Output", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

# Tạo thư mục nếu chưa có
def ensure_template_folder_exists(path='sign_templates'):
    if not os.path.exists(path):
        os.makedirs(path)

# Gọi chạy
if __name__ == "__main__":
    ensure_template_folder_exists()
    annotate_and_save_video('video1.mp4')
