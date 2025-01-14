from flask import Flask, request, jsonify
import numpy as np
import onnxruntime as ort
from PIL import Image
from io import BytesIO

# Khởi tạo Flask app
app = Flask(__name__)

# Load ONNX model
ONNX_MODEL_PATH = "/opt/airflow/runs/yolo11n.onnx"

# Tải mô hình ONNX khi ứng dụng khởi chạy
session = ort.InferenceSession(ONNX_MODEL_PATH)
input_name = session.get_inputs()[0].name  # Tên đầu vào của mô hình
output_name = session.get_outputs()[0].name  # Tên đầu ra của mô hình

# Hàm tiền xử lý ảnh (chuyển đổi ảnh sang tensor)
def preprocess_image(image: Image.Image):
    img_resized = image.resize((640, 640))  # Resize ảnh về kích thước YOLO
    img_array = np.array(img_resized).astype(np.float32) / 255.0  # Chuẩn hóa
    img_array = np.transpose(img_array, (2, 0, 1))  # Định dạng CHW
    img_array = np.expand_dims(img_array, axis=0)  # Thêm batch dimension
    return img_array

# Endpoint kiểm tra trạng thái API
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "API is running"}), 200

# Endpoint phát hiện vật thể
@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    # Đọc ảnh từ yêu cầu
    file = request.files['image']
    img = Image.open(BytesIO(file.read())).convert("RGB")

    # Tiền xử lý ảnh
    input_tensor = preprocess_image(img)

    # Chạy suy luận bằng ONNX
    results = session.run([output_name], {input_name: input_tensor})[0]

    # Hậu xử lý kết quả
    detections = []
    for result in results:
        x_min, y_min, x_max, y_max, confidence, class_id = result[:6]
        if confidence > 0.5:  # Ngưỡng tin cậy
            detections.append({
                "bbox": [float(x_min), float(y_min), float(x_max), float(y_max)],
                "confidence": float(confidence),
                "class_id": int(class_id),
            })

    return jsonify({"detections": detections}), 200

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
