from ultralytics import YOLO

# Load model
model = YOLO("/opt/airflow/scripts/yolo11n.pt")

# Train model
train_results = model.train(
    data="/opt/airflow/data_raw/data.yaml",  # dataset path
    epochs=2,  # training epochs
    imgsz=640,  # image size
    device="cpu",  # CPU or GPU
    exist_ok=True,  # suppress folder check
    project="/opt/airflow/runs",  # save location
)

# Export model to ONNX
path = model.export(format="onnx")  # exported model path
