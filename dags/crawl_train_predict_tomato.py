import os
import requests
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from bing_image_downloader import downloader
from ultralytics import YOLO

def crawl_images():
    os.makedirs("data_lake", exist_ok=True)

    # Download new images
    downloader.download(
        "tomato", 
        limit=50, 
        output_dir="data_lake", 
        force_replace=False
    )

    # Rename images for sequential numbering
    images = [img for img in os.listdir("data_lake") if img.endswith(('.png', '.jpg', '.jpeg'))]
    for idx, image in enumerate(images):
        old_path = os.path.join("data_lake", image)
        new_path = os.path.join("data_lake", f"tomato_{idx + 1:04d}.jpg")
        os.rename(old_path, new_path)

def check_and_predict():
    data_lake_path = "data_lake"
    data_pool_path = "data_pool"

    # Kiểm tra nếu có đủ 50 ảnh trong thư mục data_lake
    images = [img for img in os.listdir(data_lake_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
    if len(images) >= 50:
        os.makedirs(data_pool_path, exist_ok=True)

        for image in images[:50]:  # Lấy tối đa 50 ảnh để xử lý
            try:
                # Gửi ảnh đến API để dự đoán
                with open(f"{data_lake_path}/{image}", "rb") as img_file:
                    response = requests.post(
                        "http://localhost:8000/detect/",  # URL API dự đoán
                        files={"image": img_file}
                    )
                    response.raise_for_status()  # Kiểm tra lỗi khi gọi API

                # Lưu kết quả dự đoán vào data_pool
                label_path = os.path.join(data_pool_path, f"{os.path.splitext(image)[0]}.txt")
                with open(label_path, "w") as label_file:
                    label_file.write(response.text)  # Lưu kết quả trả về từ API

                # Di chuyển ảnh đã xử lý vào thư mục đã xử lý
                os.rename(f"{data_lake_path}/{image}", f"{data_lake_path}/processed_{image}")

            except Exception as e:
                print(f"Error processing image {image}: {e}")


def train_model():
    if len([f for f in os.listdir("data_lake/tomato") if f.endswith(('.png', '.jpg', '.jpeg'))]) >= 50:
        model = YOLO("/opt/airflow/yolo11n.pt")

        # Training the model
        train_results = model.train(
            data="/opt/airflow/data_raw/data.yaml",  # Path to training data
            epochs=10,  # Number of epochs
            imgsz=640,  # Image size
            device="cpu",  # Change to "cuda" for GPU
            project="/opt/airflow/runs",  # Save directory
            exist_ok=True
        )

        # Compare accuracy with current API model
        new_accuracy = train_results.metrics.get("accuracy", 0)

        try:
            api_accuracy_response = requests.get("http://localhost:8000/model/accuracy")
            api_accuracy = float(api_accuracy_response.text)
        except Exception as e:
            print(f"Error fetching API accuracy: {e}")
            api_accuracy = 0

        if new_accuracy > api_accuracy:
            # Export and update model
            model_path = "/opt/airflow/runs/yolo11n.onnx"
            model.export(format="onnx", dynamic=True, save_dir=model_path)

            with open(model_path, "rb") as model_file:
                requests.post(
                    "http://localhost:8000/model/update",
                    files={"model": model_file}
                )

# Airflow DAG definition
default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5)
}

dag = DAG(
    dag_id="tomato_pipeline",
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2025, 1, 10),
    catchup=False
)

crawl_task = PythonOperator(
    task_id="crawl_images",
    python_callable=crawl_images,
    dag=dag
)

predict_task = PythonOperator(
    task_id="check_and_predict",
    python_callable=check_and_predict,
    dag=dag
)

train_task = PythonOperator(
    task_id="train_model",
    python_callable=train_model,
    dag=dag
)

crawl_task >> predict_task >> train_task
