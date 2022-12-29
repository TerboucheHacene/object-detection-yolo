from yolov5 import train, val, detect, export

# Create configuration
import yaml

config = {
    "path": "/raid/home/labuserterbouche/workspace/object-detection-yolo/artifacts/vehicles_open_image",
    "train": "/raid/home/labuserterbouche/workspace/object-detection-yolo/artifacts/vehicles_open_image/train",
    "val": "/raid/home/labuserterbouche/workspace/object-detection-yolo/artifacts/vehicles_open_image/valid",
    "nc": 5,
    "names": ["Ambulance", "Bus", "Car", "Motorcycle", "Truck"],
}

with open("artifacts/vehicles_open_image/data.yaml", "w") as file:
    yaml.dump(config, file, default_flow_style=False)

SIZE = 640
BATCH_SIZE = 128
EPOCHS = 20
MODEL = "yolov5s.pt"
WORKERS = 8
PROJECT = "vehicles_open_image"
RUN_NAME = f"{MODEL}_size{SIZE}_epochs{EPOCHS}_batch{BATCH_SIZE}_small"

if __name__ == "__main__":
    # import debugger
    import debugpy

    # 5678 is the default attach port in the VS Code debug configurations
    print("Waiting for debugger attach")
    debugpy.listen(5678)
    debugpy.wait_for_client()

    # train.run(
    #     data="artifacts/vehicles_open_image/data.yaml",
    #     weights=MODEL,
    #     imgsz=SIZE,
    #     batch_size=BATCH_SIZE,
    #     epochs=EPOCHS,
    #     workers=WORKERS,
    #     project=PROJECT,
    #     name=RUN_NAME,
    # )
    img_url = "https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg"

    detect.run(source=img_url, weights="yolov5s6.pt", conf_thres=0.25, imgsz=640)
