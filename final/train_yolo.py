from ultralytics import YOLO

def train_yolo():
    # Load a pretrained model
    model = YOLO("yolov8n.pt")
    
    # Train the model for a few epochs
    # Add workers=0 to prevent hanging on Windows multiprocessing
    results = model.train(data="yolo_data/data.yaml", epochs=15, imgsz=640, device="cuda", workers=0)

if __name__ == "__main__":
    train_yolo()
