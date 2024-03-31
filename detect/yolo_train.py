
from ultralytics import YOLO


ckpt_path = r""
data_path = r""
project_path = r""
pretrain_path = r""


if __name__ == '__main__':

    # Create a new YOLOv8n model from scratch
    # model = YOLO('yolov8n.yaml')  

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO(pretrain_path)

    # Train the model using the 'coco128.yaml' dataset for 3 epochs
    results = model.train(data=data_path, 
                        imgsz=640, 
                        epochs=200, 
                        device=[0,1], 
                        batch=-1,
                        project = project_path)
