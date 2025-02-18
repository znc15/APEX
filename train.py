from ultralytics import YOLO

def main():
    # Load a model
    #model = YOLO("yolo11n.yaml")  # build a new model from YAML
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
    #model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(data="/kaggle/working/apexcode/data.yaml", epochs=100, imgsz=640, batch=64)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()