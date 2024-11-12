from ultralytics import YOLO

model = YOLO('../Capstone/runs/detect/train7/weights/best.pt')

results = model(source=0, show=True, conf=0.8, save=False)