from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO('tv2.pt')

# Define your source (webcam, video file, etc.)
source = 'tv.mp4'  # For webcam, use '0' or '1' depending on the webcam index

# Run inference and save results
results = model(source, show=True, show_boxes=True, imgsz=1024, conf=0.15,show_labels=True,line_width=2)
