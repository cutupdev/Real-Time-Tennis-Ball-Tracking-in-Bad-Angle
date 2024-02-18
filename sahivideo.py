from sahi import AutoDetectionModel
from sahi.predict import get_prediction
from sahi.utils.cv import read_image
import cv2

# Load the YOLOv8 model
yolov8_model_path = "tv.pt"
detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=yolov8_model_path,
    confidence_threshold=0.0,
    device="cuda:0"  # or 'cpu'
)

# Load your video
video_path = 'tv.mp4'
video = cv2.VideoCapture(video_path)

# Iterate over video frames
while True:
    ret, frame = video.read()
    if not ret:
        break

    # Perform prediction on the frame
    result = get_prediction(frame, detection_model)

    # Draw bounding boxes and labels on the frame
    for obj in result.object_prediction_list:
        bbox = obj.bbox.to_voc_bbox()
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        label = f"{obj.category.name}"
        cv2.putText(frame, label, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Frame with Predictions", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video and close all windows
video.release()
cv2.destroyAllWindows()
