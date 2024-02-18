from sahi import AutoDetectionModel
from sahi.predict import get_prediction
from sahi.utils.cv import read_image
import cv2

# Load the YOLOv8 model
yolov8_model_path = "yolov8x.pt"
detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=yolov8_model_path,
    confidence_threshold=0.3,
    device="cuda:0"  # or 'cpu'
)

# Read the image
image_path = "demo_data/small-vehicles1.jpeg"
image = read_image(image_path)

# Perform prediction
result = get_prediction(image, detection_model)

# Draw bounding boxes on the image
for obj in result.object_prediction_list:
    bbox = obj.bbox.to_voc_bbox()
    cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)

    label = f"{obj.category.name} {obj.score}"

    cv2.putText(image, label, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the image with predictions
cv2.imshow("Predictions", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

