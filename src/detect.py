from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import torch
from torchvision.ops import nms

# Load model once
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

def detect_objects(image_path, texts):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(text=[texts], images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])

    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=0.25
    )[0]

    boxes = results["boxes"]
    scores = results["scores"]
    labels = results["labels"]

    keep = nms(boxes, scores, iou_threshold=0.5)

    results["boxes"] = boxes[keep]
    results["scores"] = scores[keep]
    results["labels"] = labels[keep]

    return image, results