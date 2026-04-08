from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# 🔥 Device (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 🔥 Load model once
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)


def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(image, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=40,   # 🔥 longer captions
            num_beams=5          # 🔥 better quality
        )

    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption


# 🔥 OPTIONAL (better keyword extractor if you still want it)
def extract_objects_from_caption(caption):
    """
    Extract meaningful object words from caption
    (better than old keyword method)
    """
    stopwords = {"a", "the", "is", "on", "in", "with", "and", "of"}

    words = caption.lower().split()

    # Keep only meaningful words
    objects = [w for w in words if w not in stopwords and len(w) > 2]

    return list(set(objects))