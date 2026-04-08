from src.detect import detect_objects   # (optional: keep for OWL-ViT)
from src.utils import draw_boxes
from src.caption import generate_caption
from src.yolo_detect import detect_and_save  # 🔥 NEW
import os

images = ["Dogs.jpg", "street.jpg", "rooms.jpg"]

os.makedirs("outputs", exist_ok=True)

for img_name in images:
    image_path = f"data/images/{img_name}"

    print(f"\nProcessing: {img_name}")

    # 🔹 Step 1: Caption (BLIP)
    caption = generate_caption(image_path)
    print("Caption:", caption)

    # 🔹 Step 2: AUTO detection (YOLO — no prompts needed)
    save_path = f"outputs/{os.path.splitext(img_name)[0]}.png"
    objects = detect_and_save(image_path, save_path)
    print("Detected Objects:", objects)

    # 🔹 Step 3: Save text result
    txt_path = f"outputs/{os.path.splitext(img_name)[0]}.txt"
    with open(txt_path, "w") as f:
        f.write("Caption:\n")
        f.write(caption + "\n\n")

        f.write("Detected Objects:\n")
        f.write(", ".join(objects))
