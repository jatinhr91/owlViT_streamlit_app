import matplotlib.pyplot as plt
import os
import random

def get_color(label):
    random.seed(label)
    return (random.random(), random.random(), random.random())

def draw_boxes(image, results, texts, save_path="outputs/result.jpg", threshold=0.3):
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    ax = plt.gca()

    boxes = results.get("boxes", [])
    scores = results.get("scores", [])
    labels = results.get("labels", [])

    for box, score, label in zip(boxes, scores, labels):
        score = score.item()
        label = label.item()

        if score > threshold:
            x1, y1, x2, y2 = box.tolist()

            color = get_color(label)

            rect = plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                color=color,
                linewidth=2
            )
            ax.add_patch(rect)

            label_text = texts[label] if label < len(texts) else f"id:{label}"

            ax.text(
                x1,
                y1 - 5,
                f"{label_text}: {score:.2f}",
                color='white',
                fontsize=9,
                bbox=dict(facecolor=color, alpha=0.7, pad=2)
            )

    plt.axis("off")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()