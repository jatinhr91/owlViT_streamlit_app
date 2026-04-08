# 🦉 OwlViT – Image Captioning & Object Detection Web App

## Overview

**OwlViT** is an AI-powered web application that automatically generates captions for uploaded images and detects objects in them.  
It leverages **BLIP (Bootstrapped Language Image Pretraining)** for image captioning and **YOLOv8** for real-time object detection.  
The app is **interactive, user-friendly, and deployable online**, making it an ideal portfolio project for AI enthusiasts and professionals.

---

## Features

- **Automatic Image Captioning**: Generates descriptive captions for uploaded images.  
- **Real-Time Object Detection**: Detects multiple objects with bounding boxes.  
- **Duplicate Removal**: Shows each detected object only once for clarity.  
- **Interactive Web Interface**: Built using **Streamlit**.  
- **Live Demo**: Can be deployed on **Hugging Face Spaces** or **Streamlit Cloud**.

---

## Demo Example

**Caption:** a living room with a couch and a tv  
**Detected Objects:** couch, tv  

---

## Installation

### Requirements
- Python 3.8+  
- Libraries:
```text
streamlit
torch
transformers
ultralytics
Pillow
