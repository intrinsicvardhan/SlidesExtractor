from keras_ocr.tools import read
from keras_ocr.pipeline import Pipeline
import cv2
import numpy as np
from scipy.spatial import distance
from PIL import Image
from PIL import ImageChops
from skimage.metrics import structural_similarity as compare_ssim
from skimage.transform import resize
from sklearn.metrics.pairwise import cosine_similarity

# Define the IoU function
def IoU(boxA, boxB):
    xA, yA, wA, hA = boxA
    xB, yB, wB, hB = boxB
    x1 = max(xA, xB)
    y1 = max(yA, yB)
    x2 = min(xA + wA, xB + wB)
    y2 = min(yA + hA, yB + hB)
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    boxA_area = wA * hA
    boxB_area = wB * hB
    iou_overlap = intersection_area / float(boxA_area + boxB_area - intersection_area)
    return iou_overlap

# Define the extract_text function
def extract_text(image):
    img = read(image)
    pipeline = Pipeline()
    predictions, _ = pipeline.predict(img)
    text = [pred.text for pred in predictions]
    return ' '.join(text)

# Define the extract_boxes function
def extract_boxes(image):
    img = read(image)
    pipeline = Pipeline()
    predictions, _ = pipeline.predict(img)
    boxes = [pred.bbox for pred in predictions]
    confidences = [pred.confidence for pred in predictions]
    return boxes, confidences

# Define the extract_image_features function
def extract_image_features(image):
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    return image

# Define the detect_slide_transition function
def detect_slide_transition(prev_image, curr_image, threshold=0.3):
    prev_text = extract_text(prev_image)
    curr_text = extract_text(curr_image)
    prev_boxes, prev_confidences = extract_boxes(prev_image)
    curr_boxes, curr_confidences = extract_boxes(curr_image)

    # Calculate IoU overlap between previous and current text regions
    iou_overlaps = [IoU(prev_box, curr_box) for prev_box in prev_boxes for curr_box in curr_boxes]
    max_iou_overlap = max(iou_overlaps) if iou_overlaps else 0

    # Check if there is a significant change in textual content
    if max_iou_overlap < threshold:
        return True

    # Check if there is a significant change in text content
    if prev_text != curr_text:
        return True

    # Calculate image similarity metrics
    prev_image_features = extract_image_features(prev_image)
    curr_image_features = extract_image_features(curr_image)
    ssim = compare_ssim(prev_image_features, curr_image_features)
    if ssim < 0.5:
        return True

    # Calculate cosine similarity between image features
    cos_sim = cosine_similarity([prev_image_features.flatten()], [curr_image_features.flatten()])[0][0]
    if cos_sim < 0.5:
        return True

    return False
