:globe_with_meridians: **Languages** | [English](./README.md) | [日本語](./README_JP.md)

# Hand Gesture Classifier

### A Data-Centric Case Study on Domain-Aware Image Classification

> **TL;DR**
> * **Challenge**: Building a robust gesture classifier with only 150 self-collected images.
> * **Solution**: Prioritizing data quality (Letterbox Padding) and domain constraints (disabling flips for Fleming's rule).
> * **Outcome**: Achieved 96.67% accuracy by proving that "Design decisions outperform dataset size."

## Overview

This project is a deep learning–based image classification model designed to recognize three specific **left-hand gestures**.
Rather than focusing on large-scale datasets or deployment, this repository serves as a **data-centric case study**, emphasizing how **domain knowledge and data quality directly influence model performance**, especially in small, real-world datasets.

---

## Project Goal

The primary goal of this project is **not** to showcase complex architectures or production deployment,
but to demonstrate how **seemingly standard preprocessing and augmentation choices can silently break a model** when domain constraints are ignored.

This project focuses on:

* How domain knowledge should guide data preprocessing
* Why generic data augmentation assumptions are sometimes invalid
* How careful dataset construction can compensate for limited data size

---

## Problem Definition

### Target Classes

The model classifies the following **three left-hand gestures**:

* Open Palm
* Thumbs Up
* Fleming's Left Hand Rule

**Key constraint**:
All gestures are **orientation-sensitive** and **left-hand–specific**.
Mirrored gestures are semantically different and must not be treated as equivalent.

---

## Dataset Construction

### Data Collection and Ownership

All images used in this project were **self-collected by the author**.
Data collection, labeling, and cleaning were performed entirely without relying on external datasets.

This approach ensures full control over data quality and allows design decisions to be grounded in firsthand observation.

### Dataset Composition

* **Total samples**: 150 images (3 classes)

  * Training set: 120 images
  * Validation set: 30 images
* **Subjects**: Hands from 5 different individuals
* **Backgrounds**: At least 3 distinct background types
* **Lighting conditions**: Balanced mix of bright and dim environments

### Design Intent

The dataset was intentionally constructed to reduce overfitting to:

* A single hand shape
* Specific backgrounds
* Fixed lighting conditions

This diversity was introduced to improve robustness under realistic usage scenarios, despite the small dataset size.

---

## Key Design Decisions

### 1. Letterbox Padding Instead of Center Crop

#### Problem

Initial experiments using standard **center cropping** resulted in the loss of critical fingertip information due to the original smartphone image aspect ratio.
This caused ambiguous representations of certain gestures.

#### Solution

**Letterbox Padding** was applied to resize images to 224×224 while preserving the original aspect ratio.

#### Impact

* Complete hand geometry was retained
* Finger-level features critical for gesture discrimination were preserved
* Label ambiguity caused by cropped fingertips was eliminated

---

### 2. Physics-Aware Data Augmentation

#### Problem

Common augmentation techniques such as **Horizontal Flip** implicitly assume label invariance under mirroring.
For this task, flipping a left hand produces a **right-hand gesture**, which is semantically incorrect.

#### Decision

* Horizontal Flip was **explicitly disabled**
* Applied augmentations were limited to:

  * Rotation
  * Brightness jitter
  * Zoom

#### Rationale

Data augmentation must respect **physical and semantic constraints**, not just statistical diversity.

---

### 3. Small-Data Strategy

* **Backbone**: ResNet18 (Transfer Learning)
* **Motivation**: Leverage pretrained visual features while minimizing overfitting
* **Result**: Test accuracy of **96.67%**

This outcome suggests that **data quality and task formulation can outweigh dataset size** when design assumptions are correct.

---

## Model Architecture

* **Backbone**: ResNet18 (pretrained)
* **Framework**: PyTorch
* **Input**: 224×224 RGB images
* **Output**: 3-class classification

---

## Known Limitations and Error Analysis

Despite the high overall accuracy, the model occasionally confuses **Open Palm** and **Fleming's Left Hand**.

### Observed Patterns

* Misclassification tends to occur when the angle between the thumb and index finger in **Open Palm** images closely resembles the geometry of **Fleming's Left Hand**.
* Prediction uncertainty increases when images contain **complex or unfamiliar backgrounds**.

### Likely Causes

* Visual similarity in finger-angle geometry between the two classes
* Limited background diversity inherent to the small dataset size

These observations indicate that further improvements may require:

* More fine-grained gesture definitions
* Additional data collection focusing on challenging edge cases

---

## Optional: Running the Demo Locally

This repository includes a minimal **Gradio-based UI** for interactive testing.

> Note: Running the demo is optional.
> The primary value of this project lies in the design rationale and data-centric decisions described above.

### Setup

```bash
pip install -r requirements.txt
```

### Launch

```bash
python app.py
```

After launching, a browser window will open where a **left-hand gesture image** can be uploaded to view the model prediction.

---

## Testing

### 1. Manual Testing (Gradio UI)
To verify the model with sample images:

1. Start the application: `python app.py`
2. Open your browser at **http://127.0.0.1:7860**
3. Upload a sample image from the `test_images/` directory (e.g., `thumbs_up.jpg`) to see the prediction.

*Note: Sample images have been resized to 224x224 pixels for privacy.*

### 2. Automated Testing
To verify the model script integrity:

```bash
pytest tests/test_model.py
```

---

## Environment

* **Framework**: PyTorch
* **UI**: Gradio
* **Development Platform**: Google Colab

---

## Takeaway

This project demonstrates that:

* Data preprocessing is not a neutral step
* Domain knowledge must guide augmentation choices
* Small datasets can still yield strong results when assumptions are explicit and validated

The most critical component of this system is **not the model itself**,
but the **decisions made before training began**.

---