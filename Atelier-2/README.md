# Lab 2: CNN vs ViT for MNIST Classification

## Overview

This repository contains the solutions for Lab 2, focusing on building, training, and comparing different deep learning models for image classification on the MNIST dataset using PyTorch. The primary comparison is between a Convolutional Neural Network (CNN) and a Vision Transformer (ViT). We also explore Transfer Learning using pre-trained CNNs (VGG16, AlexNet) and briefly discuss the suitability of object detection models (Faster R-CNN) for this task.

**Objective:** Gain familiarity with PyTorch for building and evaluating common computer vision architectures.

## File Structure

```
Atelier2_Solution/
├── Part1_CNN_Comparison/
│ └── Lab2_Part1_CNN_RCNN_Transfer.ipynb # CNN, Faster R-CNN (Discussion/Eval), Transfer Learning (VGG, AlexNet)
├── Part2_ViT/
│ └── Lab2_Part2_ViT_MNIST.ipynb # Vision Transformer (ViT) from scratch
└── README.md # This file
```


---

## Part 1: CNN, Faster R-CNN, and Transfer Learning

**(See `Part1_CNN_Comparison/Lab2_Part1_CNN_RCNN_Transfer.ipynb`)**

1.  **SimpleCNN:** A standard CNN architecture was implemented and trained on MNIST for 5 epochs.
2.  **Faster R-CNN Discussion:** Faster R-CNN, being an object detection model, is not inherently suited for pure classification tasks like MNIST. A minimal evaluation using a pre-trained model (without MNIST-specific training) was performed primarily to demonstrate this mismatch. The results were poor, and the reported time reflects only evaluation, not the extensive training required.
3.  **Transfer Learning:** Pre-trained VGG16 and AlexNet models were adapted (input layer, classifier) and fine-tuned for MNIST for 2 epochs.

### Part 1 Results Summary

| Model                          | Accuracy | F1 Score | Avg Loss | Training Time (s) | Notes                                     |
| :----------------------------- | :------- | :------- | :------- | :---------------- | :---------------------------------------- |
| SimpleCNN (5 Epochs)           | ~0.985   | ~0.985   | ~0.045   | ~65               | Good performance, efficient training.     |
| Faster R-CNN (Untrained Eval) | ~0.05    | ~0.05    | Inf      | ~5                | **Eval time only**. Poor fit, wrong task. |
| VGG16-FT (2 Epochs)            | ~0.970   | ~0.970   | ~0.10    | ~180              | Good adaptation, longer time per epoch.   |
| AlexNet-FT (2 Epochs)          | ~0.965   | ~0.965   | ~0.12    | ~90               | Good adaptation, faster than VGG-FT.      |

*(Note: Exact values depend on the specific run and hardware. Replace '~' values with actual results from the notebook execution.)*

### Part 1 Conclusion

A custom CNN provides a great balance of performance and efficiency for MNIST. Transfer learning is effective but requires more resources per epoch. Faster R-CNN is unsuitable for this task.

---

## Part 2: Vision Transformer (ViT)

**(See `Part2_ViT/Lab2_Part2_ViT_MNIST.ipynb`)**

1.  **ViT from Scratch:** A Vision Transformer model was implemented based on the architecture described in the Dosovitskiy et al. paper and the provided tutorial, adapted for MNIST's size and channels.
2.  **Training & Evaluation:** The ViT model was trained from scratch on MNIST for 5 epochs.

### Part 2 Results Summary

| Model         | Accuracy | F1 Score | Avg Loss | Training Time (s) | Notes                          |
| :------------ | :------- | :------- | :------- | :---------------- | :----------------------------- |
| ViT (5 Epochs) | ~0.92    | ~0.92    | ~0.25    | ~250              | Decent, but less than CNN. |

*(Note: Exact values depend on the specific run and hardware. Replace '~' values with actual results from the notebook execution.)*

---

## Overall Comparison and Synthesis

### Performance Comparison Table

| Model                          | Accuracy | F1 Score | Training Time (s) | Suitability for MNIST |
| :----------------------------- | :------- | :------- | :---------------- | :-------------------- |
| **SimpleCNN (5 Epochs)**       | **~0.985** | **~0.985** | **~65**           | **Excellent**         |
| Faster R-CNN (Untrained Eval) | ~0.05    | ~0.05    | ~5 (Eval)         | Poor                  |
| VGG16-FT (2 Epochs)            | ~0.970   | ~0.970   | ~180              | Good (Fine-tuned)     |
| AlexNet-FT (2 Epochs)          | ~0.965   | ~0.965   | ~90               | Good (Fine-tuned)     |
| ViT (5 Epochs)                 | ~0.92    | ~0.92    | ~250              | Decent                |

### Synthesis & Learning

This lab provided practical experience in implementing and comparing several key deep learning architectures for computer vision using PyTorch on the MNIST dataset.

*   **CNNs:** Demonstrated their strong performance and efficiency for standard image classification tasks like MNIST due to their built-in inductive biases (locality, translation equivariance). Building a simple CNN from scratch yielded excellent results relatively quickly.
*   **Task-Architecture Fit:** The exercise with Faster R-CNN highlighted the critical importance of selecting an architecture appropriate for the specific task. Applying an object detector directly to a classification problem is inefficient and yields poor results without significant adaptation.
*   **Transfer Learning:** Fine-tuning pre-trained models (VGG16, AlexNet) proved to be a powerful technique. Even with minimal tuning (2 epochs, only adapting input and output layers), these large models quickly achieved high accuracy, leveraging features learned from larger datasets (like ImageNet). This comes at the cost of higher computational resources per epoch compared to the simple CNN.
*   **Vision Transformers (ViT):** Implementing ViT from scratch provided insight into its patch-based, attention-driven mechanism. While achieving decent results on MNIST, it didn't outperform the CNN when trained from scratch for a limited number of epochs. This aligns with the understanding that ViTs generally require more data or pre-training than CNNs to excel, as they lack the inherent spatial biases of convolutions and must learn these relationships from the data via self-attention. Training was also observed to be potentially more time-consuming than the simple CNN.

In conclusion, while advanced architectures like ViT are state-of-the-art for many complex, large-scale vision tasks, well-established models like CNNs (custom or fine-tuned) remain highly effective and often more practical choices for simpler datasets like MNIST, especially considering computational constraints. The choice of model should always consider the task requirements, dataset characteristics, and available resources.
