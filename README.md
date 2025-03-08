# Cat vs. Dog Image Classification

## Overview
This project classifies images of cats and dogs using deep learning. It leverages a Convolutional Neural Network (CNN) model trained on the **Dogs vs. Cats** dataset from Kaggle.

## Dataset
- **Source**: [Dogs vs. Cats Dataset](https://www.kaggle.com/datasets/salader/dogs-vs-cats)
- **Training Data**: 20,000 images (cats & dogs)
- **Validation Data**: 5,000 images (cats & dogs)
- **Categories**:
  - `Cat`
  - `Dog`

## Dependencies
To run this project, install the required dependencies:
```bash
pip install tensorflow keras numpy pandas matplotlib
```

## Setup & Execution
1. Clone or download the repository.
2. Download the dataset from Kaggle and extract it.
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook Cat_VS_Dog.ipynb
   ```
4. Follow the notebook cells to preprocess data, train the model, and evaluate performance.

## Model Architecture
- **Preprocessing**:
  - Images resized to **256x256** pixels.
  - Normalization applied.
- **Convolutional Neural Network (CNN)**:
  - **Conv2D** layers for feature extraction.
  - **MaxPooling2D** layers for dimensionality reduction.
  - **Flatten** layer to convert features into a 1D array.
  - **Dense** layers for classification.
- **Loss Function**: Binary Crossentropy.
- **Optimizer**: Adam.

## Results
- The model accurately classifies cat and dog images.
- Performance is evaluated using accuracy and loss metrics.

## Future Improvements
- Implement **data augmentation** to improve generalization.
- Use **transfer learning** (e.g., MobileNet, ResNet) for better accuracy.
- Deploy the model using **Flask** or **FastAPI** for real-time classification.
