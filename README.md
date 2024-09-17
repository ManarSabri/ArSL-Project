# ArSL-GraduationProject
This repository contains an implementation of a machine learning model designed to recognize Arabic Sign Language (ARSL) gestures. The project leverages a deep learning approach to classify Arabic Sign Language signs, including both numbers, letters, and a set of 20 words, using a Bidirectional Long Short-Term Memory (BiLSTM) model.

## Introduction
Arabic Sign Language (ARSL) is an essential communication tool for the Arabic-speaking deaf community. This project aims to improve accessibility by providing a model capable of recognizing Arabic sign gestures. The dataset includes 190 classes of signs (numbers, letters, words, and verbs). The current model focuses on recognizing a combination of letters, numbers, and 20 specific words.

The dataset is processed using MediaPipe for gesture recognition and is trained using a deep learning model to enhance accuracy.

## Features
Bidirectional LSTM (BiLSTM) Model: The core of this project is a BiLSTM neural network, designed to learn the sequential patterns of sign language gestures.
190-class Arabic Sign Language Dataset: The current model recognizes both letters and numbers, as well as 20 words-based classes. The project is designed to expand to all 190 distinct classes.
Comprehensive Data Preprocessing: MediaPipe is used for hand landmark detection and tracking to preprocess gesture images.
Accuracy: The model achieves an accuracy of approximately 92.25% for number and letter recognition and 99.37% for word recognition.

### Installation
To run the project, the following Python libraries are required:

pip install mediapipe
pip install tensorflow
pip install keras
pip install numpy
pip install pandas
pip install seaborn
pip install opencv-python

### Usage
Clone the repository:

git clone https://github.com/yourusername/Arabic-Sign-Language-Recognition.git
cd Arabic-Sign-Language-Recognition
Run the Jupyter Notebook: The core functionality is implemented in Jupyter notebooks. Run the notebook files (arslrecogniation-numbers-arabicletters-acc-92.ipynb and ARSLGraduation_20ClassFinalVersion.ipynb) to train the model or evaluate the existing implementation.

### Training the Model
The notebooks include all necessary steps for training, including data preprocessing, model architecture, and evaluation metrics.

### Dataset
The dataset contains Arabic Sign Language gestures, comprising hand poses for numbers, letters, and eventually words and verbs. The current phase focuses on numbers, letters, and words recognition, with the following structure:

### Numbers:
The numbers dataset includes a variety of Arabic number signs:

0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1000000, 10000000
Arabic Letters:
The letters dataset includes the following Arabic letters:

ا, ب, ت, ث, ج, ح, خ, د, ذ, ر, ز, س, ش, ص, ض, ط, ظ, ع, غ, ف, ق, ك, ل, م, ن, ه, و, ي, ة, أ, ؤ, ئ, ئـ, ء, إ, آ, ى, لا, ال
### Words:
The words dataset contains the following 20 Arabic words:

يبني (Build)
يكسر (Break)
يمشي (Walk)
يحب (Love)
يكره (Hate)
يشوي (Grill)
يحرث (Plow)
يزرع (Plant)
يسقي (Water)
يحصد (Harvest)
يفكر (Think)
يساعد (Help)
يدخن (Smoke)
يدعم (Support)
يختار (Choose)
ينادي (Call)
يتنامى (Grow)
يصبغ (Dye)
يقف (Stand)
يستحم (Shower)
### Model Architecture
The architecture includes:

Bidirectional LSTM: For capturing temporal dependencies in gestures.
Dense Layers: To output predictions for the respective number, letter, or word classes.
Dropout Layers: For regularization and preventing overfitting.
Conv2D Layers: For image processing during the preprocessing phase.
Results
The model achieves an accuracy of 92.25% on test data for the recognition of numbers and letters. The detailed model performance can be evaluated using the confusion matrix and classification report available in the notebooks.
For the word recognition model (20 classes), it achieves a categorical accuracy of 99.37% with a loss of 0.0095.

### Future Work
Expand the model to include all 190 classes, covering more words, verbs, and additional signs.
Improve real-time recognition by integrating with video data streams for live sign language interpretation.
