# Plant Disease Detection Using Deep Learning

## B.Tech Project

A deep learning project for classifying plant leaf images into disease categories.

> **Project status:** Starter / development version. The repository includes the application and training pipeline. A trained model and final accuracy should be added after training on the selected dataset.

## Features

- Plant leaf image upload
- Image preprocessing
- Deep learning image classification
- Confidence score
- Streamlit web interface
- Training and prediction scripts
- Space for evaluation results

## Project Structure

```text
plant-disease-detection/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── train.py
│   └── predict.py
├── data/
│   └── README.md
├── models/
│   └── README.md
└── results/
    └── README.md
```

## Dataset

This project is designed to use the PlantVillage dataset.

Download the dataset from an appropriate public source and place the image folders under `data/`.

Do not upload a large dataset directly to this GitHub repository.

## Installation

```bash
pip install -r requirements.txt
```

## Run the application

```bash
streamlit run app.py
```

## Train the model

```bash
python src/train.py
```

The training script expects a dataset directory containing one folder per class.

## Technologies

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pillow

## Future Scope

- Improve model accuracy with transfer learning
- Add Grad-CAM explanations
- Add more plant species
- Add multilingual support
- Deploy the application online
- Add mobile-friendly inference

## Author

B.Tech Student Project
