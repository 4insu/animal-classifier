import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "animalClassification"
list_of_files = [
    "models/animal_classification.h5",
    "research/model_predictions.ipynb",
    "research/model_training.ipynb",
    "images/deer.png",
    "src/__init__.py",
    "src/utils.py",
    "requirements.txt",
    "README.md",
    "Dockerfile",
    "setup.py",
    "app.py"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f'Creating directory: {filedir} for the file {filename}.')

    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        with open(filepath, 'w') as f:
            pass
            logging.info(f'Creating empty file: {filepath}')

    else:
        logging.info(f'{filename} already exists.')