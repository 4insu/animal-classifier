# Animal Classifier

Welcome to the Animal Classifier repository! This project is aimed at classifying over 90 different animal species using transfer learning with the EfficientNetB3 model.

## Overview

This project utilizes transfer learning by fine-tuning the pre-trained EfficientNetB3 model to accurately classify various animal species. The EfficientNetB3 model, known for its efficiency and effectiveness in handling image classification tasks, serves as a robust backbone for this purpose.

## Dataset

The dataset used for training and evaluation comprises images of over 90 different animal species. It has been carefully curated and labeled to ensure accurate classification. Due to the diverse nature of the dataset, the model is capable of recognizing a wide range of animal species. It can be downloaded from [here](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals)

## Model

EfficientNetB3, a convolutional neural network architecture, serves as the backbone for this animal classifier. Transfer learning is employed to fine-tune the model on the specific task of classifying animal species. By reusing the pre-trained weights and adjusting the final layers, the model achieves high accuracy even with a relatively small dataset.

<div align = "center">
    <img src = "images/imgreadme/model.png" alt = "DEMO 1" />
</div>

## Usage

To utilize the animal classifier, follow these steps:

1. **Clone the Repository**: Begin by cloning this repository to your local machine.

   ```bash
   git clone https://github.com/4insu/animal-classifier.git
   ```

2. **Create and Activate Virtual Environment**:

   ```bash
   conda create -n <env_name> python=3.9 -y
   conda activate <env_name>
   ```

3. **Install Dependencies**: Ensure you have all the necessary dependencies installed. You may use requirements.txt to install them using pip.

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Classifier**: Use the provided scripts or notebooks to run the classifier on your own images or test it on your own dataset.

   ```bash
   streamlit run app.py
   ```

## Results

The model achieves an impressive accuracy of 92% and f1-score of 90% on the test set, trained over 10 epochs. Augmentation techniques were employed to effectively reduce overfitting, enhancing the model's ability to generalize to unseen data. By training for more epochs, there's a possibility of achieving even better accuracy and f1-score.

## Web Application Screenshots

<p float = "left">
  <img src = "images/imgreadme/panda.png" alt = "DEMO 1" width = "400" />
  <img src = "images/imgreadme/elephant.png" alt = "DEMO 2" width = "400" /> 
</p>

## Contributing

Contributions to this project are welcome! Whether it's improving the model's performance, expanding the dataset, or enhancing the documentation, feel free to submit pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.