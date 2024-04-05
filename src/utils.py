import os
import cv2
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from tensorflow.keras.models import load_model


animal_classes = [
    "antelope", "badger", "bat", "bear", "bee", "beetle", "bison", "boar", "butterfly",
    "cat", "caterpillar", "chimpanzee", "cockroach", "cow", "coyote", "crab", "crow", "deer",
    "dog", "dolphin", "donkey", "dragonfly", "duck", "eagle", "elephant", "flamingo", "fly",
    "fox", "goat", "goldfish", "goose", "gorilla", "grasshopper", "hamster", "hare", "hedgehog",
    "hippopotamus", "hornbill", "horse", "hummingbird", "hyena", "jellyfish", "kangaroo", "koala",
    "ladybugs", "leopard", "lion", "lizard", "lobster", "mosquito", "moth", "mouse", "octopus",
    "okapi", "orangutan", "otter", "owl", "ox", "oyster", "panda", "parrot", "pelecaniformes",
    "penguin", "pig", "pigeon", "porcupine", "possum", "raccoon", "rat", "reindeer", "rhinoceros",
    "sandpiper", "seahorse", "seal", "shark", "sheep", "snake", "sparrow", "squid", "squirrel",
    "starfish", "swan", "tiger", "turkey", "turtle", "whale", "wolf", "wombat", "woodpecker", "zebra"
]


def generate_df(path):
    data = {'imgpath': [] , 'labels': [] }

    category = os.listdir(path)

    for folder in category:
        folderpath = os.path.join(path, folder)
        filelist = os.listdir(folderpath)

        for file in filelist:
            fpath = os.path.join(folderpath, file)
            data['imgpath'].append(fpath)
            data['labels'].append(folder)

    return pd.DataFrame(data)


class DatasetHandler:
    def __init__(self, dataset):
        self.dataset = dataset

    def split_data(self, train_size = 0.8, val_size = 0.8, random_state = 0):
        train_df, temp_df = train_test_split(self.dataset, train_size = train_size, shuffle = True, random_state = random_state)
        val_df, test_df = train_test_split(temp_df, train_size = val_size, shuffle = True, random_state = random_state)

        train_df = train_df.reset_index(drop = True)
        val_df = val_df.reset_index(drop = True)
        test_df = test_df.reset_index(drop = True)

        return train_df, val_df, test_df

    def dataset_info(self):
        print("----------Dataset Info-------------")
        print(self.dataset[['imgpath', 'labels']].head(5))
        print(self.dataset.shape)


class ModelEvaluator:
    def __init__(self, images, model):
        self.images = images
        self.model = model

    def classification_report(self):
        y_true = self.images.classes
        y_pred = self.model.predict(self.images)
        y_pred = np.argmax(y_pred, axis = 1)
        f1 = f1_score(y_true, y_pred, average = 'macro')

        print("F1 Score:", f1)
        print(classification_report(y_true, y_pred, target_names = self.images.class_indices.keys()))

        return f1

    def accuracy(self):
        results = self.model.evaluate(self.images, verbose=0)
        loss = results[0]
        accuracy = results[1] * 100

        print(f"{self.images} Loss: {loss:.5f}")
        print(f"{self.images} Accuracy: {accuracy:.2f}%")

    def plot_confusion_matrix(self):
        preds = self.model.predict(self.images)
        y_pred = np.argmax(preds, axis=1)
        g_dict = self.images.class_indices
        classes = list(g_dict.keys())

        # Confusion matrix
        cm = confusion_matrix(self.images.classes, y_pred)

        plt.figure(figsize=(30, 30))
        plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()

        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation = 45)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j], horizontalalignment = 'center', color = 'white' if cm[i, j] > thresh else 'black')

        plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        plt.show()

def load_trained_model(base_path, model_folder, model_filename):
    model_path = os.path.join(base_path, model_folder, model_filename)
    model = load_model(model_path)
    return model


def predict_image_class(model, base_path, img_folder, img_filename, animal_classes):
    img_path = os.path.join(base_path, img_folder, img_filename)
    img = cv2.imread(img_path)
    img = np.expand_dims(cv2.resize(img, (224, 224)), axis=0)
    predictions = np.argmax(model.predict(img), axis=1)
    predicted_class = animal_classes[predictions[0]]
    return predicted_class

def predict_image_class_from_bytes(model, image):
    img = cv2.imdecode(np.asarray(bytearray(image.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(cv2.resize(img, (224, 224)), axis = 0)
    predictions = np.argmax(model.predict(img), axis = 1)
    return animal_classes[predictions[0]]