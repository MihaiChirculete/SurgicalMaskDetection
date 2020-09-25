# utilizam back-endul matplotlib cu "Agg" pentru a putea salva figurile in background
import matplotlib

matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from pyimagesearch.resnet import ResNet
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# construim parserul pentru argumente
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
                help="calea la care va fi salvat modelul")
ap.add_argument("-e", "--epochs", type=int, required=True, help="numar epoci")
args = vars(ap.parse_args())

# initializam numarul de epoci pentru care va fi antrenat modelul si dimensiunea pachetelor folosite
NUM_EPOCHS = args['epochs']
BS = 32

# initializam calea de baza ce contine folderele cu datasetul construit
# (datasetul de spectrograme ale sunetelor)
BASE_PATH = "data/spectrograms/"

# initializam caile catre folderele de antrenare, validare si testare
# (Atentie: folderul de testare nu este cel pentru testul final, contine 1000 imagini
# preluate din setul de antrenare)
TRAIN_PATH = os.path.sep.join([BASE_PATH, "train"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "test"])

# determinam numarul total de cai catre imaginile din seturile
# de antrenare, validare si testare
totalTrain = len(list(paths.list_images(TRAIN_PATH)))
totalVal = len(list(paths.list_images(VAL_PATH)))
totalTest = len(list(paths.list_images(TEST_PATH)))

# initializam obiectul de augmentare al datasetului de spectrograme de antrenare
trainAug = ImageDataGenerator(
    rescale=1 / 255.0,
    rotation_range=20,
    zoom_range=0.05,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    horizontal_flip=True,
    fill_mode="nearest")

# initializam obiectul de augmentare al datasetului de validare si testare
valAug = ImageDataGenerator(rescale=1 / 255.0)

# initializam generatorul de antrenare
trainGen = trainAug.flow_from_directory(
    TRAIN_PATH,
    class_mode="categorical",
    target_size=(64, 64),
    color_mode="rgb",
    shuffle=True,
    batch_size=BS)

# initializam generatorul de validare
valGen = valAug.flow_from_directory(
    VAL_PATH,
    class_mode="categorical",
    target_size=(64, 64),
    color_mode="rgb",
    shuffle=False,
    batch_size=BS)

# initializam generatorul de testare
testGen = valAug.flow_from_directory(
    TEST_PATH,
    class_mode="categorical",
    target_size=(64, 64),
    color_mode="rgb",
    shuffle=False,
    batch_size=BS)

# initializam implementarea Keras a modelului ResNet si il compilam
model = ResNet.build(64, 64, 3, 2, (2, 2, 3),
                     (32, 64, 128, 256), reg=0.0005)
opt = SGD(lr=1e-1, momentum=0.9, decay=1e-1 / NUM_EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# antrenam modelul Keras
H = model.fit_generator(
    trainGen,
    steps_per_epoch=totalTrain // BS,
    validation_data=valGen,
    validation_steps=totalVal // BS,
    epochs=NUM_EPOCHS)

# resetam generatorul de testare si apoi utilizam modelul antrenat
# pentru a face predictii pe date
print("[INFO] evaluating network...")
testGen.reset()
predIdxs = model.predict_generator(testGen,
                                   steps=(totalTest // BS) + 1)


# pentru fiecare imagine din setul de testare trebuie sa gasim
# indexul etichetei cu probabilitatea prezisa maxima
predIdxs = np.argmax(predIdxs, axis=1)


# afisam un raport al clasificarii realizate
print(classification_report(testGen.classes, predIdxs,
                            target_names=testGen.class_indices.keys()))


# serializam reteaua antrenata
print("[INFO] serializing network to '{}'...".format(args["model"]))
model.save(args["model"])

# plot the training loss and accuracy
# plotam pierderile si acuratetea antrenarii
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args['model'] + "_evaluation.png")
