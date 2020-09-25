from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2

# construim parserul pentru argumente
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
                help="path to out input directory of images")
ap.add_argument("-m", "--model", required=True,
                help="path to pre-trained model")
args = vars(ap.parse_args())

# incarcam reteaua deja antrenata
print("[INFO] loading pre-trained network...")
model = load_model(args["model"])

# colectam toate caile catre imaginile pe care le vom evalua
imagePaths = list(paths.list_images(args["images"]))

# aceasta lista va contine predictiile
results = []

# fisierul in care ne vom salva predictiile
out_file = open('submission.csv', 'w+')
out_file.write('name,label\n')

# parcurgem toate imaginile
for p in imagePaths:
    # incarma imaginea originala
    orig = cv2.imread(p)

    # pre-procesam imaginea si o convertim din BGR in RGB
    # (deoarece modelul nostru a fost antrenat pe RGB),
    # o redimensionam la 64x64 pixeli, apoi scalam intensitatile pixelilor
    # in intervalul [0, 1]
    image = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (64, 64))
    image = image.astype("float") / 255.0

    # ordonam dimensiunile canalului (channels-first sau channels-last)
    # in functie de backendul Keras, apoi adaugam dimensiunea unui batch la imagine
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # facem predictii pe imagine
    pred = model.predict(image)
    pred = pred.argmax(axis=1)[0]

    # scriem in fisier predictia
    # dar o inversam deoarece submisia model era invers fata de valorile etichetelor noastre
    # astfel cu masca in submisie era de fapt fara masca si invers
    print("Evaluam {}".format(p.split("/")[-1]))
    if pred == 1:
        eticheta = 0
    if pred == 0:
        eticheta = 1

    out_file.write(p.split("/")[-1].replace(".png", ".wav").replace("with_mask", "").replace("without_mask", "") + "," +
                   str(eticheta) + "\n")

out_file.close()
