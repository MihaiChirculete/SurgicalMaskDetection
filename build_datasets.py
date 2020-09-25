import threading

import cv2
import librosa as lr
import matplotlib.pyplot as plt
import numpy as np
import utils
import shutil
import os

# Acest script este folosit pentru a construi un dataset de spectrograme din datasetul de wav-uri



def generate_training_spectrograms(train_data, start_index=0, stop_index=8000, path='data/spectrograms/train/'):
    for i in range(start_index, stop_index):
        spectrogram = utils.audio_to_image('data/train/' + train_data[i][0])
        plt.imshow(spectrogram.astype(np.float))

        # daca eticheta este 1 inseamna ca audioul este pt cineva cu masca
        # salvam spectrograma in folderul de antrenament cu masti
        # (analog pt 0)
        if train_data[i][1] == 1:
            print('Generating spectrogram for {} sound with mask. Reached file {}/{}'.format(train_data[i][0], i,
                                                                                             stop_index))
            plt.savefig(fname=path + 'with_mask/' + str(train_data[i][0]).replace('wav', ''))
            plt.close()

            # cropam imaginea
            img = cv2.imread(path + 'with_mask/' + str(train_data[i][0]).replace('wav', 'png'))
            crop_img = img[60:425, 145:511]
            cv2.imwrite(path + 'with_mask/' + str(train_data[i][0]).replace('wav', 'png'), crop_img)

        if train_data[i][1] == 0:
            print('Generating spectrogram for {} sound without mask. Reached file {}/{}'.format(train_data[i][0], i,
                                                                                                stop_index))
            plt.savefig(fname=path + 'without_mask/' + str(train_data[i][0]).replace('wav', ''))
            plt.close()

            # cropam imaginea
            img = cv2.imread(path + 'without_mask/' + str(train_data[i][0]).replace('wav', 'png'))
            crop_img = img[60:425, 145:511]
            cv2.imwrite(path + 'without_mask/' + str(train_data[i][0]).replace('wav', 'png'), crop_img)


def generate_validation_spectrograms(validation_data, start_index=0, stop_index=1000, path='data/spectrograms/validation/'):
    for i in range(start_index, stop_index):
        spectrogram = utils.audio_to_image('data/validation/' + validation_data[i][0])
        plt.imshow(spectrogram.astype(np.float))

        # daca eticheta este 1 inseamna ca audioul este pt cineva cu masca
        # salvam spectrograma in folderul de validare cu masti
        # (analog pt 0)
        if validation_data[i][1] == 1:
            print('Generating validation spectrogram for {} sound with mask. Reached file {}/{}'.format(validation_data[i][0], i,
                                                                                             stop_index))
            plt.savefig(fname=path + 'with_mask/' + str(validation_data[i][0]).replace('wav', ''))
            plt.close()

            # cropam imaginea
            img = cv2.imread(path + 'with_mask/' + str(validation_data[i][0]).replace('wav', 'png'))
            crop_img = img[60:425, 145:511]
            cv2.imwrite(path + 'with_mask/' + str(validation_data[i][0]).replace('wav', 'png'), crop_img)

        if validation_data[i][1] == 0:
            print('Generating validation spectrogram for {} sound without mask. Reached file {}/{}'.format(validation_data[i][0], i,
                                                                                                stop_index))
            plt.savefig(fname=path + 'without_mask/' + str(validation_data[i][0]).replace('wav', ''))
            plt.close()

            # cropam imaginea
            img = cv2.imread(path + 'without_mask/' + str(validation_data[i][0]).replace('wav', 'png'))
            crop_img = img[60:425, 145:511]
            cv2.imwrite(path + 'without_mask/' + str(validation_data[i][0]).replace('wav', 'png'), crop_img)
    return


# din spectrogramele de antrenare luam 500 cu masca pt testare si 500 fara
def generate_testing_spectrograms(train_path='data/spectrograms/train/', test_path='data/spectrograms/test/'):

    train_dir_with_mask = os.fsencode(train_path + 'with_mask/')
    train_dir_whihtout_mask = os.fsencode(train_path + 'without_mask/')

    i = 0
    for file in os.listdir(train_dir_with_mask):
        if i >= 500:
            break
        print("Moving spectrogram #{} with mask from train to test.".format(i))
        filename = os.fsdecode(file)
        shutil.move(train_path + 'with_mask/' + filename, test_path + 'with_mask/' + filename)
        i += 1

    i = 0
    for file in os.listdir(train_dir_whihtout_mask):
        if i >= 500:
            break
        print("Moving spectrogram #{} without mask from train to test.".format(i))
        filename = os.fsdecode(file)
        shutil.move(train_path + 'without_mask/' + filename, test_path + 'without_mask/' + filename)
        i += 1
    return


# jumatate se duc in 'with_mask' si jumatate in 'without_mask'
def generate_final_test_spectrograms(test_data, start_index=0, stop_index=3000, path='data/spectrograms/final_test/'):
    for i in range(start_index, stop_index):
        spectrogram = utils.audio_to_image('data/test/' + test_data[i])
        plt.imshow(spectrogram.astype(np.float))

        print('Generating final test spectrogram for {}. Reached file {}/{}'.format(test_data[i], i,
                                                                              stop_index))

        # prima jumatate merge in 'with_mask'
        plt.savefig(fname=path + str(test_data[i]).replace('wav', ''))

        # cropam imaginea
        img = cv2.imread(path + str(test_data[i]).replace('wav', 'png'))
        crop_img = img[60:425, 145:511]
        cv2.imwrite(path + str(test_data[i]).replace('wav', 'png'), crop_img)

        plt.close()


    return


def generate_all():
    # vector de forma (fisier.wav, eticheta)
    train_data = np.genfromtxt('data/train.txt', dtype=None, encoding=None, delimiter=',')

    # contine doar nume de fisiere wav
    final_test_data = np.genfromtxt('data/test.txt', dtype=None, encoding=None)

    # ca la trainining data
    validation_data = np.genfromtxt('data/validation.txt', dtype=None, encoding=None, delimiter=',')

    # cu asta generam imaginile de antrenare
    generate_training_spectrograms(train_data, 0, train_data.size)

    # cu asta mutam 1000 de imagini de antrenare in folderul de testare
    generate_testing_spectrograms()

    # cu asta generam imaginile de testare
    generate_final_test_spectrograms(final_test_data, 0, final_test_data.size)

    # cu asta generam imaginile de validare
    generate_validation_spectrograms(validation_data, 0, validation_data.size)