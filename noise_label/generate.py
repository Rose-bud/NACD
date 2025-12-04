import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import scipy.io as sio
import os
import numpy as np
import h5py
from tqdm import tqdm
import pdb


def _generate_noisy_labels(noisy_labels, noise_level):

    noisy_labels_tensor = torch.tensor(noisy_labels, dtype=torch.float32)

    num_labels = torch.sum(noisy_labels_tensor == 1, dim=1)
    print(num_labels)
    num_noisy_labels = torch.round(noise_level * num_labels).type(torch.int)
    print(num_noisy_labels)

    num_noisy_labels[num_noisy_labels > noisy_labels_tensor.size(1)] = (
        noisy_labels_tensor.size(1)
    )

    rand_mat = torch.rand(noisy_labels_tensor.size())
    rand_mat[noisy_labels_tensor == 1] = 0
    for i in range(noisy_labels_tensor.size(0)):
        _, inds = torch.topk(rand_mat[i], k=num_noisy_labels[i].item())
        noisy_labels_tensor[i].index_fill_(0, inds, 1)

    return noisy_labels_tensor.numpy()


def generate_noise_F(noise):
    noise_rate = noise
    data = h5py.File("./data/MIRFlickr.h5", "r")
    for i in noise_rate:
        labels_matrix = np.array(list(data["LabTrain"]))
        labels_matrix2 = np.array(list(data["LabTrain"]))
        noisy_labels_matrix = _generate_noisy_labels(labels_matrix, i)

        output_file = h5py.File(
            "./noise_label/mirflickr25k-lall-noise_{}.h5".format(i), "w"
        )

        output_file.create_dataset("result", data=noisy_labels_matrix)
        output_file.create_dataset("True", data=labels_matrix2)

        output_file.close()


def generate_noise_N(noise):
    noise_rate = noise
    data = h5py.File("./data/NUS-WIDE.h5", "r")
    for i in noise_rate:
        labels_matrix = np.array(list(data["LabTrain"]))
        labels_matrix2 = np.array(list(data["LabTrain"]))
        noisy_labels_matrix = _generate_noisy_labels(labels_matrix, i)

        output_file = h5py.File("./noise_label/nus-wide-tc21-lall-noise_{}.h5".format(i), "w")

        output_file.create_dataset("result", data=noisy_labels_matrix)
        output_file.create_dataset("True", data=labels_matrix2)

        output_file.close()


def generate_noise_M(noise):
    noise_rate = noise
    data = h5py.File("./data/MS-COCO.h5", "r")
    for i in noise_rate:
        labels_matrix = np.array(list(data["LabTrain"]))
        labels_matrix2 = np.array(list(data["LabTrain"]))
        noisy_labels_matrix = _generate_noisy_labels(labels_matrix, i)

        output_file = h5py.File("./noise_label/MSCOCO-lall-noise_{}.h5".format(i), "w")

        output_file.create_dataset("result", data=noisy_labels_matrix)
        output_file.create_dataset("True", data=labels_matrix2)

        output_file.close()



if __name__ == "__main__":
    noise_rate = [1.0, 1.5, 2.0, 2.5]
    generate_noise_F(noise_rate)
    # generate_noise_M(noise_rate)
    # generate_noise_N(noise_rate)
