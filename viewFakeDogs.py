import numpy as np
from matplotlib import pyplot as plt

images = np.load("genOut300e40kIm857.npy")
# Load the numpy array, that saved after training

print(images.shape)

for i in range(images.shape[0]):
    plt.figure()
    plt.imshow(images[i])
    plt.show()
# loop through the fake outputs by epochs
