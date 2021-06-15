import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random


def load_mnist(path, kind="train"):
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        # 'I'表示一个无符号整数，大小为四个字节
        # '>II'表示读取两个无符号整数，即8个字节
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

x_test, y_test = load_mnist('./datasets/MNIST_data/', kind="t10k")

model = tf.keras.models.load_model('mnist.h5')
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
predictnum = []
stdnum=[]
for i in range(10):
    idx=random.randint(0,1000)
    img = np.array(x_test[idx]).reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    stdnum.append(y_test[idx])
    preimg = np.array(x_test[idx]).reshape((1, 28 * 28))
    predictnum.append(np.argmax(model.predict(preimg)))
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
print("标准测试值:    ",stdnum)
print("随机测试预测值:", predictnum)