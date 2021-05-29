from numpy import *
from scipy.cluster.vq import *

from PCV.localdescriptors import sift


class Vocabulary(object):

    def __init__(self, name):
        self.name = name
        self.voc = []
        self.idf = []
        self.trainingdata = []
        self.nbr_words = 0

    def train(self, featurefiles, k=100, subsampling=10):
        """ 用含有k个单词的K-means列出在featurefiles中的特征文件训练出一个词汇。对训练数据下采样可以加快训练速度 """

        nbr_images = len(featurefiles)
        # 从文件中读取特征
        descr = []
        descr.append(sift.read_features_from_file(featurefiles[0])[1])
        descriptors = descr[0]  # 将所有的特征并在一起，以便后面进行K-means聚类
        for i in arange(1, nbr_images):
            descr.append(sift.read_features_from_file(featurefiles[i])[1])
            descriptors = vstack((descriptors, descr[i]))

        # k-means: 最后一个参数决定运行次数
        self.voc, distortion = kmeans(descriptors[::subsampling, :], k, 1)
        self.nbr_words = self.voc.shape[0]

        # 遍历所有的训练图像，并投影到词汇上
        imwords = zeros((nbr_images, self.nbr_words))
        for i in range(nbr_images):
            imwords[i] = self.project(descr[i])

        nbr_occurences = sum((imwords > 0) * 1, axis=0)

        self.idf = log((1.0 * nbr_images) / (1.0 * nbr_occurences + 1))
        self.trainingdata = featurefiles

    def project(self, descriptors):
        """ 将描述子投影到词汇上，以创建单词直方图 """

        # 图像单词直方图
        imhist = zeros((self.nbr_words))
        words, distance = vq(descriptors, self.voc)
        for w in words:
            imhist[w] += 1

        return imhist

    def get_words(self, descriptors):
        """ Convert descriptors to words. """
        return vq(descriptors, self.voc)[0]