import numpy as np
import random
import math
import datetime


class NeuralNetwork:
    def __init__(self, images, p, n, min_error):
        """
        :param images: array of input images
        :param p: int count of output images
        :param n: int count of values in a image
        """
        self.images = images
        self.p = p
        self.n = n
        self.min_error = min_error
        self.weights = np.zeros((self.n, self.p), dtype=np.float64)
        self.weights_ = np.zeros((self.p, self.n), dtype=np.float64)
        self.fill_weights()

    def fill_weights(self):
        """ This method fills randomly matrices of weights with value [-1, 1] """
        for i in range(self.n):
            for j in range(self.p):
                self.weights[i, j] = self.weights_[j, i] = random.uniform(-1, 1)

    def load_weights(self):
        """ This method loads arrays of weights from config files """
        self.weights = np.loadtxt('weights/direct_weights.txt')
        self.weights_ = np.loadtxt('weights/inverse_weights.txt')

    def save_weights(self):
        """ This method saves arrays of weights to config files """
        np.savetxt('weights/direct_weights.txt', self.weights)
        np.savetxt('weights/inverse_weights.txt', self.weights_)

    def get_error(self, x, x_):
        """ This method calculate error between recovered and source images
        :param x: array - source image
        :param x_: array - recovered image
        :return: float - error between values of recovered and source images
        """
        e = 0

        for i in range(self.n):
            e += pow(x_[i] - x[i], 2)

        return e /2


    def compute_weights(self, ii, oi, ii_, gamma, alpha, alpha_):
        """ This method corrects weights for first and second layers
        :param ii: array input image
        :param oi: array output image
        :param ii_: array recovered image
        :param gamma: array of gamma values
        :param alpha: float alpha for fist layer
        :param alpha_: float alpha for second layer
        """
    
        for i in range(self.n):
            for j in range(self.p):
                self.weights[i, j] -= alpha * gamma[j] * ii[i]
                self.weights_[j, i] -= alpha_ * oi[j] * (ii_[i] - ii[i])


    def compute_adaptive_step(self, x, size):
        """ Compute adaptive step for passed layer
        :param x: array image of first of second layer
        :param size: int array size
        :return: float adaptive step
        """
        s = 1

        for i in range(size):
            s += x[i]**2

        return 1. / s

    def compute_gamma(self, gamma, ii, ii_):
        """ This method calculate gamma
        :param gamma: array for updating
        :param ii: array input image
        :param ii_: array recovered image
        :return: gamma
        """
        for j in range(self.p):
            gamma[j] = 0
            for i in range(self.n):
                gamma[j] += (ii_[i] - ii[i]) * self.weights_[j, i]

        return gamma

    def training(self):
        """ This method starts training net (calculate weights) using Backpropagation algorithm """
        e = self.min_error + 1
        exp_min = 10000000
        iteration = 0
        gamma = np.zeros(self.p, dtype=np.float64)

        while e >= self.min_error:
            e = 0
            iteration += 1
            start_time = datetime.datetime.now()

            for ii in self.images:
                oi = np.dot(ii, self.weights)
                ii_ = np.dot(oi, self.weights_)

                alpha = 0.0002  # self.compute_adaptive_step(ii, self.n)
                alpha_ = 0.0002  # self.compute_adaptive_step(oi, self.p)
                gamma = self.compute_gamma(gamma, ii, ii_)

                self.compute_weights(ii, oi, ii_, gamma, alpha, alpha_)

            for ii in self.images:
                oi = np.dot(ii, self.weights)
                ii_ = np.dot(oi, self.weights_)
                e += self.get_error(ii, ii_)

            delta_time = datetime.datetime.now() - start_time

            if e < exp_min:
                exp_min = e

            print(iteration, 'step, err =', e, 'exp_min =', exp_min, 'time', delta_time.microseconds / 1000., 'ms')

        self.save_weights()

    def process(self):
        rec_images = []
        for ii in self.images:
            oi = np.dot(ii, self.weights)
            ii_ = np.dot(oi, self.weights_)

            rec_images.append(ii_)

        return rec_images
