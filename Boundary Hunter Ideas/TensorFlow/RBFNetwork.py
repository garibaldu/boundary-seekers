import tensorflow as tf
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import random
import math

in_size = 2
num_centroids = 4

inputs = tf.placeholder('float64', [None])

centroids = tf.Variable(np.random.uniform(low=-1.0, high=1.0, size=(num_centroids, in_size)))
betas = tf.Variable(np.repeat(1.0, num_centroids))
