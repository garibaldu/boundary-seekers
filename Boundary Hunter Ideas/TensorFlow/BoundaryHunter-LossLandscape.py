import tensorflow as tf
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import random
import math

plt.switch_backend("TkAgg")

def generateChevronData():
    xBounds = [-50, 50]
    yBounds = [-50, 50]
    totalPoints = 100
    
    points = []
    targets = []
    
    for i in range(0, totalPoints):
        x = random.randint(xBounds[0], xBounds[1])
        y = random.randint(yBounds[0], yBounds[1])
        
        if x >= y and x <= -y:
            points.append([x/50.0,y/50.0])
            targets.append(0)
        else:
            points.append([x/50.0,y/50.0])
            targets.append(1)
        
    return np.array(points), np.array(targets)
    
def plotScatter(points, color):
    xs = [x[0] for x in points]
    ys = [y[1] for y in points]
    
    plt.scatter(xs, ys, c=color)


def sigmoid(tensor):
    return 1.0/(1.0 + tf.exp(-tensor))

C = tf.constant(2.0, 'float64', name='C')
B = tf.constant(1.0, 'float64', name='B')

# Construct Weights
#weights = tf.Variable(np.random.rand(4), 'weights')

#weights = tf.Variable(np.array([-1.0, 1.0]), 'weights', dtype='float64')
radius = tf.Variable(0.3, 'radius', dtype='float64')
normal = tf.constant([-1.0, 1.0], dtype='float64')
point = tf.placeholder('float64', [None])

# Set up inputs/outputs to the network
inputs = tf.placeholder('float64', [None, 2], name='inputs')
targets = tf.placeholder('float64', [None], name='targets')

# Transform inputs
inputs_prime = tf.subtract(inputs, point)

# Compute the predictions
predictions = sigmoid(tf.matmul(inputs_prime, tf.expand_dims(normal, 1)))

# Compute the responsibility
responsiblity = 1 - (1/(1 + tf.exp(-(0.8)*(tf.reduce_sum(tf.pow(inputs_prime, 2), 1) - radius))))

# Compute the probobality of being right
p_right = tf.multiply(tf.pow(predictions, tf.expand_dims(targets, 1)), tf.pow(1-predictions, tf.expand_dims(1-targets, 1)))

# Compute the probobality of being wrong
p_wrong = tf.multiply(tf.pow(predictions, tf.expand_dims(1-targets, 1)), tf.pow(1-predictions, tf.expand_dims(targets, 1)))

profit = tf.reduce_sum(tf.multiply(tf.expand_dims(responsiblity, 1), tf.subtract(tf.multiply(C, p_wrong), tf.multiply(B, p_right))))


# Set up the data
random.seed(1234)
points, out = generateChevronData()
model = tf.global_variables_initializer()

x = np.arange(-1.5, 1.55, 0.02)
y = np.arange(-1.5, 1.55, 0.02)

z = np.zeros((len(x), len(y))).tolist()

with tf.Session() as session:
    session.run(model)

    for i in range(len(x)):
        for j in range(len(y)):
            z[i][j] = session.run(profit, feed_dict={point:[x[i],y[j]], inputs:points.tolist(), targets:out.tolist()})

#zi = griddata(x,y,z, x,y,interp='linear')
plt.pcolormesh(x,y,z)
plt.colorbar()
#plt.scatter(x, y, c=z,edgecolors='none', norm=matplotlib.colors.LogNorm())
plt.show()


