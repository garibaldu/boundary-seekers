import tensorflow as tf
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import scipy.stats as stats

np.random.seed(1234)
random.seed(1234)

def init_network(inputs, layers):
    network = []
    
    current_in = inputs
    for l in layers:
        layer = tf.Variable(-0.5 + np.random.rand(l, current_in + 1), dtype='float64')
        current_in = l

        network.append(layer)

    return network


def apply_network(network, inputs):
    current_out = inputs
    for layer in network:
        current_out = tf.concat([tf.expand_dims(np.repeat([1.0], current_out.shape[0]), 1), current_out], axis=1)
        current_out = sigmoid(tf.matmul(current_out, tf.transpose(layer)))

    return current_out

def sigmoid(tensor):
    return 1.0/(1.0 + tf.exp(-tensor))

def split_data(n, K):
    partitions = []
    
    idx = list(range(n))

    np.random.shuffle(idx)
    
    sub_size = int(len(idx)/K)
    for i in range(0, len(idx), sub_size):
        Tr = []
        Ts = []

        for j in range(0, len(idx)):
            if j >= i and j < (i+sub_size):
                Ts.append(idx[j])
            else:
                Tr.append(idx[j])

        partitions.append((Tr,Ts))

    return partitions

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
            targets.append(0.0)
        else:
            points.append([x/50.0,y/50.0])
            targets.append(1.0)
        
    return np.array(points), np.array(targets)

def train_perceptron_network(Tr, Ts, points, targets):

    inputs = tf.placeholder('float64', [in_size])
    targets = tf.placeholder('float64', [out_size])

    in_prime = tf.transpose(tf.expand_dims(inputs, 1))

    network = init_network(2, [2,1])
    output = apply_network(network, in_prime)

    errors = tf.pow(tf.subtract(tf.expand_dims(targets, 1), output), 2.0)
    error = tf.reduce_sum(errors)

    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(error)


    model = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(model)

        for e in range(6000):
            for d in range(len(Tr)):
                session.run(train_op, feed_dict={inputs: points[Tr[d]], targets: [out[Tr[d]]]})

        train_err = 0
        for d in range(len(Tr)):
            train_err += session.run(error, feed_dict={inputs: points[Tr[d]], targets: [out[Tr[d]]]})

        test_err = 0
        for d in range(len(Ts)):
            test_err += session.run(error, feed_dict={inputs: points[Ts[d]], targets: [out[Ts[d]]]})

    return (train_err/len(Tr)), (test_err/len(Ts))


def conf_interval(pop):
    z = z_critical = stats.norm.ppf(q = 0.95)
    moe = z * (pop.std()/math.sqrt(len(pop)))

    return (pop.mean() - moe, pop.mean() + moe)

K = 10
points, out = generateChevronData()
in_size = 2
out_size = 1

split = split_data(len(points), K)

train_errs = []
test_errs = []

for s in split:
    train_err, test_err = train_perceptron_network(s[0], s[1], points, out)

    train_errs.append(train_err)
    test_errs.append(test_err)
    
    print("Train Error: ", train_err)
    print("Test Error", test_err)
    print()


mean_train_err = np.array(train_errs).mean()
mean_test_err = np.array(test_errs).mean()

print("AVG Train Error: ", mean_train_err)
print("AVG Test Error: ", mean_test_err)

print("Train Conf: ", conf_interval(np.array(train_errs)))
print("Test Conf: ", conf_interval(np.array(test_errs)))
