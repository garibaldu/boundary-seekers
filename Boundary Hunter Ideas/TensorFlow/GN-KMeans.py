import tensorflow as tf
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import scipy.stats as stats

np.random.seed(1234)
random.seed(1234)

plt.switch_backend("TkAgg")

def plotScatter(points, color):
    xs = [x[0] for x in points]
    ys = [y[1] for y in points]
    
    plt.scatter(xs, ys, c=color)

def plot_weights(weights, gate, center, color):
    plot_centroid(center)
    plot_line(weights, center, color)
    plot_line(gate, center, 'r')

    #print("B: " + str(byas))
    #print("XCoef: " + str(Xcoef))

def plot_line(weights, center, color):
    n = np.array([weights[0] * center[0] + weights[1] * center[1], 
            -weights[0], 
            -weights[1]])

    byas = -1 * n[0]/n[2]
    Xcoef = -1 * n[1]/n[2]

    plt.plot([-1.0, 1.0], [-1*Xcoef + byas, Xcoef + byas], '{}-'.format(color))

def plot_centroid(centroid):
    plt.plot(centroid[0], centroid[1], markersize=10, marker='x', color='g', mew=5)

def plot_incorrect(point):
    plt.plot(point[0], point[1], markersize=5, marker='x', color='r', mew=5)

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

def generate_split_data():
    xBounds = [-50, 50]
    yBounds = [-50, 50]
    totalPoints = 100
    
    points = []
    targets = []
    
    for i in range(0, totalPoints):
        x = random.randint(xBounds[0], xBounds[1])
        y = random.randint(yBounds[0], yBounds[1])
        
        if x < 25 and x > -25 :
            points.append([x/50.0,y/50.0])
            targets.append(0.0)
        else:
            points.append([x/50.0,y/50.0])
            targets.append(1.0)
        
    return np.array(points), np.array(targets)


def generate_clumps():
    xBounds = [-50, 50]
    yBounds = [-50, 50]
    totalPoints = 100
    
    points = []
    targets = []
    
    for i in range(0, int(totalPoints/2.0)):
        x = random.randint(xBounds[0], 0)
        y = random.randint(yBounds[0], 0)

        if -x - 30 < y:
            points.append([x/50.0,y/50.0])
            targets.append(1.0)
        else:
            points.append([x/50.0,y/50.0])
            targets.append(0.0)

    for i in range(0, int(totalPoints/2.0)):
        x = random.randint(0, xBounds[1])
        y = random.randint(0, yBounds[1])
        
        if -x + 30 > y:
            points.append([x/50.0,y/50.0])
            targets.append(1.0)
        else:
            points.append([x/50.0,y/50.0])
            targets.append(0.0)
        
    return np.array(points), np.array(targets)


def generate_rectangle_data():
    xBounds = [-50, 50]
    yBounds = [-50, 50]
    totalPoints = 100
    
    points = []
    targets = []
    
    for i in range(0, totalPoints):
        x = random.randint(xBounds[0], xBounds[1])
        y = random.randint(yBounds[0], yBounds[1])
        
        if np.abs(x) < 30 and np.abs(y) < 30 :
            points.append([x/50.0,y/50.0])
            targets.append(0.0)
        else:
            points.append([x/50.0,y/50.0])
            targets.append(1.0)
        
    return np.array(points), np.array(targets) 

def sigmoid(phi):
    return 1.0/(1.0 + tf.exp(-phi))


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

##def train_network(Tr, Ts, points, targets):
##
##    inputs = tf.placeholder('float64', [in_size])
##    targets = tf.placeholder('float64', [out_size])
##
##    centroids = tf.Variable(np.random.uniform(low=-1.0, high=1.0, size=(num_centroids, in_size)))
##    #betas = tf.Variable(np.repeat(1.0, num_centroids))
##    hidden_weights = tf.Variable(np.random.uniform(low=-0.5, high=0.5, size=(num_centroids, in_size)))
##    gate_weights = tf.Variable(np.random.uniform(low=-0.5, high=0.5, size=(num_centroids, in_size)))
##    output_weights = tf.Variable(np.random.uniform(low=-0.5, high=0.5, size=(num_outputs, num_centroids + 1)))
##
##    input_by_plane = lambda x: tf.subtract(inputs, x)
##    transformed_by_points = tf.map_fn(input_by_plane, centroids)
##
##    # Peform Computation
##    prob = tf.reduce_sum(tf.multiply(transformed_by_points, hidden_weights), 1)
##
##    #square_diff = lambda c: tf.reduce_sum(tf.pow(tf.subtract(inputs, c), 2.0))
##    #g = tf.exp(-1.0 * tf.multiply(betas, tf.map_fn(square_diff, centroids)))
##    g = tf.reduce_sum(tf.multiply(transformed_by_points, gate_weights), 1)
##    hidden_out = sigmoid(tf.multiply(g, prob))#tf.add(0.5 * (1 - g), tf.multiply(g, prob))
##    #gated = tf.multiply(g, prob)
##    #hidden_out = sigmoid(gated)
##    hidden_out_prime = tf.concat([[1.0], hidden_out], 0)
##
##    output = sigmoid(tf.matmul(tf.transpose(tf.expand_dims(hidden_out_prime, 1)), tf.transpose(output_weights)))
##    errors = tf.pow(tf.subtract(tf.expand_dims(targets, 1), output), 2.0)
##    error = tf.reduce_sum(errors)
##
##    train_op = tf.train.GradientDescentOptimizer(0.006).minimize(error)
##    #clip_op_betas = tf.assign(betas, tf.clip_by_value(betas, 0, np.infty))
##
##    model = tf.global_variables_initializer()
##
##    with tf.Session() as session:
##        session.run(model)
##        
##        for e in range(6000):
##            for d in range(len(Tr)):
##                session.run(train_op, feed_dict={inputs: points[Tr[d]], targets: [out[Tr[d]]]})
##
##        train_err = 0
##        for d in range(len(Tr)):
##            train_err += session.run(error, feed_dict={inputs: points[Tr[d]], targets: [out[Tr[d]]]})
##
##        test_err = 0
##        for d in range(len(Ts)):
##            test_err += session.run(error, feed_dict={inputs: points[Ts[d]], targets: [out[Ts[d]]]})
##
##    return (train_err/len(Tr)), (test_err/len(Ts))

def train_network(Tr, Ts, points, targets):

    inputs = tf.placeholder('float64', [in_size])
    targets = tf.placeholder('float64', [out_size])

    hidden_weights = tf.Variable(np.random.uniform(low=-0.5, high=0.5, size=(num_centroids, in_size+1)))
    gate_weights = tf.Variable(np.random.uniform(low=-0.5, high=0.5, size=(num_centroids, in_size+1)))
    output_weights = tf.Variable(np.random.uniform(low=-0.5, high=0.5, size=(num_outputs, num_centroids + 1)))

    inputs_prime = tf.concat([[1.0], inputs], axis=0)

    # Peform Computation
    prob = tf.reduce_sum(tf.multiply(inputs_prime, hidden_weights), 1)

    g = tf.reduce_sum(tf.multiply(inputs_prime, gate_weights), 1)
    hidden_out = sigmoid(tf.multiply(g, prob))
    hidden_out_prime = tf.concat([[1.0], hidden_out], 0)

    output = sigmoid(tf.matmul(tf.transpose(tf.expand_dims(hidden_out_prime, 1)), tf.transpose(output_weights)))
    errors = tf.pow(tf.subtract(tf.expand_dims(targets, 1), output), 2.0)
    error = tf.reduce_sum(errors)

    train_op = tf.train.GradientDescentOptimizer(0.006).minimize(error)

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
num_centroids = 1
num_outputs = 1

split = split_data(len(points), K)

train_errs = []
test_errs = []

for s in split:
    train_err, test_err = train_network(s[0], s[1], points, out)

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
