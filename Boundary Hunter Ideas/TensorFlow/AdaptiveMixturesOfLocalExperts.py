import tensorflow as tf
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import random
import math

plt.switch_backend("TkAgg")

def plotScatter(points, color):
    xs = [x[0] for x in points]
    ys = [y[1] for y in points]
    
    plt.scatter(xs, ys, c=color)


def plot_weights(weights):
    byas = -1 * weights[0]/weights[2]
    Xcoef = -1 * weights[1]/weights[2]

    plt.plot([-1.0, 1.0], [-1*Xcoef + byas, Xcoef + byas], 'k-')

    print("B: " + str(byas))
    print("XCoef: " + str(Xcoef))

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

def init_network(inputs, layers):
    network = []
    
    current_in = inputs
    for l in layers:
        layer = tf.Variable(0.02 * (-0.5 + np.random.rand(l, current_in + 1)), dtype='float64')
        current_in = l

        network.append(layer)

    return network


def apply_network(network, inputs):
    current_out = inputs
    for layer in network:
        current_out = sigmoid(tf.matmul(current_out, tf.transpose(layer)))
        return current_out

    return current_out


def create_bh(inputs, out):
    return tf.Variable(-0.5 + np.random.rand(out, inputs + 1), dtype='float64')

def apply_bh(network, inputs):
    return sigmoid(tf.matmul(inputs, tf.transpose(network)))

def sigmoid(tensor):
    return 1.0/(1.0 + tf.exp(-tensor))

# Set up the data
random.seed(1234)
points, out = generate_split_data()#generateChevronData()

num_bh = 2

one = tf.constant([1.0], dtype='float64')

inpt = tf.placeholder('float64', [2], name='inpt')
target = tf.placeholder('float64', name='target')

inpt_prime = tf.transpose(tf.expand_dims(tf.concat([one, inpt], axis=0), 1))

# Create boundary hunters
boundary_hunters = [create_bh(2, 1) for i in range(num_bh)]

# Create gating network
gating_network = init_network(2, [3, num_bh])

boundary_hunter_outs = [apply_bh(net, inpt_prime)[0] for net in boundary_hunters]
gate_out = apply_network(gating_network, inpt_prime)
norm_gate_out = tf.nn.softmax(gate_out)

dif = lambda x: tf.pow(tf.subtract(target, x), 2.0)
o = tf.convert_to_tensor(boundary_hunter_outs, dtype=tf.float64)
square_diff = tf.map_fn(dif, tf.convert_to_tensor(boundary_hunter_outs, dtype=tf.float64))

#errors = tf.transpose(tf.exp((-1.0/2.0) * square_diff))
#error = -tf.log(tf.reduce_sum(tf.multiply(gate_out, errors)))
errors = tf.multiply(gate_out, square_diff)
error = tf.reduce_sum(errors)

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(error)

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)

    #print(session.run(boundary_hunter_outs, feed_dict={inpt: points[0], target: out[0]}))
    #print(session.run(o, feed_dict={inpt: points[0], target: out[0]}))
    #print(session.run(square_diff, feed_dict={inpt: points[0], target: out[0]}))
    #print(session.run(error, feed_dict={inpt: points[0], target: out[0]}))
    
    #print(session.run(norm_gate_out, feed_dict={inpt: points[0], target: out[0]}))
    #print(session.run(error, feed_dict={inpt: points[0], target: out[0]}))
    #print()
    #for i in range(100):
    #    session.run(train_op, feed_dict={inpt: points[0], target: out[0]})
    #print(session.run(norm_gate_out, feed_dict={inpt: points[0], target: out[0]}))
    #print(session.run(error, feed_dict={inpt: points[0], target: out[0]}))

    err = 0
    for d in range(len(points)):
            err += session.run(error, feed_dict={inpt: points[d], target: out[d]})

    print(err)

    for e in range(1000):
        for d in range(len(points)):
            session.run(train_op, feed_dict={inpt: points[d], target: out[d]})

    err = 0
    for d in range(len(points)):
            err += session.run(error, feed_dict={inpt: points[d], target: out[d]})

    print(err)

    final_hunters = session.run(boundary_hunters)
    print(final_hunters)




# Plot information
# Plot points on graph
c1 = []
c2 = []

for i in range(0, len(points)):
    if out[i] == 0:
        c1.append(points[i])
    else:
        c2.append(points[i])

print("Type 0: ", len(c1))
print("Type 1: ", len(c2))
        
plotScatter(c1,'y')
plotScatter(c2, 'b')

for bh in final_hunters:
    net = bh[0]
    print(net)
    plot_weights(net)

plt.gca().set_aspect('equal')

plt.show()


