import tensorflow as tf
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import random
import math

np.random.seed(1234)
random.seed(1234)

plt.switch_backend("TkAgg")

def plotScatter(points, color):
    xs = [x[0] for x in points]
    ys = [y[1] for y in points]
    
    plt.scatter(xs, ys, c=color)

def plot_weights(weights, gate, color):
    plot_line(weights, color)
    plot_line(gate, 'r')

    #print("B: " + str(byas))
    #print("XCoef: " + str(Xcoef))

def plot_line(weights, color):
    n = weights

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

def generateChevronDataWithNoise():
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
        elif x <= y:
            points.append([x/50.0,y/50.0])
            targets.append(float(np.random.randint(2)))
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

points, out = generate_clumps()#generateChevronDataWithNoise()#generateChevronData()#generate_split_data()#generate_rectangle_data()#generateChevronDataWithNoise()#
in_size = 2
out_size = 1
num_centroids = 1
num_outputs = 1

np.random.seed(21354)
random.seed(5324)

inputs = tf.placeholder('float64', [in_size])
targets = tf.placeholder('float64', [out_size])

hidden_weights = tf.Variable(np.random.uniform(low=-0.5, high=0.5, size=(num_centroids, in_size+1)))
gate_weights = tf.Variable(np.random.uniform(low=-0.5, high=0.5, size=(num_centroids, in_size+1)))
byas = tf.Variable(np.random.uniform(low=-0.5, high=0.5, size=(num_centroids)))
output_weights = tf.Variable(np.random.uniform(low=-0.5, high=0.5, size=(num_outputs, num_centroids + 1)))

inputs_prime = tf.concat([[1.0], inputs], axis=0)

# Peform Computation
prob = tf.reduce_sum(tf.multiply(inputs_prime, hidden_weights), 1)

g = sigmoid(tf.reduce_sum(tf.multiply(inputs_prime, gate_weights), 1))
#hidden_out = tf.add(byas, tf.multiply(g, tf.subtract(prob, byas)))
hidden_out = sigmoid(tf.add(g * prob, (1-g) * byas))

targets_prime = tf.expand_dims(targets, 1)

output = hidden_out
errors = -(targets_prime * tf.log(output) +  (1 -targets_prime) * tf.log(1 - output))#tf.pow(tf.subtract(tf.expand_dims(targets, 1), output), 2.0)
error = tf.reduce_sum(errors)

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(error)
#clip_byas = tf.assign(byas, tf.clip_by_value(byas, 0, 1))

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    
    for e in range(6000):
        for d in range(len(points)):
            session.run(train_op, feed_dict={inputs: points[d], targets: [out[d]]})
            #session.run(clip_byas)
            #session.run(clip_op_betas)

        if e % 10 == 0:
            print(session.run(byas))
            err = 0
            for d in range(len(points)):
                err += session.run(error, feed_dict={inputs: points[d], targets: [out[d]]})
                #print(session.run(prob, feed_dict={inputs: points[d], targets: [out[d]]}))
                #print(session.run(g, feed_dict={inputs: points[d], targets: [out[d]]}))
            print(err)
            #print(session.run(betas))

    incorrect = []
    for d in range(len(points)):
        o = session.run(output, feed_dict={inputs: points[d], targets: [out[d]]})
        if not int(round(o[0])) == out[d]:
            incorrect.append(points[d])

    gates = session.run(gate_weights)
    byas = session.run(byas)
    boundarys = session.run(hidden_weights)


print(byas)

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

for i in range(len(boundarys)):
    plot_weights(boundarys[i], gates[i], 'g')


for point in incorrect:
    plot_incorrect(point)

plt.gca().set_aspect('equal')
plt.xlim(xmin=-1.5, xmax=1.5)
plt.ylim(ymin=-1.5, ymax=1.5)

plt.show()
