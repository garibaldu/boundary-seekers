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

def sigmoid(phi):
    return 1.0/(1.0 + tf.exp(-phi))

points, out = generateChevronData()#generate_clumps()#generate_split_data()#
in_size = 2
out_size = 1
num_centroids = 1
num_outputs = 1

inputs = tf.placeholder('float64', [in_size])
targets = tf.placeholder('float64', [out_size])

centroids = tf.Variable(np.random.uniform(low=-1.0, high=1.0, size=(num_centroids, in_size)))
#betas = tf.Variable(np.repeat(1.0, num_centroids))
hidden_weights = tf.Variable(np.random.uniform(low=-0.5, high=0.5, size=(num_centroids, in_size)))
gate_weights = tf.Variable(np.random.uniform(low=-0.5, high=0.5, size=(num_centroids, in_size)))
output_weights = tf.Variable(np.random.uniform(low=-0.5, high=0.5, size=(num_outputs, num_centroids + 1)))

input_by_plane = lambda x: tf.subtract(inputs, x)
transformed_by_points = tf.map_fn(input_by_plane, centroids)

# Peform Computation
prob = tf.reduce_sum(tf.multiply(transformed_by_points, hidden_weights), 1)

#square_diff = lambda c: tf.reduce_sum(tf.pow(tf.subtract(inputs, c), 2.0))
#g = tf.exp(-1.0 * tf.multiply(betas, tf.map_fn(square_diff, centroids)))
g = tf.reduce_sum(tf.multiply(transformed_by_points, gate_weights), 1)
hidden_out = sigmoid(tf.multiply(g, prob))#tf.add(0.5 * (1 - g), tf.multiply(g, prob))
#gated = tf.multiply(g, prob)
#hidden_out = sigmoid(gated)
hidden_out_prime = tf.concat([[1.0], hidden_out], 0)

output = sigmoid(tf.matmul(tf.transpose(tf.expand_dims(hidden_out_prime, 1)), tf.transpose(output_weights)))
errors = tf.pow(tf.subtract(tf.expand_dims(targets, 1), output), 2.0)
error = tf.reduce_sum(errors)

train_op = tf.train.GradientDescentOptimizer(0.006).minimize(error)
#clip_op_betas = tf.assign(betas, tf.clip_by_value(betas, 0, np.infty))

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)

    
    
    for e in range(3000):
        for d in range(len(points)):
            session.run(train_op, feed_dict={inputs: points[d], targets: [out[d]]})
            #session.run(clip_op_betas)

        if e % 10 == 0:
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
        if not int(round(o[0,0])) == out[d]:
            incorrect.append(points[d])

    centroids = session.run(centroids)
    gates = session.run(gate_weights)
    #betas = session.run(betas)
    boundarys = session.run(hidden_weights)


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

for centroid in centroids:
    plot_centroid(centroid)

for i in range(len(boundarys)):
    plot_weights(boundarys[i], gates[i], centroids[i], 'g')

#for plane in boundarys:
#    plot_weights(boundarys, 'g')

for point in incorrect:
    plot_incorrect(point)

#plot_weights(final_gate, 'g')

plt.gca().set_aspect('equal')
plt.xlim(xmin=-1.5, xmax=1.5)
plt.ylim(ymin=-1.5, ymax=1.5)

plt.show()
