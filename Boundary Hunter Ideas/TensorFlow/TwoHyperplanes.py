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
            points.append([1, x/50.0,y/50.0])
            targets.append(0)
        else:
            points.append([1, x/50.0,y/50.0])
            targets.append(1)
        
    return np.array(points), np.array(targets)
    
def plotScatter(points, color):
    xs = [x[1] for x in points]
    ys = [y[2] for y in points]
    
    plt.scatter(xs, ys, c=color)


def plot_weights(weights, color):
    byas = -1 * weights[0]/weights[2]
    Xcoef = -1 * weights[1]/weights[2]

    plt.plot([-1.0, 1.0], [-1*Xcoef + byas, Xcoef + byas], '{}-'.format(color))

    print("\nLine {}".format(color))
    print("B: " + str(byas))
    print("XCoef: " + str(Xcoef))

    

def sigmoid(tensor, s=1):
    return 1.0/(1.0 + tf.exp(-(tf.multiply(tensor, s))))

def loss(predictions, targets, significance):
    # Compute the probobality of being right
    p_right = tf.multiply(tf.pow(predictions, tf.expand_dims(targets, 1)), tf.pow(1-predictions, tf.expand_dims(1-targets, 1)))

    # Compute the probobality of being wrong
    p_wrong = tf.multiply(tf.pow(predictions, tf.expand_dims(1-targets, 1)), tf.pow(1-predictions, tf.expand_dims(targets, 1)))

    return tf.multiply(significance, tf.subtract(tf.multiply(C, p_wrong), tf.multiply(B, p_right)))


#def loss(predictions, targets, significance):
    # Compute the probobality of being right
#    c = tf.multiply(tf.expand_dims(targets, 1), tf.log(predictions)) + tf.multiply(tf.expand_dims(1 - targets, 1), tf.log(1-predictions))

    # Compute the probobality of being wrong
#    return -(1.0/len(points)) * tf.multiply(c, significance)
    

# Set up the data
random.seed(1234)
points, out = generateChevronData()

C = tf.constant(2.5, 'float64', name='C')
B = tf.constant(1.0, 'float64', name='B')

weights = tf.Variable(np.array([0.0, 1.0,1.0, 0.0, -1.0, 1.0]), 'weights', dtype='float64')
#weights = tf.Variable(np.random.rand(6), 'weights', dtype='float64')
decision_boundary_normal = [weights[0],weights[1],weights[2]]
caring_boundary_normal = [weights[3],weights[4],weights[5]]

# Set up inputs/outputs to the network
inputs = tf.placeholder('float64', [None, 3], name='inputs')
targets = tf.placeholder('float64', [None], name='targets')

predictions = sigmoid(tf.matmul(inputs, tf.expand_dims(decision_boundary_normal, 1)))
significance = sigmoid(tf.matmul(inputs, tf.expand_dims(caring_boundary_normal, 1)))

raw_loss = loss(predictions, targets, 1-significance)
loss = tf.reduce_sum(raw_loss)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    print(session.run(loss, feed_dict={inputs:points, targets:out}))

    for i in range(100):
        session.run(train_op, feed_dict={inputs:points, targets:out})
        
    print(session.run(loss, feed_dict={inputs:points, targets:out}))
    #print(session.run(predictions, feed_dict={inputs:points, targets:out}))
    #print(session.run(significance, feed_dict={inputs:points, targets:out}))

    decision_weights = session.run(decision_boundary_normal)
    caring_weights = session.run(caring_boundary_normal)
    

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

plot_weights(decision_weights, 'r')
plot_weights(caring_weights, 'g')

plt.gca().set_aspect('equal')

plt.show()
