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
weights = tf.Variable(np.random.rand(4), 'weights')
#weights = tf.Variable(np.array([-1.0, 1.0, 0.0, -0.0]), 'weights', dtype='float64')
radius = tf.Variable(0.5, 'radius', dtype='float64')
normal = [weights[0], weights[1]]
normal2 = [weights[2], weights[3]]

# Set up inputs/outputs to the network
inputs = tf.placeholder('float64', [None, 2], name='inputs')
targets = tf.placeholder('float64', [None], name='targets')

# Transform inputs
inputs_prime = tf.subtract(inputs, point)

# Compute the predictions
predictions = sigmoid(tf.matmul(inputs_prime, tf.expand_dims(normal, 1)))

# Compute the responsibility
responsiblity = 1 - (1/(1 + tf.exp(-(5.0)*(tf.reduce_sum(tf.pow(inputs_prime, 2), 1) - radius))))

# Compute the probobality of being right
p_right = tf.multiply(tf.pow(predictions, tf.expand_dims(targets, 1)), tf.pow(1-predictions, tf.expand_dims(1-targets, 1)))

# Compute the probobality of being wrong
p_wrong = tf.multiply(tf.pow(predictions, tf.expand_dims(1-targets, 1)), tf.pow(1-predictions, tf.expand_dims(targets, 1)))

profit = tf.reduce_sum(tf.multiply(tf.expand_dims(responsiblity, 1), tf.subtract(tf.multiply(C, p_wrong), tf.multiply(B, p_right))))

train_op = tf.train.GradientDescentOptimizer(0.001).minimize(profit)
train_op_weights = tf.train.GradientDescentOptimizer(0.001).minimize(profit, var_list=[weights])
train_op_radius = tf.train.GradientDescentOptimizer(0.001).minimize(profit, var_list=[radius])


clip_radius = tf.assign(radius, tf.clip_by_value(radius, 0.3, 0.8))

# Set up the data
random.seed(1234)
points, out = generateChevronData()
model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    print(session.run(profit, feed_dict={inputs:points, targets:out}))
    print(session.run(weights))
    print(session.run(radius))
    print()
    #for i in range(10000):
    #    session.run(train_op, feed_dict={inputs:points, targets:out})
    #    session.run(clip_radius)
    for i in range(10):
        for j in range(100):
            session.run(train_op_weights, feed_dict={inputs:points, targets:out})

        for j in range(100):
            session.run(train_op_radius, feed_dict={inputs:points, targets:out})
            session.run(clip_radius)


    final_error = session.run(profit, feed_dict={inputs:points, targets:out})
    weights_value = session.run(weights)
    radius_value = session.run(radius)

print("Error: ", final_error)
print("Weights: ", weights_value)
print("Radius: ", radius_value)

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

plt.scatter(weights_value[2], weights_value[3], c='g')

n = np.array([weights_value[0] * weights_value[2] + weights_value[1] * weights_value[3], 
              -weights_value[0], 
              -weights_value[1]])

byas = -1 * n[0]/n[2]
Xcoef = -1 * n[1]/n[2]

x = np.linspace(-1.5, 1.5, 500)
y = np.linspace(-1.5, 1.5, 500)
X, Y = np.meshgrid(x,y)
F = ((X - weights_value[2]))**2 + ((Y - weights_value[3]))**2 - radius_value**2
plt.contour(X,Y,F,[0])

print()
print(n)
print("\nLine")
print("B: " + str(byas))
print("XCoef: " + str(Xcoef))

plt.plot([-1.0, 1.0], [-1*Xcoef + byas, Xcoef + byas], 'k-')
plt.gca().set_aspect('equal')

plt.show()

    
