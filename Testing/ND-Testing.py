import numpy as np
import tensorflow as tf

def __perms(n):
    if not n:
        return

    p = []

    for i in range(0, 2**n):
        s = bin(i)[2:]
        s = "0" * (n-len(s)) + s

        s_prime = np.array(list(map(lambda x: int(x), list(s))))
        p.append(s_prime)

    return p

def care(normal, bias, example):
    z = np.dot(normal, example) + bias
    return 1.0/(1.0 + np.exp(-z))

def deci(normal, bias, example):
    z = np.dot(normal, example) + bias
    return 1.0/(1.0 + np.exp(-z))

def sigmoid(phi):
    return 1.0/(1.0 + tf.exp(-phi))

def compute_penalty(weights, size):
    mask = np.concatenate((np.array([0], dtype=np.float32), np.ones(size, dtype=np.float32)))
    return tf.reduce_sum(tf.abs(tf.multiply(mask, weights)))

def train_boundary_hunter(points, out, iterations):
    in_size = len(points[0])
    out_size = 1

    inputs = tf.placeholder('float32', [in_size])
    targets = tf.placeholder('float32', [out_size])

    hidden_weights = tf.Variable(np.random.uniform(low=-0.5, high=0.5, size=(1, in_size+1)), dtype='float32')
    gate_weights = tf.Variable(np.random.uniform(low=-0.5, high=0.5, size=(1, in_size+1)), dtype='float32')
    byas = tf.Variable(np.random.uniform(low=-0.5, high=0.5, size=(1)), dtype='float32')
    #output_weights = tf.Variable(np.random.uniform(low=-0.5, high=0.5, size=(out_size, num_centroids + 1)), dtype='float32')

    inputs_prime = tf.concat([[1.0], inputs], axis=0)

    # Peform Computation
    # Peform Computation
    prob = tf.reduce_sum(tf.multiply(inputs_prime, hidden_weights), 1)

    g = sigmoid(tf.reduce_sum(tf.multiply(inputs_prime, gate_weights), 1))
    #hidden_out = tf.add(byas, tf.multiply(g, tf.subtract(prob, byas)))
    hidden_out = sigmoid(tf.add(g * prob, (1-g) * byas))

    reward = tf.log(compute_penalty(hidden_weights, in_size) + compute_penalty(gate_weights, in_size))

    targets_prime = tf.expand_dims(targets, 1)
    output = hidden_out
    errors = -(targets_prime * tf.log(output) +  (1 -targets_prime) * tf.log(1 - output))#tf.pow(tf.subtract(tf.expand_dims(targets, 1), output), 2.0)
    error = tf.reduce_sum(errors)
    minimize = error - 0.02 * reward

    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(minimize)
    #clip_byas = tf.assign(byas, tf.clip_by_value(byas, 0, 1))

    model = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(model)
        
        for e in range(iterations):
            for d in range(len(points)):
                session.run(train_op, feed_dict={inputs: points[d], targets: [out[d]]})
                #session.run(clip_byas)
                

            if e % 10 == 0:
                print(session.run(byas))
                err = 0
                for d in range(len(points)):
                   err += session.run(error, feed_dict={inputs: points[d], targets: [out[d]]})
                print(err)
                print(session.run(reward))
                print()


        gates = session.run(gate_weights)[0]
        byas = session.run(byas)[0]
        boundarys = session.run(hidden_weights)[0]

    return (boundarys, gates, byas)

def get_final_class(predictions):
    tally_0 = 0
    tally_1 = 0

    for p in predictions:
        if (not p == None) and p >= 0.5:
            tally_1 += 1
        elif (not p == None) and p < 0.5:
            tally_0 += 1

    if tally_0 == 0 and tally_1 == 0:
        return None
    
    return 0 if tally_0 > tally_1 else 1

def run_boundary_hunters(boundarys, gates, points, out):
    in_size = len(points[0])
    out_size = 1
    
    inputs = tf.placeholder('float32', [in_size])
    targets = tf.placeholder('float32', [out_size])
    hidden_weights = tf.placeholder('float32', [None])
    gate_weights = tf.placeholder('float32', [None])

    inputs_prime = tf.concat([[1.0], inputs], axis=0)

    g = sigmoid(tf.reduce_sum(tf.multiply(inputs_prime, gate_weights)))
    prob = sigmoid(tf.reduce_sum(tf.multiply(inputs_prime, hidden_weights)))

    model = tf.global_variables_initializer()

    unsure = 0
    guessed = 0
    correct = 0
    with tf.Session() as session:
        session.run(model)

        for d in range(len(points)):
            predictions = []
            for b in range(len(boundarys)):
                prediction = None
                care = session.run(g, feed_dict={inputs: points[d], hidden_weights: boundarys[b], gate_weights: gates[b]})

                if care > 0.5:
                    prediction = session.run(prob, feed_dict={inputs: points[d], hidden_weights: boundarys[b], gate_weights: gates[b]})
                predictions.append(prediction)

            p = get_final_class(predictions)
            #print(predictions, ": ", p)
            if not p == None:
                guessed += 1
            
            if p == out[d]:
                correct += 1
            elif p == None:
                unsure += 1

    return float(correct)/float(guessed), float(unsure)/float(len(points))

N = 7
# Generate All Points On Hypercube
examples = __perms(N)
targets = []

# Generate Boundary Hunter
bias = np.random.uniform(0, 1, 1)
decision = np.random.uniform(-1, 1, N)
decision_b = np.random.uniform(-1, 1, 1)
caring = np.random.uniform(-1, 1, N)
caring_b = np.random.uniform(-1, 1, 1)

uncertian = 0
class1 = 0
class0 = 0

for example in examples:
    clas = None
    c = care(caring, caring_b, example)

    if c < 0.5:
        uncertian += 1
        r = np.random.rand(1)
        if r > bias:
            clas = 1
        else:
            clas = 0
    else:
        d = deci(decision, decision_b, example)
        if d >= 0.5:
            clas = 1
            class1 += 1
        else:
            clas=0
            class0 += 1
    targets.append(clas)

if class0 == 0 or class1 == 0:
    print("Class 0: ", class0)
    print("Class 1: ", class1)
    print("Err")
    raise "GSFE"


bh = train_boundary_hunter(examples, targets, 20000)

print("Uncertian: ", uncertian)
print("Class 0: ", class0)
print("Class 1: ", class1)

print("Bias: ", bias)
print("{}, {}".format(decision_b, decision))
print("{}, {}".format(caring_b, caring))
print(run_boundary_hunters([np.concatenate((decision_b, decision))], [np.concatenate((caring_b, caring))], examples, targets))

print()
print(bh)
print(run_boundary_hunters([bh[0]], [bh[1]], examples, targets))

