import SonarDataset
import tensorflow as tf
import numpy as np

def sigmoid(phi):
    return 1.0/(1.0 + tf.exp(-phi))

def train_boundary_hunter(points, out, iterations):
    in_size = len(data[0])
    out_size = 1
    num_centroids = 1

    inputs = tf.placeholder('float64', [in_size])
    targets = tf.placeholder('float64', [out_size])

    hidden_weights = tf.Variable(np.random.uniform(low=-0.5, high=0.5, size=(num_centroids, in_size+1)))
    gate_weights = tf.Variable(np.random.uniform(low=-0.5, high=0.5, size=(num_centroids, in_size+1)))
    byas = tf.Variable(np.random.uniform(low=-0.5, high=0.5, size=(num_centroids)))
    output_weights = tf.Variable(np.random.uniform(low=-0.5, high=0.5, size=(out_size, num_centroids + 1)))

    inputs_prime = tf.concat([[1.0], inputs], axis=0)

    # Peform Computation
    prob = sigmoid(tf.reduce_sum(tf.multiply(inputs_prime, hidden_weights), 1))

    g = sigmoid(tf.reduce_sum(tf.multiply(inputs_prime, gate_weights), 1))
    hidden_out = tf.add(byas, tf.multiply(g, tf.subtract(prob, byas)))

    output = hidden_out
    errors = tf.pow(tf.subtract(tf.expand_dims(targets, 1), output), 2.0)
    error = tf.reduce_sum(errors)

    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(error)
    clip_byas = tf.assign(byas, tf.clip_by_value(byas, 0, 1))

    model = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(model)
        
        for e in range(iterations):
            for d in range(len(points)):
                session.run(train_op, feed_dict={inputs: points[d], targets: [out[d]]})
                session.run(clip_byas)
                

            if e % 100 == 0:
                print(session.run(byas))
                err = 0
                for d in range(len(points)):
                    err += session.run(error, feed_dict={inputs: points[d], targets: [out[d]]})
                print(err)


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
    in_size = len(data[0])
    out_size = 1
    
    inputs = tf.placeholder('float64', [in_size])
    targets = tf.placeholder('float64', [out_size])
    hidden_weights = tf.placeholder('float64', [None])
    gate_weights = tf.placeholder('float64', [None])

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
            print(predictions, ": ", p)
            if not p == None:
                guessed += 1
            
            if p == out[d]:
                correct += 1
            elif p == None:
                unsure += 1

    return float(correct)/float(guessed), float(unsure)/float(len(data))

def split_tt(test_size, data, targets):
    idx = np.array(range(len(data)))
    np.random.shuffle(idx)
    
    train_data = []
    train_targets = []
    test_data = []
    test_targets = []

    for i in range(len(idx)):
        if i < test_size:
            test_data.append(data[idx[i]])
            test_targets.append(targets[idx[i]])
        else:
            train_data.append(data[idx[i]])
            train_targets.append(targets[idx[i]])

    return (train_data, train_targets), (test_data, test_targets)

data, targets = SonarDataset.load()
np.random.seed(432541)
train, test = split_tt(104, data, targets)

boundarys = []
gates = []
byass = []

total_bh_to_train = 50
iterations = 15000
for i in range(total_bh_to_train):
    print("Training Number: ", i)
    boundary, gate, byas = train_boundary_hunter(train[0], train[1], iterations)

    boundarys.append(boundary)
    gates.append(gate)

print("Classifying...")
train_accuracy, train_unsure = run_boundary_hunters(boundarys, gates, train[0], train[1])
test_accuracy, test_unsure = run_boundary_hunters(boundarys, gates, test[0], test[1])

print("Train Accuracy: ", train_accuracy, ", Unsure: ", train_unsure)
print("Test Accuracy: ", test_accuracy, ", Unsure: ", test_unsure)



