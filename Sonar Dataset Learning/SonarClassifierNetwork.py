import SonarDataset
import tensorflow as tf
import numpy as np

def sigmoid(phi):
    return 1.0/(1.0 + tf.exp(-phi))

def train_network(training_points, training_out, testing_points, testing_out, num_c):
    in_size = len(data[0])
    out_size = 1
    num_centroids = num_c
    
    inputs = tf.placeholder('float64', [in_size])
    targets = tf.placeholder('float64', [out_size])

    centroids = tf.Variable(np.random.uniform(low=-1.0, high=1.0, size=(num_centroids, in_size)))
    #betas = tf.Variable(np.repeat(1.0, num_centroids))
    hidden_weights = tf.Variable(np.random.uniform(low=-0.5, high=0.5, size=(num_centroids, in_size)))
    gate_weights = tf.Variable(np.random.uniform(low=-0.5, high=0.5, size=(num_centroids, in_size)))
    byas = tf.Variable(np.random.uniform(low=-0.5, high=0.5, size=(num_centroids)))
    output_weights = tf.Variable(np.random.uniform(low=-0.5, high=0.5, size=(out_size, num_centroids + 1)))

    input_by_plane = lambda x: tf.subtract(inputs, x)
    transformed_by_points = tf.map_fn(input_by_plane, centroids)

    # Peform Computation
    prob = tf.reduce_sum(tf.multiply(transformed_by_points, hidden_weights), 1)

    g = tf.reduce_sum(tf.multiply(transformed_by_points, gate_weights), 1)
    hidden_out = tf.add(byas, tf.multiply(g, tf.subtract(prob, byas)))#sigmoid(tf.multiply(g, prob))#tf.add(0.5 * (1 - g), tf.multiply(g, prob))
        #gated = tf.multiply(g, prob)
        #hidden_out = sigmoid(gated)
    hidden_out_prime = tf.concat([[1.0], hidden_out], 0)

    output = sigmoid(tf.matmul(tf.transpose(tf.expand_dims(hidden_out_prime, 1)), tf.transpose(output_weights)))
        
    errors = tf.pow(tf.subtract(tf.expand_dims(targets, 1), output), 2.0)
    error = tf.reduce_sum(errors)

    train_op = tf.train.GradientDescentOptimizer(0.0004).minimize(error)
    #clip_op_betas = tf.assign(betas, tf.clip_by_value(betas, 0, np.infty))

    model = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(model)

        
        
        for e in range(20000):
            for d in range(len(training_points)):
                session.run(train_op, feed_dict={inputs: training_points[d], targets: [training_out[d]]})

            if e % 10 == 0:
                err = 0
                for d in range(len(training_points)):
                    err += session.run(error, feed_dict={inputs: training_points[d], targets: [training_out[d]]})
                print(err)


        testing_incorrect = 0
        training_incorrect = 0

        for d in range(len(training_points)):
            o = session.run(output, feed_dict={inputs: training_points[d], targets: [training_out[d]]})
            if not int(round(o[0,0])) == training_out[d]:
               training_incorrect += 1

        for d in range(len(testing_points)):
            o = session.run(output, feed_dict={inputs: testing_points[d], targets: [testing_out[d]]})
            if not int(round(o[0,0])) == testing_out[d]:
               testing_incorrect += 1

        #centroids = session.run(centroids)
        #gates = session.run(gate_weights)
        #byas = session.run(byas)
        #boundarys = session.run(hidden_weights)

    return (float(training_incorrect)/float(len(training_points))), (float(testing_incorrect)/float(len(testing_points)))


def split_data(K, data, targets):
    idx = np.array(range(len(data)))
    np.random.shuffle(idx)

    size = int(float(len(data))/float(K))
    partitions = []
    
    for i in range(0, len(data), size):
        train_data = []
        train_targets = []
        test_data = []
        test_targets = []

        for j in range(len(idx)):
            if j > i and j < i + size:
                test_data.append(data[idx[j]])
                test_targets.append(targets[idx[j]])
            else:
                train_data.append(data[idx[j]])
                train_targets.append(targets[idx[j]])

        train = (train_data, train_targets)
        test = (test_data, test_targets)
        partitions.append((train, test))

    return partitions

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
#partitions = split_data(10, data, targets)
np.random.seed(432541)
train, test = split_tt(104, data, targets)

training_inaccuracy, testing_inaccuracy = train_network(train[0], train[1], test[0], test[1], 50)
print("Training: ", training_inaccuracy, ", Testing: ", testing_inaccuracy)
#training_accuracys = []
#testing_accuracys = []

##for p in [partitions[0]]:
##    train = p[0]
##    test = p[1]
##    
##    training_accuracy, testing_accuracy = train_network(train[0], train[1], test[0], test[1], 10)
##    print("Training: ", training_accuracy, ", Testing: ", testing_accuracy)
##
##    training_accuracys.append(training_accuracy)
##    testing_accuracys.append(testing_accuracy)
##
##training_accuracys = np.array(training_accuracys)
##testing_accuracys = np.array(testing_accuracys)
##
##print("Training Accuracy Mean", training_accuracys.mean(), ", STD: ", training_accuracys.std())
##print("Testing Accuracy Mean", testing_accuracys.mean(), ", STD: ", testing_accuracys.std()) 
