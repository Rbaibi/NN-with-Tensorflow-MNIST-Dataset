#Import modules
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)


x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')
classes = 10
epochs = 15
batch_size = 100

hl1_nodes = 500
hl2_nodes = 400
hl3_nodes = 500
hl4_nodes = 400
hl5_nodes = 200


#Define Neural network architecture
def create_neural_network(data):
    '''
    Build Neural network architecture
    '''
    HL1 = {'W':tf.Variable(tf.random_normal([784, hl1_nodes])),
                      'B':tf.Variable(tf.random_normal([hl1_nodes]))}

    HL2 = {'W':tf.Variable(tf.random_normal([hl1_nodes, hl2_nodes])),
                      'B':tf.Variable(tf.random_normal([hl2_nodes]))}

    HL3 = {'W':tf.Variable(tf.random_normal([hl2_nodes, hl3_nodes])),
                      'B':tf.Variable(tf.random_normal([hl3_nodes]))}

    HL4 = {'W':tf.Variable(tf.random_normal([hl3_nodes, hl4_nodes])),
                      'B':tf.Variable(tf.random_normal([hl4_nodes]))}

    HL5 = {'W':tf.Variable(tf.random_normal([hl4_nodes, hl5_nodes])),
                      'B':tf.Variable(tf.random_normal([hl5_nodes]))}
    

    output_layer = {'W':tf.Variable(tf.random_normal([hl5_nodes, classes])),
                    'B':tf.Variable(tf.random_normal([classes])),}


    l1 = tf.add(tf.matmul(data,HL1['W']), HL1['B'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,HL2['W']), HL2['B'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,HL3['W']), HL3['B'])
    l3 = tf.nn.relu(l3)
    
    l4 = tf.add(tf.matmul(l3,HL4['W']), HL4['B'])
    l4 = tf.nn.relu(l4)

    l5 = tf.add(tf.matmul(l4,HL5['W']), HL5['B'])
    l5 = tf.nn.relu(l5)

    network = tf.matmul(l5,output_layer['W']) + output_layer['B']

    return network

#Train the neural network
def train_neural_network(x):
    '''
    Train the neural network
    '''
    prediction = create_neural_network(x)

    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    saver = tf.train.Saver()
    
    hm_epochs = epochs
    with tf.Session() as sess:
    
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        # Save the variables to disk.
        save_path = saver.save(sess, "/tmp/model.ckpt")
        print("Model saved in path: %s" % save_path)

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)
