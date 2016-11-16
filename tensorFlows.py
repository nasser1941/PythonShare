import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

""""
graph = tf.get_default_graph()
input_value = tf.constant(1.0)
operations = graph.get_operations()
#print operations[0].node_def
sess = tf.Session()


weight = tf.Variable(0.8)
#for op in graph.get_operations(): print(op.name)

output_value = tf.mul(weight,input_value)
init = tf.initialize_all_variables()
sess.run(init)

print sess.run(output_value)

x = tf.constant(1.0, name='input')
w = tf.Variable(0.8, name='weight')
y = tf.mul(w, x, name='output')

summary_writer = tf.train.SummaryWriter('log_simple_graph', sess.graph)
"""
# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50

graph = tf.get_default_graph()
rng = np.random

train_X = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

train_Y = np.array([])
for xin in train_X:
    result = 2+2*xin + rng.randn()
    train_Y = np.append(train_Y, result)

n_samples = train_X.shape[0]

X = tf.placeholder("float")
Y = tf.placeholder("float")

sess = tf.Session()
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

pred = tf.add(tf.mul(X, W), b)

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        #Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), "W=", sess.run(W), "b=", sess.run(b)


    print "Optimization Finished!"
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print "Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n'

    #Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
