import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
# from spektral.layers import GraphConv
# from spektral.layers.ops import sp_matrix_to_sp_tensor
# from spektral.utils import batch_iterator
from dataset_split import *
from graphplt import *

# l2_regularization = 4e-4
# total_classes = 2#Health and Patient
lr = 0.001
training_iters = 100 # dsx 100000
batch_size = 128
n_input = 49
n_steps = 64
n_hidden_units = 128
n_classes = 2
# class GraphNet(Model):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.conv1 = GraphConv(32, activation='elu', kernel_regularizer=l2(l2_regularization))
#         self.conv2 = GraphConv(32, activation='elu', kernel_regularizer=l2(l2_regularization))
#         self.flatten = Flatten()
#         self.fc1 = Dense(512, activation='relu')
#         self.fc2 = Dense(total_classes, activation='softmax')
#
#     def call(self, inputs):
#         x, fltr = inputs
#         x = self.conv1([x, fltr])
#         x = self.conv2([x, fltr])
#         output = self.flatten(x)
#         output = self.fc1(output)
#         output = self.fc2(output)
#         return output



def compute_accuracy(v_x, v_y):
    global prediction
    y_pre = sess.run(prediction, feed_dict={x: v_x, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={x: v_x, y: v_y, keep_prob: 1})
    return result


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # strides=[1,x_movement,y_movement,1]
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv_pool_layer(X, img_len, img_hi, out_seq):
    W = weight_variable([img_len, img_len, img_hi, out_seq])
    b = bias_variable([out_seq])
    h_conv = tf.nn.relu(conv2d(X, W) + b)
    return max_pool_2x2(h_conv)


def lstm(X):
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X, initial_state=_init_state, time_major=False)
    W = weight_variable([n_hidden_units, n_classes])
    b = bias_variable([n_classes])
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], W) + b
    return results



def train_model():
    # load mnist data
    # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 16])
    print(x)
    print(y)
    keep_prob = tf.placeholder(tf.float32)
    print(keep_prob)
    # reshape(data you want to reshape, [-1, reshape_height, reshape_weight, imagine layers]) image layers=1 when the imagine is in white and black, =3 when the imagine is RGB
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    print(x_image)

    # ********************** conv1 *********************************
    # transfer a 5*5*1 imagine into 32 sequence
    # W_conv1 = weight_variable([5,5,1,32])
    # b_conv1 = bias_variable([32])
    # input a imagine and make a 5*5*1 to 32 with stride=1*1, and activate with relu
    # h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28*28*32
    # h_pool1 = max_pool_2x2(h_conv1) # output size 14*14*32
    h_pool1 = conv_pool_layer(x_image, 5, 1, 32)
    print(h_pool1)

    # ********************** conv2 *********************************
    # transfer a 5*5*32 imagine into 64 sequence
    # W_conv2 = weight_variable([5,5,32,64])
    # b_conv2 = bias_variable([64])
    # input a imagine and make a 5*5*32 to 64 with stride=1*1, and activate with relu
    # h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14*14*64
    # h_pool2 = max_pool_2x2(h_conv2) # output size 7*7*64
    h_pool2 = conv_pool_layer(h_pool1, 5, 32, 64)
    print(h_pool2)
    # reshape data
    X_in = tf.reshape(h_pool2, [-1, 49, 64])
    print(X_in)
    X_in = tf.transpose(X_in, [0, 2, 1])
    print(X_in)
    # put into a lstm layer
    prediction = lstm(X_in)
    print(prediction)
    # ********************* func1 layer *********************************
    # W_fc1 = weight_variable([7*7*64, 1024])
    # b_fc1 = bias_variable([1024])
    # reshape the image from 7,7,64 into a flat (7*7*64)
    # h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
    # h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

    # ********************* func2 layer *********************************
    # W_fc2 = weight_variable([1024, 10])
    # b_fc2 = bias_variable([10])
    # prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # calculate the loss
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    # use Gradientdescentoptimizer
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # init session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(training_iters):
        # batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x, batch_y ,val_data, val_label, test_data, test_label, graph_created = load_data()

        # y(128, 10)
        # x(128, 784) (16, 784)

        print("batch_x", batch_x.shape)
        print("batch_y", batch_y.shape)
        sess.run(train_step, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
        if i % 50 == 0:
            print(sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, }))
    # train_loss = []
    # train_acc = []
    # val_loss = []
    # val_acc = []
    # test_loss = []
    # test_acc = []
    # learning_rate = 1e-3  # Learning rate for Adam
    # batch_size = 32       # Batch size
    # epochs = 1000      # Number of training epochs dsx 1000
    # train_data, train_label, val_data, val_label, test_data, test_label, graph_created = load_data()
    # train_data, val_data, test_data = train_data[..., None], val_data[..., None], test_data[..., None]
    # # Create filter for GCN and convert to sparse tensor
    # fltr = sp_matrix_to_sp_tensor(GraphConv.preprocess(graph_created))
    # model = GraphNet()
    # optimizer = Adam(lr=learning_rate)
    # loss_fn = SparseCategoricalCrossentropy()
    # accuracy_fn = SparseCategoricalAccuracy()
    # # Training step
    # @tf.function
    # def train(x, y):
    #     with tf.GradientTape() as tape:
    #         predictions = model([x, fltr], training=True)
    #         loss = loss_fn(y, predictions)
    #         loss += sum(model.losses)
    #     acc = accuracy_fn(y, predictions)
    #     gradients = tape.gradient(loss, model.trainable_variables)
    #     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    #     model.summary()
    #     return loss, acc
    #
    #
    # # Evaluation step
    # @tf.function
    # def evaluate(x, y):
    #     predictions = model([x, fltr], training=False)
    #     loss = loss_fn(y, predictions)
    #     loss += sum(model.losses)
    #     acc = accuracy_fn(y, predictions)
    #     return loss, acc
    #
    #
    # curent_batch = 0
    # batches_in_epoch = int(np.ceil(train_data.shape[0] / batch_size))
    # batches_tr = batch_iterator([train_data, train_label], batch_size=batch_size, epochs=epochs)
    # results_train = []
    # for batch in batches_tr:
    #     curent_batch += 1
    #     loss_train, acc_train = train(*batch)
    #     results_train.append((loss_train, acc_train))
    #     if curent_batch == batches_in_epoch:
    #         batches_val = batch_iterator([val_data, val_label], batch_size=batch_size)
    #         results_val = [evaluate(*batch) for batch in batches_val]
    #         results_val = np.array(results_val)
    #         loss_val, acc_val = results_val.mean(0)
    #         batches_te = batch_iterator([test_data, test_label], batch_size=batch_size)
    #         results_test = [evaluate(*batch) for batch in batches_te]
    #         results_test = np.array(results_test)
    #         results_train = np.array(results_train)
    #         print('Train loss: {:.4f}, acc: {:.4f} | '
    #               'Valid loss: {:.4f}, acc: {:.4f} | '
    #               'Test loss: {:.4f}, acc: {:.4f}'
    #               .format(*results_train.mean(0),
    #                       *results_val.mean(0),
    #                       *results_test.mean(0)))
    #         train_loss.append(loss_train)
    #         train_acc.append(acc_train)
    #         val_loss.append(loss_val)
    #         val_acc.append(acc_val)
    #         loss_test, acc_test = results_test.mean(0)
    #         test_loss.append(loss_test)
    #         test_acc.append(acc_test)
    #         # Reset epoch
    #         results_train = []
    #         curent_batch = 0
    # return train_loss, train_acc, val_loss, val_acc, test_loss, test_acc


if __name__ == "__main__":
    train_loss, train_acc, val_loss, val_acc, test_loss, test_acc = train_model()
    # graphplt(train_loss, "Train_loss", "No. of epochs", "Loss During Training")
    # graphplt(train_acc, "Train_Accuracy", "No. of epochs", "Accuracy During Training")
    # graphplt(val_loss, "Validation_loss", "No. of epochs", "Loss During Validation")
    # graphplt(val_acc, "Validation_Accuracy", "No. of epochs", "Accuracy During Validation")
    # graphplt(test_loss, "Testing_loss", "No. of epochs", "Loss During Testing")
    # graphplt(test_acc, "Testing_Accuracy", "No. of epochs", "Accuracy During Testing")