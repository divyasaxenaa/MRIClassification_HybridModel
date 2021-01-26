import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPool2D, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import InverseTimeDecay
from tensorflow.keras.regularizers import l2
from spektral.layers import GraphConv
from spektral.layers.ops import sp_matrix_to_sp_tensor
from spektral.utils import batch_iterator
from dataset_split import *
from graphplt import *

l2_regularization = 4e-4
total_classes = 2#Health and Patient

class GraphCNNNet(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = GraphConv(128, activation='elu', kernel_regularizer=l2(l2_regularization))
        self.conv2 = GraphConv(64, activation='elu', kernel_regularizer=l2(l2_regularization))
        self.conv_1 =  Conv1D(32, kernel_size=(2), kernel_initializer = 'he_uniform')
        self.dropout = Dropout(0.5)
        self.flatten = Flatten()
        self.fc1 = Dense(512, activation='relu')
        self.fc2 = Dense(total_classes, activation='softmax')


    def call(self, inputs):
        x, fltr = inputs
        x = self.conv1([x, fltr])
        x = self.conv2([x, fltr])
        x = self.conv_1(x)
        x = self.dropout(x)
        output = self.flatten(x)
        output = self.fc1(output)
        output = self.fc2(output)
        return output

class GraphNet(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = GraphConv(64, activation='elu', kernel_regularizer=l2(l2_regularization))
        self.conv2 = GraphConv(32, activation='elu', kernel_regularizer=l2(l2_regularization))
        self.flatten = Flatten()
        self.fc1 = Dense(512, activation='relu')
        self.fc2 = Dense(total_classes, activation='softmax')


    def call(self, inputs):
        x, fltr = inputs
        x = self.conv1([x, fltr])
        x = self.conv2([x, fltr])
        output = self.flatten(x)
        output = self.fc1(output)
        output = self.fc2(output)
        return output


class CNNNet(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = Conv1D(64,kernel_size=(2), activation='elu', kernel_regularizer=l2(l2_regularization))
        self.conv2 = Conv1D(32, kernel_size=(2),activation='elu', kernel_regularizer=l2(l2_regularization))
        self.flatten = Flatten()
        self.fc1 = Dense(512, activation='relu')
        self.fc2 = Dense(total_classes, activation='softmax')


    def call(self, inputs):
        x, fltr = inputs
        x = self.conv1(x)
        x = self.conv2(x)
        output = self.flatten(x)
        output = self.fc1(output)
        output = self.fc2(output)
        return output

class GraphRNNNet(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = GraphConv(128, activation='elu', kernel_regularizer=l2(l2_regularization))
        self.conv2 = GraphConv(64, activation='elu', kernel_regularizer=l2(l2_regularization))
        self.conv_1 =  LSTM(32, kernel_initializer = 'he_uniform')
        self.flatten = Flatten()
        self.fc1 = Dense(512, activation='relu')
        self.fc2 = Dense(total_classes, activation='softmax')


    def call(self, inputs):
        x, fltr = inputs
        x = self.conv1([x, fltr])
        x = self.conv2([x, fltr])
        x = self.conv_1(x)
        output = self.flatten(x)
        output = self.fc1(output)
        output = self.fc2(output)
        return output
#divya
#
# class RNNNet(Model):
#     def __init__(self, **kwargs):
#         print(**kwargs)
#         super().__init__(**kwargs)
#
#         self.conv1 = LSTM(128, kernel_initializer = 'he_uniform')
#         self.conv2 = LSTM(64, kernel_initializer = 'he_uniform')
#         self.conv_1 =  LSTM(32, kernel_initializer = 'he_uniform')
#         self.flatten = Flatten()
#         self.fc1 = Dense(512, activation='relu')
#         self.fc2 = Dense(total_classes, activation='softmax')
#
#
#     def call(self, inputs):
#         print("1")
#         x, fltr = inputs
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv_1(x)
#         output = self.flatten(x)
#         output = self.fc1(output)
#         output = self.fc2(output)
#         return output


def train_model(model_name):
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    test_loss = []
    test_acc = []
    learning_rate = 1e-3  # Learning rate for Adam
    batch_size = 32       # Batch size
    epochs = 1      # Number of training epochs dsx 1000
    train_data, train_label, val_data, val_label, test_data, test_label, graph_created = load_data()
    train_data, val_data, test_data = train_data[..., None], val_data[..., None], test_data[..., None]

    # Create filter for GCN and convert to sparse tensor
    fltr = sp_matrix_to_sp_tensor(GraphConv.preprocess(graph_created))
    if model_name == "GraphNet":
        model = GraphNet()
    elif model_name == "GraphCNNNet":
        model = GraphCNNNet()
    elif model_name == "CNNNet":
        model = CNNNet()
    elif model_name == "GraphRNNNet":
        model = GraphRNNNet()
#divya
    # elif model_name == "RNNNet":
    #     model = GraphRNNNet("RNNNet")
    
    STEPS_PER_EPOCH = 1
    lr_schedule = InverseTimeDecay(0.001, 
        decay_steps=STEPS_PER_EPOCH*10, 
        decay_rate=1, 
        staircase=False)
    optimizer = Adam(lr_schedule)
    loss_fn = SparseCategoricalCrossentropy()
    accuracy_fn = SparseCategoricalAccuracy()
    # Training step
    @tf.function
    def train(x, y):
        # import pdb; pdb.set_trace()
        # print(model.summary())
        with tf.GradientTape() as tape:
            predictions = model([x, fltr], training=True)
            loss = loss_fn(y, predictions)
            loss += sum(model.losses)
        acc = accuracy_fn(y, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        model.summary()
        return loss, acc


    # Evaluation step
    @tf.function
    def evaluate(x, y):
        predictions = model([x, fltr], training=False)
        loss = loss_fn(y, predictions)
        loss += sum(model.losses)
        acc = accuracy_fn(y, predictions)
        return loss, acc


    curent_batch = 0
    batches_in_epoch = int(np.ceil(train_data.shape[0] / batch_size))
    batches_tr = batch_iterator([train_data, train_label], batch_size=batch_size, epochs=epochs)
    results_train = []
    for batch in batches_tr:
        curent_batch += 1
        loss_train, acc_train = train(*batch)
        results_train.append((loss_train, acc_train))
        if curent_batch == batches_in_epoch:
            batches_val = batch_iterator([val_data, val_label], batch_size=batch_size)
            results_val = [evaluate(*batch) for batch in batches_val]
            results_val = np.array(results_val)
            loss_val, acc_val = results_val.mean(0)
            batches_te = batch_iterator([test_data, test_label], batch_size=batch_size)
            results_test = [evaluate(*batch) for batch in batches_te]
            results_test = np.array(results_test)
            results_train = np.array(results_train)
            print('Train loss: {:.4f}, acc: {:.4f} | '
                  'Valid loss: {:.4f}, acc: {:.4f} | '
                  'Test loss: {:.4f}, acc: {:.4f}'
                  .format(*results_train.mean(0),
                          *results_val.mean(0),
                          *results_test.mean(0)))
            train_loss.append(loss_train)
            train_acc.append(acc_train)
            val_loss.append(loss_val)
            val_acc.append(acc_val)
            loss_test, acc_test = results_test.mean(0)
            test_loss.append(loss_test)
            test_acc.append(acc_test)
            # Reset epoch
            results_train = []
            curent_batch = 0
    return train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, model_name


if __name__ == "__main__":
    train_loss1, train_acc1, val_loss1, val_acc1, test_loss1, test_acc1, model_name1 = train_model("GraphNet")
    train_loss2, train_acc2, val_loss2, val_acc2, test_loss2, test_acc2, model_name2 = train_model("GraphCNNNet")
    train_loss3, train_acc3, val_loss3, val_acc3, test_loss3, test_acc3, model_name3 = train_model("CNNNet")
    train_loss4, train_acc4, val_loss4, val_acc4, test_loss4, test_acc4, model_name4 = train_model("GraphRNNNet")
    #divya train_loss5, train_acc5, val_loss5, val_acc5, test_loss5, test_acc5, model_name5 = train_model("RNNNet")
    graphplt(train_loss1,train_loss2,train_loss3,train_loss4,train_loss5, "Train_loss ", "No. of epochs", "Loss During Training")
    graphplt(train_acc1,train_acc2,train_acc3,train_acc4,train_acc5,"Train_Accuracy ", "No. of epochs", "Accuracy During Training")
    graphplt(val_loss1,val_loss2,val_loss3,val_loss4,val_loss5, "Validation_loss ", "No. of epochs", "Loss During Validation")
    graphplt(val_acc1,val_acc2,val_acc3,val_acc4 ,val_acc5,"Validation_Accuracy ", "No. of epochs", "Accuracy During Validation")
    graphplt(test_loss1,test_loss2,test_loss3,test_loss4, test_loss5,"Testing_loss  ", "No. of epochs", "Loss During Testing")
    graphplt(test_acc1,test_acc2,test_acc3,test_acc4,test_acc5, "Testing_Accuracy  ", "No. of epochs", "Accuracy During Testing")
