import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split

X = np.load('../X.npy')
Y = np.load('../Y (1).npy')

print('X:', X[-6:-1], "len:", len(X), "len2:", len(X[0]))
print('Y:', Y[-6:-1], "len:", len(Y)) #1 is resistant, -1 susceptible


num_features = len(X[0]) #however many non-redudndant k-mers there are
n = len(X)
num_classes = 2
learning_rate = 0.01
training_steps = 1000
batch_size = 256
display_step = 50
kmer_length = 31
# W = tf.Variable(tf.ones([num_features, num_classes]), name="weights")
# b = tf.Variable(tf.zeros([num_classes]), name="bias")
#
# def logistic_regression(x):
#     return tf.nn.softmax(tf.matmul(x, W) + b)
#
# def cross_entropy(y_pred, y_true):
#
#     retuRN

#ethionamide
#
# kmer_ids = []
# redundant_presences = []#given 2d array of each k-mer's presence
# profile_dict = {}
# for i in range(len(redundant_presence)):
#     if redundant_presence[i] not in profile_dict:
#         profile_dict[redundant_presence[i]] = {}
#     profile_dict[redundant_presence[i]].append(kmer_ids[i])
#
# presences = [profile for profile in profile_dict.keys()]
# #lookup requires search of kmer ids: not tractable!

#to handle possible sequencing errors
def stringent(presences):
    return

def conservative(presences):
    return

def vote(presences):
    return

def smooth(presences):
    return

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

def train(model):
    with tf.GradientTape as tape:
        tape.watch(x)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)

#input should be the matrix of each k-mer's presence in the genome
model = keras.Sequential()
model.add(keras.layers.Dense(2, use_bias=True, kernel_regularizer='l1'))
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.BinaryAccuracy()]
)

history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=training_steps,
    validation_data=(x_val, y_val)
)

results = model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)
