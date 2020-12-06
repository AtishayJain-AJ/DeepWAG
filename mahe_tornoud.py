# import tensorflow as tf
# from tensorflow import keras
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
import pandas as pd

X = np.load('../ethionamide/X.npy')
Y = np.load('../ethionamide/Y.npy')

# print('X:', X[-6:-1], "len:", len(X), "len2:", len(X[0]))
# print('Y:', Y[-6:-1], "len:", len(Y)) #1 is resistant, -1 susceptible


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
#
# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#
# def train(model):
#     with tf.GradientTape as tape:
#         tape.watch(x)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5)
#
# #input should be the matrix of each k-mer's presence in the genome
# model = keras.Sequential()
# model.add(keras.layers.Dense(2, use_bias=True, kernel_regularizer='l1'))
# model.compile(
#     optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
#     loss=keras.losses.BinaryCrossentropy(),
#     metrics=[keras.metrics.BinaryAccuracy()]
# )
#
# history = model.fit(
#     x_train,
#     y_train,
#     batch_size=batch_size,
#     epochs=training_steps,
#     validation_data=(x_val, y_val)
# )


#subsample dataset 100 times, including 50% of total pos/neg strains x
#for each fit a Lasso logistic model with 200 candidate values x
#for each coefficient, plot the number of subsamples in which the coefficient is selected as lasso increases
#
#finally fit an unpenalized logistic regression model including only k-mers with
#selection probability above a threshold at some point
d = len(X[0])
print(d)
counts = np.zeros((200, d))
n_subsamples = 10
for subs in range(n_subsamples):
    print('on subsample', subs)
    pos_idc = np.argwhere(y_train > 0)
    # print(pos_idc.shape)
    neg_idc = np.argwhere(y_train < 0)
    pos_sub_idc = np.random.choice(pos_idc[:][0], len(pos_idc) // 2)
    neg_sub_idc = np.random.choice(neg_idc[:][0], len(neg_idc) // 2)
    subsample_indices = np.concatenate((np.array(pos_sub_idc), np.array(neg_sub_idc)))
    x_subsample = x_train[subsample_indices]
    y_subsample = y_train[subsample_indices]

    for i in range(100):
        j = 200 - i
        c = j / 200
        logreg = LogisticRegression(penalty='l1', C=c, solver='liblinear')
        logreg.fit(x_subsample, y_subsample)
        score = logreg.score(x_test, y_test)
        # print('coefficient:', c, 'score:', score)
        log_odds = np.abs(logreg.coef_[0])

        selections = log_odds > 0.001

        counts[i] += selections
        # df = pd.DataFrame(counts[i],
        #      columns=['coef'])\
        #     .sort_values(by='coef', ascending=False)
        # print(df)

counts = counts / n_subsamples
lasso_threshold = 20
threshold = 0.2

max_score = 0
best_model = None

for lasso_threshold in range(0, 200):
    for j in range(1, 21):
        threshold = j / 20
        # df = pd.DataFrame(counts[lasso_threshold],
        #      columns=['proportion'])\
        #     .sort_values(by='proportion', ascending=False)
        # print(df)

        pruned_features_idc = np.argwhere(counts[lasso_threshold] > threshold)
        if len(pruned_features_idc) == 0:
            continue
        # print(pruned_features_idc)
        x_train_pruned = x_train[:,pruned_features_idc][:,:,0]
        x_val_pruned = x_val[:,pruned_features_idc][:,:,0]
        # print(x_train_pruned.shape)
        model = LogisticRegression(penalty='none')
        model.fit(x_train_pruned, y_train)
        score = model.score(x_val_pruned, y_val)
        # print("test acc:", score)
        if score > max_score:
            print('new record at lasso:', lasso_threshold, 'threshold:', threshold, 'with score:', score)
            x_test_pruned = x_test[:, pruned_features_idc][:,:,0]
            max_score = score
            best_model = model

#.predict
test_proba = best_model.predict_proba(x_test_pruned)
test_y = best_model.predict(x_test_pruned)
test_labels_tl = y_test
test_auroc = roc_auc_score(test_labels_tl, test_proba[:, 1])
precision, recall, threshold = precision_recall_curve(test_labels_tl, test_proba[:, 1])
test_auprc = auc(recall, precision)
test_f1 = f1_score(test_labels_tl, test_y)

print("AUROC:", test_auroc, "AUPRC:", test_auprc, "F1:", test_f1)
# print('test performance:', best_model.score(x_test_pruned, y_test))
