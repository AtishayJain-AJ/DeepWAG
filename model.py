from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras import Input
from keras.models import Model
from keras.layers import Conv1D, Dense, Dropout, GlobalAveragePooling1D, MaxPooling1D, Flatten, Lambda
from keras.regularizers import l2
from keras import backend as K
from keras.losses import binary_crossentropy
import tensorflow as tf
import numpy as np
import random

# need to make pairs
X = np.load('X.npy') # 571 seq x 1503 kmers
Y = np.load('Y.npy') #571 seq x 1


num_features = len(X[0]) #however many non-redudndant k-mers there are
n = len(X)
print(n)
train_proportion = .5


## shuffle 
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

X = X[indices]
Y = Y[indices]

to_split = round(n * train_proportion)


## training sets 
train_x, train_y = X[:to_split], Y[:to_split]
train_susceptibles = np.where(train_y == -1)[0]
train_sus_samples = tf.gather(train_x, train_susceptibles)

train_resistants = np.where(train_y == 1)[0]
train_res_samples = tf.gather(train_x, train_resistants)


## testing sets
test_x, test_y = X[to_split:], Y[to_split:]
# print("after split test")
# print(test_x.shape, test_y.shape)

test_susceptibles = np.where(test_y == -1)[0]
test_sus_samples = tf.gather(test_x, test_susceptibles)

test_resistants = np.where(test_y == 1)[0]
test_res_samples = tf.gather(test_x, test_resistants)


#1, the same. 0, not the same. goal is to produce pairs and targets
class SiameseModel:
    def __init__(self, data_subsets = ["train", "val"]):
        self.data = {}
        self.sus_samples = {} #samples that are susceptible
        self.res_samples = {} #samples that are resistant

        self.data["train"] = train_x
        self.data["val"] = test_x

        self.sus_samples["train"] = train_sus_samples
        self.res_samples["train"] = train_res_samples

        self.sus_samples["val"] = test_sus_samples
        self.res_samples["val"] = test_res_samples
         
    def generate_pairs(self,batch_size,s="train"):
        x = self.data[s]
        sus_samples = self.sus_samples[s]
        res_samples = self.res_samples[s]

        rng = np.random.default_rng()
        # initialize 2 empty arrays for paired batches
        pairs=[np.zeros((batch_size, num_features)) for i in range(2)]

        # initialize vector for the targets
        targets=np.zeros((batch_size))

        # create vector of 0s (should be different classes) and 1s (should be same)
        categories = rng.choice([-1, 1], size=(batch_size), replace=True)

        #first half will be 0s, second half will be 1s
        targets[batch_size//2:] = 1

        for i in range(batch_size):
            category = categories[i]

            samples, num_samples = None, 0
            if category == 1: # if resistant
                samples, num_samples = res_samples, len(res_samples)
            else:
                samples, num_samples = sus_samples, len(sus_samples)

            idx_1 = random.randint(0, num_samples - 1)

            pairs[0][i,:] = samples[idx_1]
   
            if i >= batch_size // 2: # if in second half, where targets are 0s, must choose from diff category
                if category == 1:  #if resistant, pair should be sus
                    samples, num_samples = sus_samples, len(sus_samples)
                else: #else is sus, so pair should be res
                    samples, num_samples = res_samples, len(res_samples)
            
            idx_2 = random.randint(0, num_samples - 1)
            pairs[1][i, :] = samples[idx_2]
          
                
        return pairs, targets

    def generate(self, batch_size, s="train"):
        """a generator for batches, so model.fit_generator can be used. """
        while True:
            pairs, targets = self.generate_pairs(batch_size,s)
            yield (pairs, targets) 

    def test_oneshot(self,model,N,k,s="val",verbose=0):
        """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
        n_correct = 0
        if verbose:
            print("Evaluating model on {} random {} way one-shot learning tasks ... \n".format(k,N))
        for i in range(k):
            inputs, targets = self.make_oneshot_task(N,s)
            probs = model.predict(inputs)
            if np.argmax(probs) == np.argmax(targets):
                n_correct+=1
        percent_correct = (100.0*n_correct / k)
        if verbose:
            print("Got an average of {}% {} way one-shot learning accuracy \n".format(percent_correct,N))
        return percent_correct

    def train(self, model, epochs, verbosity):
        model.fit_generator(self.generate(batch_size))

    def make_oneshot_task(self,N,s="val",language=None):
        """Create pairs of test image, support set for testing N way one-shot learning. """
        x=self.data[s]
        sus_samples = self.sus_samples[s]
        res_samples = self.res_samples[s]
        rng = np.random.default_rng()

        n_examples, n_features = x.shape

        categories = rng.choice([-1, 1], size=(batch_size), replace=True)
        targets = rng.choice([0, 1], size=(batch_size), replace=True)
        pairs=[np.zeros((batch_size, num_features)) for i in range(2)]

        for i in range(batch_size):
            category = categories[i]
            target = targets[i]

            samples, num_samples = None, 0
            if category == 1:
                samples, num_samples = res_samples, len(res_samples)
            else:
                samples, num_samples = sus_samples, len(sus_samples)

            idx_1 = random.randint(0, num_samples - 1)
            pairs[0][i,:] = samples[idx_1]

            if not target:
                if category == 1:  
                    samples, num_samples = sus_samples, len(sus_samples)
                else:
                    samples, num_samples = res_samples, len(res_samples)
            
            idx_2 = random.randint(0, num_samples - 1)
            pairs[1][i, :] = samples[idx_2]
                
        return pairs, targets


def initialize_weights(shape, name=None, dtype=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer weights with mean as 0.0 and standard deviation of 0.01
    """
    return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)

def initialize_bias(shape, name=None, dtype=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
    """
    return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)

def get_siamese_model(input_shape):
    """
        Model architecture
    """

    # Define the tensors for the two inputs
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    
    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv1D(64, 10, activation='relu', input_shape=input_shape,
                   kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling1D())
    model.add(Conv1D(128, 7, activation='relu', kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling1D())
    model.add(Conv1D(128, 4, activation='relu', kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling1D())
    model.add(Conv1D(256, 4, activation='relu', kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(512, activation='sigmoid',
                   kernel_regularizer=l2(1e-3),
                   kernel_initializer=initialize_weights,bias_initializer=initialize_bias))
    
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    
    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid',bias_initializer=initialize_bias)(L1_distance)
    
    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    
    # return the model
    optimizer = Adam(lr = 0.00006)
    siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)
    return siamese_net


loader = SiameseModel()
# loader.train(model, 10, 1)

#Training loop
print("Begin training")
evaluate_every = 1 # interval for evaluating on one-shot tasks
loss_every= 50 # interval for printing loss (iterations)
batch_size = 248
n_iter = 200
N_way = 2 # how many classes for testing one-shot tasks
n_val = 50 #how mahy one-shot tasks to validate on
best = -1
print("training")
for i in range(1, n_iter):
    (inputs,targets)=loader.generate_pairs(batch_size)
    siamese_net = get_siamese_model((1503, 1)) ## NOTE: not sure if this input shape is correct?
    loss = siamese_net.train_on_batch(inputs,targets)
    print("loss", loss)
    if i % evaluate_every == 0:
        print("evaluating")
        val_acc = loader.test_oneshot(siamese_net,N_way,n_val,verbose=True)
        if val_acc >= best:
            print("saving val acc")
            best=val_acc

    if i % loss_every == 0:
        print("iteration {}, training loss: {:.2f},".format(i,loss))
