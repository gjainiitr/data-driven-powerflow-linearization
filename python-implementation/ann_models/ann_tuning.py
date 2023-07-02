# May have to write custom loss and metrics function
# very imp. to change neuron structure (V.V.Imp)
# If activation func is linear,then Adam is required or not??

import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import SGD

P = pd.read_csv('data/30P.csv', header = None)
Q = pd.read_csv('data/30Q.csv', header = None)
V = pd.read_csv('data/30V.csv', header = None)
Va = pd.read_csv('data/30Va.csv', header = None)

X = Va.join(V, lsuffix='_Va', rsuffix='_V')
Y = P.join(Q, lsuffix='_P', rsuffix='_Q')

#Train_test splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Function to create model, required for KerasRegressor
def create_model(optimizer='adam'):
    model =Sequential()
    model.add(Dense(num_bus*2,input_dim =num_bus*2,kernel_initializer='normal',kernel_regularizer=regularizers.l2(0.001),activation='linear'))
    model.add(Dense(num_bus*2,kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='linear'))
    model.compile(loss=mean_absolute_percentage_error, optimizer=optimizer, metrics=[mean_absolute_percentage_error])
    return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


model = KerasRegressor(build_fn=create_model, epochs=100, batch_size=10, verbose=0)


# define the grid search parameters
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100,200,300,500,1000]
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
weight_constraint = [1, 2, 3, 4, 5]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

'''
100 neurons : image classification, text detection, ocr
We have to test out multiple skeletons
Less neurons preferred

Various architectures (examples) -
input -> 2*num_bus -> output
input -> num_bus/4 -> num_bus/4 -> output
input -> num_bus/4 -> num_bus/2 -> num_bus/4 -> output


Select parameters in groups and not all at once as it will take way more time
1. batch_size & epochs
2. optimizer
3. learn_rate & momentum
4. init_mode
5. activation
6. weight_constraint & dropout_rate

It is also important to adjust number of neurons (most important tuning)
'''
param_grid = dict()

'''
n_jobs = -1 : all cores of pc : error = Waiting for existing lock by process '55614' -> n_jobs=1    

'''
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, Y)



# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))