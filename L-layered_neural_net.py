import numpy as np

# Relu activation function
def relu(Z):
    A = np.maximum(Z, 0)
    return A, Z

# Sigmoid activation function
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return A, Z

# Sigmoid backward function
def sigmoid_backward(dA, activation_cache):
    Z = activation_cache
    gZ = -(np.exp(-Z))/(np.power((1+np.exp(-Z)), 2))
    dZ = dA * gZ
    return dZ

# Derivative of a relu function
def relu_derivative(Z):
    X = Z.copy(deep=True)
    X[X>0] = 1
    X[X<=0] = 0
    return X

# Relu backward function
def relu_backward(dA, activation_cache):
    Z = activation_cache
    gZ = relu_derivative(Z)
    dZ = dA * gZ
    return dZ


# Inititalizing parameters function
def initialize_params(layer_dims):
    
    np.random.seed(3)
    
    parameteres={}
    L = len(layer_dims)
    
    for l in range(1,L):
        parameters['W'+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])
        parameters['b'+str(l)] = np.zeros((layer_dims[l], 1))
    
    return parameters

# General 1 step forward function
def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    
    return Z, cache

# Activating the linear forward output using sigmoid and relu
def linear_activation_forward(A_prev, W, b, activation):
    
    if (activation=="sigmoid"):
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    if (activation=="relu"):
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    cache = (linear_cache, activation_cache)
    
    return A, cache

# forward propagation function
def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    
    for l in range(1,L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W'+str(l)],parameters['b'+str(l)] , activation='relu')
        caches.append(cache)
    
    AL, cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], activation='sigmoid')
    caches.append(cache)
    
    return AL, caches

# cost function
def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = (-1/m)*np.sum(np.multiply(Y, np.log(AL)), np.multiply(Y-1, np.log(1-AL)))
    cost = np.squeeze(cost)
    
    return cost

# General 1 step backward 
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = (1/m)*np.dot(dZ, A_prev.T)
    db = (1/m)* np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db

# backward activation of the linear step
def linear_activation_backward(dA, cache, activation):
    
    linear_cache, activation_cache = cache
    
    if (activation=='sigmoid'):
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    if (activation=='relu'):
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

# backward propagation function
def l_model_backward(AL, Y, caches):
    grads = {}
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    current_cache = caches[-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation='sigmoid')
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA_prev_temp, current_cache, activation='relu')
        grads["dA"+str(l)] = dA_prev_temp
        grads["dW"+str(l+1)] = dW_temp
        grads["db"+str(l+1)] = db_temp
    
    return grads

# updating parameters function
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters['W'+str(l+1)] = parameters['W'+str(l+1)] - learning_rate * grads['dW'+str(l+1)]
        parameters['b'+str(l+1)] = parameters['b'+str(l+1)] - learning_rate * grads['db'+str(l+1)]
    
    return parameters

# Neural Network model
# Learning step
def L_layer_model(X, Y, layer_dims, learning_rate, num_iterations, print_cost=False):
    np.random.seed(1)
    costs=[]
    
    #Initialize parameters
    parameters = initialize_params(layer_dims)
    
    #gradient descent
    for i range(0, num_iterations):
        
        #Forward propagation
        AL, caches = L_model_forward(X, parameters)
        
        #computing cost
        cost = compute_cost(AL, Y)
        
        #backward propagation
        grads = l_model_backward(Y, AL, caches)
        
        #update parameters
        parameters = update_parameters(parameters, grads, learning_rate)
        
        #printing and appending cost
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plotting the cost    
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

