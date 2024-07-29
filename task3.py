import numpy as np
import gzip
import pickle
from urllib import request

# Load the MNIST dataset
def load_data():
    url = "http://yann.lecun.com/exdb/mnist/mnist.pkl.gz"
    filename = "mnist.pkl.gz"
    request.urlretrieve(url, filename)

    with gzip.open(filename, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    return train_set, valid_set, test_set

# Activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

# Initialize weights and biases
def initialize_parameters(input_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

# Forward propagation
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# Backward propagation
def backward_propagation(X, Y, Z1, A1, Z2, A2, W1, W2, learning_rate):
    m = X.shape[0]
    dZ2 = A2 - Y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_derivative(A1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    
    return W1, b1, W2, b2

# Convert labels to one-hot encoding
def one_hot_encode(labels, num_classes):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

# Train the neural network
def train(X_train, y_train, X_val, y_val, input_size, hidden_size, output_size, learning_rate, epochs):
    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)
    y_train_encoded = one_hot_encode(y_train, output_size)
    y_val_encoded = one_hot_encode(y_val, output_size)
    
    for epoch in range(epochs):
        Z1, A1, Z2, A2 = forward_propagation(X_train, W1, b1, W2, b2)
        W1, b1, W2, b2 = backward_propagation(X_train, y_train_encoded, Z1, A1, Z2, A2, W1, W2, learning_rate)
        
        if epoch % 10 == 0:
            train_loss = -np.mean(np.sum(y_train_encoded * np.log(A2 + 1e-8), axis=1))
            val_Z1, val_A1, val_Z2, val_A2 = forward_propagation(X_val, W1, b1, W2, b2)
            val_loss = -np.mean(np.sum(y_val_encoded * np.log(val_A2 + 1e-8), axis=1))
            train_acc = np.mean(np.argmax(A2, axis=1) == y_train)
            val_acc = np.mean(np.argmax(val_A2, axis=1) == y_val)
            print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    return W1, b1, W2, b2

# Evaluate the model
def evaluate(X, y, W1, b1, W2, b2):
    _, _, _, A2 = forward_propagation(X, W1, b1, W2, b2)
    predictions = np.argmax(A2, axis=1)
    accuracy = np.mean(predictions == y)
    return accuracy

# Main function
def main():
    train_set, valid_set, test_set = load_data()
    X_train, y_train = train_set
    X_val, y_val = valid_set
    X_test, y_test = test_set
    
    input_size = 784
    hidden_size = 64
    output_size = 10
    learning_rate = 0.1
    epochs = 100
    
    W1, b1, W2, b2 = train(X_train, y_train, X_val, y_val, input_size, hidden_size, output_size, learning_rate, epochs)
    
    test_accuracy = evaluate(X_test, y_test, W1, b1, W2, b2)
    print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "_main_":
    main()