import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Neural Network implementation
class NeuralNetwork:
 def __init__(self, input_size, hidden_size, output_size):
  self.input_size = input_size
  self.hidden_size = hidden_size
  self.output_size = output_size
  self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
  self.bias_hidden = np.zeros((1, self.hidden_size))
  self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)
  self.bias_output = np.zeros((1, self.output_size))
 def sigmoid(self, x):
     return 1 / (1 + np.exp(-x))
 def sigmoid_derivative(self, x):
  return x * (1 - x)
 def forward(self, X):
  self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
  self.hidden_output = self.sigmoid(self.hidden_input)
  self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
  self.final_output = self.sigmoid(self.final_input)
  return self.final_output
 def backward(self, X, y, output):
  self.error = y - output
  self.delta_output = self.error * self.sigmoid_derivative(output)
  self.error_hidden = self.delta_output.dot(self.weights_hidden_output.T)
  self.delta_hidden = self.error_hidden * self.sigmoid_derivative(self.hidden_output)
  self.weights_hidden_output += self.hidden_output.T.dot(self.delta_output)
  self.bias_output += np.sum(self.delta_output, axis=0, keepdims=True)
  self.weights_input_hidden += X.T.dot(self.delta_hidden)
  self.bias_hidden += np.sum(self.delta_hidden, axis=0)
 def train(self, X, y, epochs):
  for _ in range(epochs):
   output = self.forward(X)
  self.backward(X, y, output)
 def predict(self, X):
  return np.argmax(self.forward(X), axis=1)
# Set random seed for reproducibility
np.random.seed(42)
# Initialize the neural network
input_size = X_train.shape[1]
hidden_size = 8
output_size = len(np.unique(y_train))
epochs = 10000
nn = NeuralNetwork(input_size, hidden_size, output_size)
# Train the neural network
nn.train(X_train, np.eye(output_size)[y_train], epochs)
# Make predictions on the test set
y_pred = nn.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

