import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from pandas.plotting import radviz


def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)

    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))

    parameters = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    return parameters

def forward_propagation(X, parameters):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    z1 = np.dot(w1, X) + b1     
    a1 = np.tanh(z1)            
    z2 = np.dot(w2, a1) + b2
    a2 = 1 / (1 + np.exp(-z2))  

    cache = {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}

    return a2, cache

def compute_cost(a2, Y):
    m = Y.shape[1]     

    logprobs = np.multiply(np.log(a2), Y) + np.multiply((1 - Y), np.log(1 - a2))
    cost = - np.sum(logprobs) / m

    return cost

def backward_propagation(parameters, cache, X, Y):
    m = Y.shape[1]

    w2 = parameters['w2']

    a1 = cache['a1']
    a2 = cache['a2']

    dz2 = a2 - Y
    dw2 = (1 / m) * np.dot(dz2, a1.T)
    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
    dz1 = np.multiply(np.dot(w2.T, dz2), 1 - np.power(a1, 2))
    dw1 = (1 / m) * np.dot(dz1, X.T)
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

    grads = {'dw1': dw1, 'db1': db1, 'dw2': dw2, 'db2': db2}

    return grads

def update_parameters(parameters, grads, learning_rate=0.4):
    
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    dw1 = grads['dw1']
    db1 = grads['db1']
    dw2 = grads['dw2']
    db2 = grads['db2']

    w1 = w1 - dw1 * learning_rate
    b1 = b1 - db1 * learning_rate
    w2 = w2 - dw2 * learning_rate
    b2 = b2 - db2 * learning_rate

    parameters = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    return parameters

def test_diff_learning_rate(x, learning_rate, num_iterations):

    import numpy as np
    import matplotlib.pyplot as plt

    def f(x):
        return (x - 3) ** 2

    def df(x):
        return 2 * (x - 3)

    # 儲存初始參數
    x_values = [x]
    f_values = [f(x)]

    # 梯度下降
    for _ in range(num_iterations):
        x = x - learning_rate * df(x)  # 更新x
        x_values.append(x)
        f_values.append(f(x))
        
    # 資料視覺化
    print(f"最終的x值: {x}")
    print(f"最終的f(x)值: {f(x)}")

    x_range = np.linspace(-1, 7, 100)
    y_range = f(x_range)

    plt.plot(x_range, y_range, label='f(x) = (x - 3)^2')
    plt.scatter(x_values, f_values, color='red')
    for i in range(num_iterations):
        plt.annotate(f'{i+1}', (x_values[i], f_values[i]))

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Gradient Descent')
    plt.legend()
    plt.show()
    
def show_tanh_function():
  x = np.linspace(-10, 10, 400)
  y_tanh = np.tanh(x)

  plt.figure(figsize=(10, 6))
  plt.plot(x, y_tanh, label='y = tanh(x)', color='green')
  plt.title('Hyperbolic Tangent Function')
  plt.xlabel('x')
  plt.ylabel('tanh(x)')
  plt.grid(True)
  plt.legend()

  plt.show()
  
def show_sigmoid_function():

  x = np.linspace(-10, 10, 400)
  y_sigmoid = 1 / (1 + np.exp(-x))

  plt.figure(figsize=(10, 6))
  plt.plot(x, y_sigmoid, label='y = 1 / (1 + exp(-x))', color='blue')
  plt.title('Sigmoid Function')
  plt.xlabel('x')
  plt.ylabel('Sigmoid(x)')
  plt.grid(True)
  plt.legend()

  plt.show()
    
def show_cross_entropy_loss():
    
    import matplotlib.pyplot as plt
    predicted_probabilities = np.linspace(0.01, 1, 100)

    loss_when_true = -np.log(predicted_probabilities)
    loss_when_false = -np.log(1 - predicted_probabilities)

    plt.figure(figsize=(10, 6))
    plt.plot(predicted_probabilities, loss_when_true, label='True Label = 1')
    plt.plot(predicted_probabilities, loss_when_false, label='True Label = 0')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Cross Entropy Loss')
    plt.title('Cross Entropy Loss vs. Predicted Probability')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def evaluate_accuracy(data_test, parameters):

  x_test = data_test.iloc[:, 0:4].values.T
  y_test = data_test.iloc[:, 4:].values.T
  y_test = y_test.astype('uint8')

  w1 = parameters['w1']
  b1 = parameters['b1']
  w2 = parameters['w2']
  b2 = parameters['b2']

  z1 = np.dot(w1, x_test) + b1
  a1 = np.tanh(z1)
  z2 = np.dot(w2, a1) + b2
  a2 = 1 / (1 + np.exp(-z2))

  class_num = y_test.shape[0]
  sample_num = y_test.shape[1]

  output = np.empty(shape=(class_num, sample_num), dtype=int)

  for i in range(class_num):
      for j in range(sample_num):
          if a2[i][j] > 0.5:
              output[i][j] = 1
          else:
              output[i][j] = 0

  count = 0
  for k in range(0, sample_num):
      if output[0][k] == y_test[0][k] and output[1][k] == y_test[1][k] and output[2][k] == y_test[2][k]:
          count = count + 1

  accuracy = count / int(y_test.shape[1]) * 100
  print(f'Accuracy: {accuracy}')

def build_and_train(input_dim, hidden_dim, output_dim, epochs, learning_rate):

  import numpy as np
  from sklearn.datasets import load_breast_cancer
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import StandardScaler, OneHotEncoder

  def sigmoid(x):
      return 1 / (1 + np.exp(-x))

  def sigmoid_derivative(x):
      return x * (1 - x)

  def compute_loss(y_true, y_pred):
      epsilon = 1e-12
      y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
      return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

  data = load_breast_cancer()
  X = data.data
  y = data.target.reshape(-1, 1) 

  encoder = OneHotEncoder(sparse_output=False)
  y = encoder.fit_transform(y)

  scaler = StandardScaler()
  X = scaler.fit_transform(X)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  np.random.seed(42)
  W1 = np.random.randn(input_dim, hidden_dim) 
  b1 = np.zeros((1, hidden_dim))
  W2 = np.random.randn(hidden_dim, output_dim)
  b2 = np.zeros((1, output_dim))

  loss_list = []

  for epoch in range(epochs):
    
      z1 = np.dot(X_train, W1) + b1
      a1 = sigmoid(z1)
      z2 = np.dot(a1, W2) + b2
      y_pred = sigmoid(z2)
      
      loss = compute_loss(y_train, y_pred)
      loss_list.append(loss)
      
      d_loss_y_pred = (y_pred - y_train) / y_train.shape[0]
      d_y_pred_z2 = sigmoid_derivative(y_pred)
      d_z2_a1 = W2
      d_a1_z1 = sigmoid_derivative(a1)
      d_z1_W1 = X_train

      d_z2 = d_loss_y_pred * d_y_pred_z2
      d_W2 = np.dot(a1.T, d_z2)
      d_b2 = np.sum(d_z2, axis=0, keepdims=True)
      
      d_a1 = np.dot(d_z2, W2.T)
      d_z1 = d_a1 * d_a1_z1
      d_W1 = np.dot(d_z1_W1.T, d_z1)
      d_b1 = np.sum(d_z1, axis=0, keepdims=True)
      
      
      W1 -= learning_rate * d_W1
      b1 -= learning_rate * d_b1
      W2 -= learning_rate * d_W2
      b2 -= learning_rate * d_b2
      
      if epoch % 1000 == 0:
          print(f"Epoch {epoch}, Loss: {loss}")

  z1_test = np.dot(X_test, W1) + b1
  a1_test = sigmoid(z1_test)
  z2_test = np.dot(a1_test, W2) + b2
  y_pred_test = sigmoid(z2_test)
  y_pred_test = (y_pred_test == y_pred_test.max(axis=1, keepdims=True)).astype(int) 

  accuracy = np.mean(np.all(y_pred_test == y_test, axis=1))
  print(f"Test Accuracy: {accuracy}")

  return loss_list

def test_gradient_descent(x, learning_rate, num_iterations):
    import numpy as np
    import matplotlib.pyplot as plt

    def f(x):
        return (x - 3) ** 4

    def df(x):
        return 4 * (x - 3) ** 3

    x_values = [x]
    f_values = [f(x)]

    for _ in range(num_iterations):
        x = x - learning_rate * df(x)  # 更新x
        x_values.append(x)
        f_values.append(f(x))

    print(f"最終的x值: {x}")
    print(f"最終的f(x)值: {f(x)}")

    x_range = np.linspace(-1, 7, 100)
    y_range = f(x_range)

    plt.plot(x_range, y_range, label='f(x) = (x - 3)^2')
    plt.scatter(x_values, f_values, color='red')
    for i in range(num_iterations):
        plt.annotate(f'{i+1}', (x_values[i], f_values[i]))

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Gradient Descent')
    plt.legend()
    plt.show()

def get_cats_image():

  import requests
  from PIL import Image
  import numpy as np
  from io import BytesIO

  url = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/93/Cat_poster_2.jpg/1920px-Cat_poster_2.jpg"
  headers = {
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

  try:
      response = requests.get(url, headers=headers)
      response.raise_for_status() 
      img = Image.open(BytesIO(response.content))
      img.verify() 

      img = Image.open(BytesIO(response.content))
      img_resized = img.resize((380, 300))
      img_array = np.array(img_resized)

  except requests.exceptions.RequestException as e:
      print(f"Error downloading the image: {e}")
  except (Image.UnidentifiedImageError, Image.DecompressionBombError) as e:
      print(f"Error processing the image: {e}")
  except Exception as e:
      print(f"An unexpected error occurred: {e}")
    
  return img_array

def gd_and_gd_with_momentum(x_init, learning_rate, num_iterations, momentum):
  
  import numpy as np
  import matplotlib.pyplot as plt
  import matplotlib.animation as animation

  # 定義損失函數及其梯度
  def loss_function(x):
      return x**4 - 6*x**3 + 8*x**2

  def gradient(x):
      return 4*x**3 - 18*x**2 + 16*x

  gd_path = [x_init]
  gd_momentum_path = [x_init]

  # 初始化速度變量
  v = 0

  # 進行梯度下降和帶有 momentum 的梯度下降
  x_gd = x_init
  x_gd_momentum = x_init

  for i in range(num_iterations):

      grad = gradient(x_gd)
      x_gd -= learning_rate * grad
      
      v = momentum * v + gradient(x_gd_momentum)
      x_gd_momentum -= learning_rate * v
      
      gd_path.append(x_gd)
      gd_momentum_path.append(x_gd_momentum)

  fig, ax = plt.subplots()
  x = np.linspace(-1, 4, 400)
  y = loss_function(x)
  ax.plot(x, y, label='Loss Function')

  gd_points, = ax.plot([], [], 'o-', label='GD')
  gd_momentum_points, = ax.plot([], [], 'o-', label='GD with Momentum')

  def init():
      gd_points.set_data([], [])
      gd_momentum_points.set_data([], [])
      return gd_points, gd_momentum_points

  def update(frame):
      gd_points.set_data(gd_path[:frame], loss_function(np.array(gd_path[:frame])))
      gd_momentum_points.set_data(gd_momentum_path[:frame], loss_function(np.array(gd_momentum_path[:frame])))
      return gd_points, gd_momentum_points

  ani = animation.FuncAnimation(fig, update, frames=num_iterations, init_func=init, blit=True)

  plt.xlabel('x')
  plt.ylabel('Loss')
  plt.title('Comparison of GD and GD with Momentum')
  plt.legend()
  plt.show()

  ani.save('gd_vs_gd_momentum.gif', writer='imagemagick')

def test_gd_with_momentum(x_init, learning_rate, num_iterations, momentum):

  import numpy as np
  import matplotlib.pyplot as plt
  import matplotlib.animation as animation

  def loss_function(x):
      return np.sin(x) + 0.1 * x**2

  def gradient(x):
      return np.cos(x) + 0.2 * x

  gd_path = [x_init]
  gd_momentum_path = [x_init]

  v = 0

  x_gd = x_init
  x_gd_momentum = x_init

  for i in range(num_iterations):
      # 標準梯度下降
      grad = gradient(x_gd)
      x_gd -= learning_rate * grad
      
      # 帶有 momentum 的梯度下降
      grad_momentum = gradient(x_gd_momentum)
      v = momentum * v + grad_momentum
      x_gd_momentum -= learning_rate * v
      
      gd_path.append(x_gd)
      gd_momentum_path.append(x_gd_momentum)

  fig, ax = plt.subplots()
  x = np.linspace(-5, 5, 400)
  y = loss_function(x)
  ax.plot(x, y, label='Loss Function')

  gd_points, = ax.plot([], [], 'o-', label='GD')
  gd_momentum_points, = ax.plot([], [], 'o-', label='GD with Momentum')

  def init():
      gd_points.set_data([], [])
      gd_momentum_points.set_data([], [])
      return gd_points, gd_momentum_points

  def update(frame):
      gd_points.set_data(gd_path[:frame], loss_function(np.array(gd_path[:frame])))
      gd_momentum_points.set_data(gd_momentum_path[:frame], loss_function(np.array(gd_momentum_path[:frame])))
      return gd_points, gd_momentum_points

  ani = animation.FuncAnimation(fig, update, frames=num_iterations, init_func=init, blit=True)

  plt.xlabel('x')
  plt.ylabel('Loss')
  plt.title('Comparison of GD and GD with Momentum')
  plt.legend()
  plt.show()

  ani.save('gd_vs_gd_momentum_sine.gif', writer='imagemagick')


