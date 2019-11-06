
# Amphi 1 - Feedforward Neural Networks

# 1. Introduction

In lesson 6, 7, 8, 9 of DSC101, some linear models were introduced:
- Linear Regression
- Logistic Regression
- Linear SVM (Support Vector Machine with Linear Kernel)
- Perceptron
- Linear Discriminant Analysis

In reality, these models are usually not adaptable because non-linearly separability of problems.

Other methods are based on discriminant methods:

- Gaussian Naives Bayes
- Quadratic Discriminant Analysis (QDA)

They are quadratic methods and also not adaptable to more complex reality problems.

Some generalization strategies were introduced. They share the same idea: explore a larger function class:

- Consider more complex function class: Polynomial Regression. (Higher degree means more parameters, more complex model). Similarly to Polynomial Regression, we can use Piecewise Linear Functions. Complexity can be adapted by the number of linear pieces, but it depends on the concrete problem.

- Use kernel method: Transform the optimization problem to dual form. Use more complex kernel: polynomial, sigmoid, tanh, rbf...  Each kernel is in fact equivalent to a function class (e.g., linear kernel equivalent to linear model i.e. linear boundary)

In this lesson, we would like to use another approach to consider a more general class of functions using composite functions. The idea:

- In linear methods, the output is predicted by
$$
f(\mathbf x) = w(\mathbf x)
$$

for regression, or
$$
f(\mathbf x) = 1_{w(\mathbf x) \geq 0}
$$

for binary classification, where $w$ is a linear function.

Now if we allow the predicted output to be of the form:
$$
f(\mathbf x) = w_{l}(w_{l-1}(\ldots w_1(\mathbf x)))
$$

and
$$
f(\mathbf x) = \mathbf 1 (w_{l}(w_{l-1}(\ldots w_1(\mathbf x))) \geq 0)
$$


where $w_l, w_{l-1}, \ldots, w_1$ are functions, some of them are linear while the others are not, we can target more complex function forms.

For example, we can think of $w_1, w_3, \ldots, w_{9}$ as linear functions and $w_2, w_4, \ldots, w_8$ as the function $\mathbf x 1_{\mathbf x \geq 0}$ applied to each coordinate of $\mathbf x$, this will make the output predicted by piecewise linear functions.

A model where prediction can be done by a composition of alternatively linear functions and element-wise non-linear functions are called *Feedforward Neural Network* or *Multilayer Perceptron*. 

# 2. Graphical Representation and Terminology

## 2.1 Graphical Representation

Suppose that we use a Feedforward Network that suggests a function class ($\mathbf R^D \to \mathbf R^K$) of the form:

$$
\mathbf x \mapsto g(\mathbf V(h(\mathbf W \mathbf x + \mathbf c)) + \mathbf d)
$$

where $x \in \mathbf R^D$; $\mathbf W$ is a $M \times D$-matrix; $\mathbf c$ is an $M-$dimensional vector, $h$ is a non-linear function $\mathbf R^M \to \mathbf R^M$ acting element-wisely; $V$ is a $K \times M$ matrix; $\mathbf d$ is a $K-$dimensional vector and $g$ is a non-linear function   $\mathbf R^K \to \mathbf R^K$ acting element-wisely.

We can sketch a graph of this FNN:

<img src="https://raw.githubusercontent.com/riduan91/DSC111/master/Lesson1/Amphi/F3.png" width=800></img>

This is a composition function of linear and non-linear function consecutively.

In general, a FFN is like this:

<img src="https://raw.githubusercontent.com/riduan91/DSC111/master/Lesson1/Amphi/F4.png" width=800></img>

where $h^{(l)}$ are known and chosen by us.

This represents:

$$
\mathbf z^{(0)} \mapsto \mathbf z^{(L)} =  h^{(L)} \left( \mathbf W^{(L)} \left( h^{(L-1)}\left( \ldots \left( \mathbf h^{(1)} (\mathbf W^{(1)} \mathbf z^{(0)} + \mathbf W_0^{(1)})\right) \ldots \right)\right) + \mathbf W_0^{(L)} \right)
$$

## 2.2 Terminology

- $L$ is called the depth of the network. It equals the number of linear functions in the composition.
- The network can be **fully connected** or not.
- $\max(D_0, D_1, \ldots, D_L)$ is sometimes called **width** of the network.
- The **layer** $l$ (the $l$-th layer) can refer to

 - $(\mathbf z^{(l)})$
or

 - $(\mathbf W^{(l)})$
or

 - $(\mathbf z^{(l-1)}, \mathbf W^{(l)}, \mathbf a^{(l)}, h^{(l)}, z^{(l)})$
- Each coordinate $z_j^{(i)}$ is called a **unit** of the layer $i$.
- The layer 0 is called the **input layer**.
- The layer $L$ is called the **output layer**.
- The layers $1, 2, \ldots, L-1$ are called **hidden layers**.
- The functions $h^{(l)}$ are called **activation functions**.
- $\mathbf w_{j,0}^{(l)}, j = 1, \ldots, D_l$ are called **bias**.
- $\mathbf w_{j,i}^{(l)}, j = 1, \ldots, D_l, i =1 , \ldots, D_{l-1}$ are called **weights**.

By **deep learning** we mean methods considering classes of functions that are composition of several transformations on the input. Feed forward networks are example of deep learning models (if depth >=2). 

Like many other models, we need to define some quantity depending on the unknown parameters and find functions (i.e., the value of these parameters) in this class that optimize the quantity. The quantity can be a sum of losses between true values and predicted values (Empirical Risk Minimization) or the likelihood of observing the training data (maximum likelihood).

## 2.3 Basic Models as 1-layer Feedforward Network

**Linear Regression**

Linear Regression is an example of Feed Forward Network where:
- $L=1$: the network has depth 1.
- The output layer is of size (width) 1.
- The activation function $h^{(1)}$ is the identity function
- The loss function is (mean) squared loss.

<img src="https://raw.githubusercontent.com/riduan91/DSC111/master/Lesson1/Amphi/F6.png" width=400></img>

**(Binary) Logistic Regression**
Logistic Regression is another example where:

- $L=1$
- The output layer has size 1
- The activation function $h^{(1)}$ is $\sigma(a) = \frac1{1+\exp(-a)}$
- The loss function is the **binary cross-entropy** function
$$
L(t, y) = -t \log y - (1-t) \log (1-y)
$$

<img src="https://raw.githubusercontent.com/riduan91/DSC111/master/Lesson1/Amphi/F7.png" width=400></img>

**Multiclass Logistic Regression with Multinomial Distribution Strategy (also called Softmax Regression)**

- $L = 1$
- The output layer has size $K$.
- The activation function $h^{(1)}$ is the softmax function
$$
\sigma(a_k) = \frac{\exp(a_k)}{\sum_{j=1}^K \exp(a_j)}
$$
- The loss function is the **cross-entropy** function:
$$
L(t, y_1, \ldots, y_K) = -\sum_{j=1}^K t_j \log y_j
$$

where $t_j = 1$ if the true value $t$ belongs to class $j$ and 0 otherwise.

<img src="https://raw.githubusercontent.com/riduan91/DSC111/master/Lesson1/Amphi/F12.png" width=400></img>

## 2.4 Examples about XOR problem

Suppose $\mathbf x = (x_1, x_2) \in \mathbf \{0, 1\}^2$ and $t \in \{0, 1\}$ is a function of $\mathbf x$.

**If $t = x_1 \textrm{ and } x_2$**, a 1-layer network for binary classification (e.g., logistic regression) can efficiently solve the problem.

<img src="https://raw.githubusercontent.com/riduan91/DSC111/master/Lesson1/Amphi/F8.png" width=400></img>

Similarly, **if  $t = x_1 \textrm{ or } x_2$**

<img src="https://raw.githubusercontent.com/riduan91/DSC111/master/Lesson1/Amphi/F9.png" width=400></img>

**if $t = (\textrm{not } x_1) \textrm { or } (\textrm{not } x_2)$**

<img src="https://raw.githubusercontent.com/riduan91/DSC111/master/Lesson1/Amphi/F10.png" width=400></img>

However, if $t = x_1 \textrm{ xor } x_2$, a 1-layer FFN cannot solve the problem. Note that we have:
$$
x_1 \textrm{ xor } x_2 = (x_1 \textrm{ or } x_2) \textrm{ and } ((\textrm{not } x_1) \textrm { or } (\textrm{not } x_2))
$$

So, the following 2-FNN can help solve the problem:
<img src="https://raw.githubusercontent.com/riduan91/DSC111/master/Lesson1/Amphi/F11.png" width=600></img>

# 3. Keras

## 3.1 Installation

**TensorFlow** is a Python library/framework for numerical computation that are used in machine learning. It implements tensors, which help to define deep learning models' architecture.

**Keras** is a Python library that can run on top of **Tensorflow** (and other frameworks like Theano). It simplifies the way we code in defining deep learning models.

To install **Tensorflow** and **Keras**, Python 3 is recommended, but with version 3.5

`conda create -n py35 python=3.5`

`conda activate py35`

`conda install anaconda`

`conda install opencv`

`conda install theano`

`conda install tensorflow`

`conda install keras`

## 3.2 Example: AND problem


```python
import numpy as np
np.random.seed(1234)
NB_SAMPLES = 1000
X = np.random.binomial(1, 0.6, size=(1000, 2))
y = X[:, 0] * X[:, 1]
```


```python
import pandas as pd
pd.DataFrame([X[:,0], X[:,1], y])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>990</th>
      <th>991</th>
      <th>992</th>
      <th>993</th>
      <th>994</th>
      <th>995</th>
      <th>996</th>
      <th>997</th>
      <th>998</th>
      <th>999</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 1000 columns</p>
</div>




```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
```


```python
from keras.models import Sequential
from keras.layers.core import Dense, Activation

model = Sequential()
model.add(Dense(1, input_shape = (2,)))
model.add(Activation('sigmoid'))
```


```python
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_11 (Dense)             (None, 1)                 3         
    _________________________________________________________________
    activation_11 (Activation)   (None, 1)                 0         
    =================================================================
    Total params: 3
    Trainable params: 3
    Non-trainable params: 0
    _________________________________________________________________
    


```python
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.initializers import RandomUniform
```


```python
OPTIMIZER = SGD(lr = 0.1)
BATCH_SIZE = 128
NB_EPOCH = 100
VALIDATION_SPLIT = 0.2
VERBOSE = 1

model.compile(loss='binary_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
```

    Train on 560 samples, validate on 140 samples
    Epoch 1/100
    560/560 [==============================] - 1s 1ms/step - loss: 0.9295 - acc: 0.4946 - val_loss: 0.9431 - val_acc: 0.4071
    Epoch 2/100
    560/560 [==============================] - 0s 21us/step - loss: 0.8803 - acc: 0.4518 - val_loss: 0.8928 - val_acc: 0.4071
    Epoch 3/100
    560/560 [==============================] - 0s 32us/step - loss: 0.8416 - acc: 0.4518 - val_loss: 0.8522 - val_acc: 0.4071
    Epoch 4/100
    560/560 [==============================] - 0s 41us/step - loss: 0.8105 - acc: 0.4518 - val_loss: 0.8193 - val_acc: 0.4071
    Epoch 5/100
    560/560 [==============================] - 0s 36us/step - loss: 0.7847 - acc: 0.4518 - val_loss: 0.7918 - val_acc: 0.4071
    Epoch 6/100
    560/560 [==============================] - 0s 30us/step - loss: 0.7628 - acc: 0.4518 - val_loss: 0.7674 - val_acc: 0.5929
    Epoch 7/100
    560/560 [==============================] - 0s 29us/step - loss: 0.7430 - acc: 0.6357 - val_loss: 0.7451 - val_acc: 0.5929
    Epoch 8/100
    560/560 [==============================] - 0s 21us/step - loss: 0.7252 - acc: 0.6357 - val_loss: 0.7264 - val_acc: 0.5929
    Epoch 9/100
    560/560 [==============================] - 0s 27us/step - loss: 0.7092 - acc: 0.6357 - val_loss: 0.7082 - val_acc: 0.5929
    Epoch 10/100
    560/560 [==============================] - 0s 21us/step - loss: 0.6939 - acc: 0.6357 - val_loss: 0.6928 - val_acc: 0.5929
    Epoch 11/100
    560/560 [==============================] - 0s 30us/step - loss: 0.6798 - acc: 0.6357 - val_loss: 0.6777 - val_acc: 0.5929
    Epoch 12/100
    560/560 [==============================] - 0s 21us/step - loss: 0.6663 - acc: 0.6357 - val_loss: 0.6633 - val_acc: 0.5929
    Epoch 13/100
    560/560 [==============================] - 0s 29us/step - loss: 0.6535 - acc: 0.6357 - val_loss: 0.6501 - val_acc: 0.5929
    Epoch 14/100
    560/560 [==============================] - 0s 29us/step - loss: 0.6410 - acc: 0.6357 - val_loss: 0.6369 - val_acc: 0.5929
    Epoch 15/100
    560/560 [==============================] - 0s 21us/step - loss: 0.6293 - acc: 0.6357 - val_loss: 0.6244 - val_acc: 1.0000
    Epoch 16/100
    560/560 [==============================] - 0s 20us/step - loss: 0.6179 - acc: 0.9089 - val_loss: 0.6127 - val_acc: 1.0000
    Epoch 17/100
    560/560 [==============================] - 0s 32us/step - loss: 0.6071 - acc: 1.0000 - val_loss: 0.6029 - val_acc: 1.0000
    Epoch 18/100
    560/560 [==============================] - 0s 23us/step - loss: 0.5966 - acc: 1.0000 - val_loss: 0.5919 - val_acc: 1.0000
    Epoch 19/100
    560/560 [==============================] - 0s 41us/step - loss: 0.5865 - acc: 1.0000 - val_loss: 0.5819 - val_acc: 1.0000
    Epoch 20/100
    560/560 [==============================] - 0s 25us/step - loss: 0.5766 - acc: 1.0000 - val_loss: 0.5725 - val_acc: 1.0000
    Epoch 21/100
    560/560 [==============================] - 0s 29us/step - loss: 0.5672 - acc: 1.0000 - val_loss: 0.5625 - val_acc: 1.0000
    Epoch 22/100
    560/560 [==============================] - 0s 32us/step - loss: 0.5581 - acc: 1.0000 - val_loss: 0.5535 - val_acc: 1.0000
    Epoch 23/100
    560/560 [==============================] - 0s 18us/step - loss: 0.5492 - acc: 1.0000 - val_loss: 0.5442 - val_acc: 1.0000
    Epoch 24/100
    560/560 [==============================] - 0s 34us/step - loss: 0.5404 - acc: 1.0000 - val_loss: 0.5357 - val_acc: 1.0000
    Epoch 25/100
    560/560 [==============================] - 0s 21us/step - loss: 0.5321 - acc: 1.0000 - val_loss: 0.5276 - val_acc: 1.0000
    Epoch 26/100
    560/560 [==============================] - 0s 27us/step - loss: 0.5241 - acc: 1.0000 - val_loss: 0.5191 - val_acc: 1.0000
    Epoch 27/100
    560/560 [==============================] - 0s 23us/step - loss: 0.5165 - acc: 1.0000 - val_loss: 0.5112 - val_acc: 1.0000
    Epoch 28/100
    560/560 [==============================] - 0s 20us/step - loss: 0.5089 - acc: 1.0000 - val_loss: 0.5035 - val_acc: 1.0000
    Epoch 29/100
    560/560 [==============================] - 0s 25us/step - loss: 0.5018 - acc: 1.0000 - val_loss: 0.4957 - val_acc: 1.0000
    Epoch 30/100
    560/560 [==============================] - 0s 20us/step - loss: 0.4946 - acc: 1.0000 - val_loss: 0.4887 - val_acc: 1.0000
    Epoch 31/100
    560/560 [==============================] - 0s 30us/step - loss: 0.4877 - acc: 1.0000 - val_loss: 0.4819 - val_acc: 1.0000
    Epoch 32/100
    560/560 [==============================] - 0s 23us/step - loss: 0.4811 - acc: 1.0000 - val_loss: 0.4754 - val_acc: 1.0000
    Epoch 33/100
    560/560 [==============================] - 0s 20us/step - loss: 0.4746 - acc: 1.0000 - val_loss: 0.4694 - val_acc: 1.0000
    Epoch 34/100
    560/560 [==============================] - 0s 30us/step - loss: 0.4683 - acc: 1.0000 - val_loss: 0.4629 - val_acc: 1.0000
    Epoch 35/100
    560/560 [==============================] - 0s 21us/step - loss: 0.4625 - acc: 1.0000 - val_loss: 0.4566 - val_acc: 1.0000
    Epoch 36/100
    560/560 [==============================] - 0s 29us/step - loss: 0.4565 - acc: 1.0000 - val_loss: 0.4509 - val_acc: 1.0000
    Epoch 37/100
    560/560 [==============================] - 0s 21us/step - loss: 0.4506 - acc: 1.0000 - val_loss: 0.4451 - val_acc: 1.0000
    Epoch 38/100
    560/560 [==============================] - 0s 23us/step - loss: 0.4450 - acc: 1.0000 - val_loss: 0.4399 - val_acc: 1.0000
    Epoch 39/100
    560/560 [==============================] - 0s 25us/step - loss: 0.4396 - acc: 1.0000 - val_loss: 0.4347 - val_acc: 1.0000
    Epoch 40/100
    560/560 [==============================] - 0s 29us/step - loss: 0.4343 - acc: 1.0000 - val_loss: 0.4295 - val_acc: 1.0000
    Epoch 41/100
    560/560 [==============================] - 0s 34us/step - loss: 0.4291 - acc: 1.0000 - val_loss: 0.4243 - val_acc: 1.0000
    Epoch 42/100
    560/560 [==============================] - 0s 21us/step - loss: 0.4241 - acc: 1.0000 - val_loss: 0.4193 - val_acc: 1.0000
    Epoch 43/100
    560/560 [==============================] - 0s 23us/step - loss: 0.4191 - acc: 1.0000 - val_loss: 0.4145 - val_acc: 1.0000
    Epoch 44/100
    560/560 [==============================] - 0s 27us/step - loss: 0.4144 - acc: 1.0000 - val_loss: 0.4099 - val_acc: 1.0000
    Epoch 45/100
    560/560 [==============================] - 0s 23us/step - loss: 0.4097 - acc: 1.0000 - val_loss: 0.4056 - val_acc: 1.0000
    Epoch 46/100
    560/560 [==============================] - 0s 25us/step - loss: 0.4052 - acc: 1.0000 - val_loss: 0.4008 - val_acc: 1.0000
    Epoch 47/100
    560/560 [==============================] - 0s 27us/step - loss: 0.4008 - acc: 1.0000 - val_loss: 0.3966 - val_acc: 1.0000
    Epoch 48/100
    560/560 [==============================] - 0s 25us/step - loss: 0.3966 - acc: 1.0000 - val_loss: 0.3919 - val_acc: 1.0000
    Epoch 49/100
    560/560 [==============================] - 0s 25us/step - loss: 0.3922 - acc: 1.0000 - val_loss: 0.3879 - val_acc: 1.0000
    Epoch 50/100
    560/560 [==============================] - 0s 21us/step - loss: 0.3881 - acc: 1.0000 - val_loss: 0.3839 - val_acc: 1.0000
    Epoch 51/100
    560/560 [==============================] - 0s 27us/step - loss: 0.3841 - acc: 1.0000 - val_loss: 0.3802 - val_acc: 1.0000
    Epoch 52/100
    560/560 [==============================] - 0s 25us/step - loss: 0.3802 - acc: 1.0000 - val_loss: 0.3763 - val_acc: 1.0000
    Epoch 53/100
    560/560 [==============================] - 0s 20us/step - loss: 0.3763 - acc: 1.0000 - val_loss: 0.3723 - val_acc: 1.0000
    Epoch 54/100
    560/560 [==============================] - 0s 20us/step - loss: 0.3726 - acc: 1.0000 - val_loss: 0.3684 - val_acc: 1.0000
    Epoch 55/100
    560/560 [==============================] - 0s 23us/step - loss: 0.3689 - acc: 1.0000 - val_loss: 0.3651 - val_acc: 1.0000
    Epoch 56/100
    560/560 [==============================] - 0s 25us/step - loss: 0.3653 - acc: 1.0000 - val_loss: 0.3616 - val_acc: 1.0000
    Epoch 57/100
    560/560 [==============================] - 0s 21us/step - loss: 0.3617 - acc: 1.0000 - val_loss: 0.3579 - val_acc: 1.0000
    Epoch 58/100
    560/560 [==============================] - 0s 20us/step - loss: 0.3583 - acc: 1.0000 - val_loss: 0.3543 - val_acc: 1.0000
    Epoch 59/100
    560/560 [==============================] - 0s 27us/step - loss: 0.3549 - acc: 1.0000 - val_loss: 0.3509 - val_acc: 1.0000
    Epoch 60/100
    560/560 [==============================] - 0s 25us/step - loss: 0.3517 - acc: 1.0000 - val_loss: 0.3471 - val_acc: 1.0000
    Epoch 61/100
    560/560 [==============================] - 0s 20us/step - loss: 0.3483 - acc: 1.0000 - val_loss: 0.3440 - val_acc: 1.0000
    Epoch 62/100
    560/560 [==============================] - 0s 20us/step - loss: 0.3452 - acc: 1.0000 - val_loss: 0.3408 - val_acc: 1.0000
    Epoch 63/100
    560/560 [==============================] - 0s 23us/step - loss: 0.3421 - acc: 1.0000 - val_loss: 0.3378 - val_acc: 1.0000
    Epoch 64/100
    560/560 [==============================] - 0s 27us/step - loss: 0.3390 - acc: 1.0000 - val_loss: 0.3348 - val_acc: 1.0000
    Epoch 65/100
    560/560 [==============================] - 0s 20us/step - loss: 0.3361 - acc: 1.0000 - val_loss: 0.3317 - val_acc: 1.0000
    Epoch 66/100
    560/560 [==============================] - 0s 18us/step - loss: 0.3331 - acc: 1.0000 - val_loss: 0.3288 - val_acc: 1.0000
    Epoch 67/100
    560/560 [==============================] - 0s 27us/step - loss: 0.3302 - acc: 1.0000 - val_loss: 0.3260 - val_acc: 1.0000
    Epoch 68/100
    560/560 [==============================] - 0s 29us/step - loss: 0.3274 - acc: 1.0000 - val_loss: 0.3232 - val_acc: 1.0000
    Epoch 69/100
    560/560 [==============================] - 0s 27us/step - loss: 0.3246 - acc: 1.0000 - val_loss: 0.3207 - val_acc: 1.0000
    Epoch 70/100
    560/560 [==============================] - 0s 27us/step - loss: 0.3218 - acc: 1.0000 - val_loss: 0.3179 - val_acc: 1.0000
    Epoch 71/100
    560/560 [==============================] - 0s 27us/step - loss: 0.3192 - acc: 1.0000 - val_loss: 0.3152 - val_acc: 1.0000
    Epoch 72/100
    560/560 [==============================] - 0s 21us/step - loss: 0.3165 - acc: 1.0000 - val_loss: 0.3124 - val_acc: 1.0000
    Epoch 73/100
    560/560 [==============================] - 0s 27us/step - loss: 0.3140 - acc: 1.0000 - val_loss: 0.3100 - val_acc: 1.0000
    Epoch 74/100
    560/560 [==============================] - 0s 25us/step - loss: 0.3114 - acc: 1.0000 - val_loss: 0.3074 - val_acc: 1.0000
    Epoch 75/100
    560/560 [==============================] - 0s 27us/step - loss: 0.3091 - acc: 1.0000 - val_loss: 0.3050 - val_acc: 1.0000
    Epoch 76/100
    560/560 [==============================] - 0s 25us/step - loss: 0.3065 - acc: 1.0000 - val_loss: 0.3027 - val_acc: 1.0000
    Epoch 77/100
    560/560 [==============================] - 0s 25us/step - loss: 0.3040 - acc: 1.0000 - val_loss: 0.3004 - val_acc: 1.0000
    Epoch 78/100
    560/560 [==============================] - 0s 27us/step - loss: 0.3016 - acc: 1.0000 - val_loss: 0.2982 - val_acc: 1.0000
    Epoch 79/100
    560/560 [==============================] - 0s 25us/step - loss: 0.2993 - acc: 1.0000 - val_loss: 0.2959 - val_acc: 1.0000
    Epoch 80/100
    560/560 [==============================] - 0s 25us/step - loss: 0.2970 - acc: 1.0000 - val_loss: 0.2937 - val_acc: 1.0000
    Epoch 81/100
    560/560 [==============================] - 0s 25us/step - loss: 0.2948 - acc: 1.0000 - val_loss: 0.2916 - val_acc: 1.0000
    Epoch 82/100
    560/560 [==============================] - 0s 18us/step - loss: 0.2926 - acc: 1.0000 - val_loss: 0.2891 - val_acc: 1.0000
    Epoch 83/100
    560/560 [==============================] - 0s 25us/step - loss: 0.2904 - acc: 1.0000 - val_loss: 0.2869 - val_acc: 1.0000
    Epoch 84/100
    560/560 [==============================] - 0s 20us/step - loss: 0.2883 - acc: 1.0000 - val_loss: 0.2847 - val_acc: 1.0000
    Epoch 85/100
    560/560 [==============================] - 0s 21us/step - loss: 0.2861 - acc: 1.0000 - val_loss: 0.2827 - val_acc: 1.0000
    Epoch 86/100
    560/560 [==============================] - 0s 27us/step - loss: 0.2840 - acc: 1.0000 - val_loss: 0.2806 - val_acc: 1.0000
    Epoch 87/100
    560/560 [==============================] - 0s 20us/step - loss: 0.2820 - acc: 1.0000 - val_loss: 0.2786 - val_acc: 1.0000
    Epoch 88/100
    560/560 [==============================] - 0s 23us/step - loss: 0.2799 - acc: 1.0000 - val_loss: 0.2766 - val_acc: 1.0000
    Epoch 89/100
    560/560 [==============================] - 0s 25us/step - loss: 0.2780 - acc: 1.0000 - val_loss: 0.2747 - val_acc: 1.0000
    Epoch 90/100
    560/560 [==============================] - 0s 23us/step - loss: 0.2760 - acc: 1.0000 - val_loss: 0.2727 - val_acc: 1.0000
    Epoch 91/100
    560/560 [==============================] - 0s 25us/step - loss: 0.2741 - acc: 1.0000 - val_loss: 0.2709 - val_acc: 1.0000
    Epoch 92/100
    560/560 [==============================] - 0s 20us/step - loss: 0.2722 - acc: 1.0000 - val_loss: 0.2688 - val_acc: 1.0000
    Epoch 93/100
    560/560 [==============================] - 0s 21us/step - loss: 0.2703 - acc: 1.0000 - val_loss: 0.2670 - val_acc: 1.0000
    Epoch 94/100
    560/560 [==============================] - 0s 23us/step - loss: 0.2684 - acc: 1.0000 - val_loss: 0.2652 - val_acc: 1.0000
    Epoch 95/100
    560/560 [==============================] - 0s 20us/step - loss: 0.2667 - acc: 1.0000 - val_loss: 0.2635 - val_acc: 1.0000
    Epoch 96/100
    560/560 [==============================] - 0s 25us/step - loss: 0.2648 - acc: 1.0000 - val_loss: 0.2619 - val_acc: 1.0000
    Epoch 97/100
    560/560 [==============================] - 0s 18us/step - loss: 0.2630 - acc: 1.0000 - val_loss: 0.2602 - val_acc: 1.0000
    Epoch 98/100
    560/560 [==============================] - 0s 23us/step - loss: 0.2613 - acc: 1.0000 - val_loss: 0.2586 - val_acc: 1.0000
    Epoch 99/100
    560/560 [==============================] - 0s 20us/step - loss: 0.2595 - acc: 1.0000 - val_loss: 0.2569 - val_acc: 1.0000
    Epoch 100/100
    560/560 [==============================] - 0s 20us/step - loss: 0.2578 - acc: 1.0000 - val_loss: 0.2552 - val_acc: 1.0000
    


```python
model.get_layer(index = 0).get_weights()
```




    [array([[2.1415565],
            [2.0179265]], dtype=float32), array([-3.0638707], dtype=float32)]



## 3.3 Example: XOR problem with 1-layer Net


```python
X = np.random.binomial(1, 0.5, size=(1000, 2))
y = (X[:, 0] + X[:, 1]) % 2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
pd.DataFrame([X[:,0], X[:,1], y])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>990</th>
      <th>991</th>
      <th>992</th>
      <th>993</th>
      <th>994</th>
      <th>995</th>
      <th>996</th>
      <th>997</th>
      <th>998</th>
      <th>999</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 1000 columns</p>
</div>




```python
OPTIMIZER = SGD(lr=0.1)

model = Sequential()
model.add(Dense(1, input_shape = (2,)))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=100, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
```

    Train on 560 samples, validate on 140 samples
    Epoch 1/100
    560/560 [==============================] - 1s 2ms/step - loss: 0.8277 - acc: 0.7321 - val_loss: 0.7801 - val_acc: 0.7571
    Epoch 2/100
    560/560 [==============================] - 0s 48us/step - loss: 0.8001 - acc: 0.7321 - val_loss: 0.7598 - val_acc: 0.7571
    Epoch 3/100
    560/560 [==============================] - 0s 41us/step - loss: 0.7807 - acc: 0.5982 - val_loss: 0.7452 - val_acc: 0.5143
    Epoch 4/100
    560/560 [==============================] - 0s 45us/step - loss: 0.7665 - acc: 0.4607 - val_loss: 0.7336 - val_acc: 0.5143
    Epoch 5/100
    560/560 [==============================] - 0s 59us/step - loss: 0.7549 - acc: 0.4607 - val_loss: 0.7255 - val_acc: 0.5143
    Epoch 6/100
    560/560 [==============================] - 0s 46us/step - loss: 0.7468 - acc: 0.4607 - val_loss: 0.7194 - val_acc: 0.5143
    Epoch 7/100
    560/560 [==============================] - 0s 64us/step - loss: 0.7405 - acc: 0.4607 - val_loss: 0.7155 - val_acc: 0.5143
    Epoch 8/100
    560/560 [==============================] - 0s 41us/step - loss: 0.7360 - acc: 0.4607 - val_loss: 0.7119 - val_acc: 0.5143
    Epoch 9/100
    560/560 [==============================] - 0s 38us/step - loss: 0.7320 - acc: 0.4607 - val_loss: 0.7097 - val_acc: 0.5143
    Epoch 10/100
    560/560 [==============================] - 0s 38us/step - loss: 0.7294 - acc: 0.4607 - val_loss: 0.7074 - val_acc: 0.5143
    Epoch 11/100
    560/560 [==============================] - 0s 50us/step - loss: 0.7265 - acc: 0.4607 - val_loss: 0.7057 - val_acc: 0.5143
    Epoch 12/100
    560/560 [==============================] - 0s 45us/step - loss: 0.7243 - acc: 0.4607 - val_loss: 0.7044 - val_acc: 0.5143
    Epoch 13/100
    560/560 [==============================] - 0s 43us/step - loss: 0.7223 - acc: 0.4607 - val_loss: 0.7029 - val_acc: 0.5143
    Epoch 14/100
    560/560 [==============================] - 0s 38us/step - loss: 0.7202 - acc: 0.4607 - val_loss: 0.7021 - val_acc: 0.5143
    Epoch 15/100
    560/560 [==============================] - 0s 39us/step - loss: 0.7192 - acc: 0.4607 - val_loss: 0.7010 - val_acc: 0.5143
    Epoch 16/100
    560/560 [==============================] - 0s 63us/step - loss: 0.7174 - acc: 0.4607 - val_loss: 0.7000 - val_acc: 0.5143
    Epoch 17/100
    560/560 [==============================] - 0s 34us/step - loss: 0.7159 - acc: 0.4607 - val_loss: 0.6993 - val_acc: 0.5143
    Epoch 18/100
    560/560 [==============================] - 0s 61us/step - loss: 0.7145 - acc: 0.4607 - val_loss: 0.6987 - val_acc: 0.5143
    Epoch 19/100
    560/560 [==============================] - 0s 38us/step - loss: 0.7133 - acc: 0.4607 - val_loss: 0.6980 - val_acc: 0.5143
    Epoch 20/100
    560/560 [==============================] - 0s 36us/step - loss: 0.7122 - acc: 0.4607 - val_loss: 0.6975 - val_acc: 0.5143
    Epoch 21/100
    560/560 [==============================] - 0s 63us/step - loss: 0.7111 - acc: 0.4607 - val_loss: 0.6971 - val_acc: 0.5143
    Epoch 22/100
    560/560 [==============================] - 0s 36us/step - loss: 0.7099 - acc: 0.4607 - val_loss: 0.6966 - val_acc: 0.5143
    Epoch 23/100
    560/560 [==============================] - 0s 48us/step - loss: 0.7091 - acc: 0.4607 - val_loss: 0.6963 - val_acc: 0.5143
    Epoch 24/100
    560/560 [==============================] - 0s 45us/step - loss: 0.7083 - acc: 0.4607 - val_loss: 0.6958 - val_acc: 0.5143
    Epoch 25/100
    560/560 [==============================] - 0s 50us/step - loss: 0.7073 - acc: 0.4607 - val_loss: 0.6955 - val_acc: 0.5143
    Epoch 26/100
    560/560 [==============================] - 0s 73us/step - loss: 0.7064 - acc: 0.4607 - val_loss: 0.6953 - val_acc: 0.5143
    Epoch 27/100
    560/560 [==============================] - 0s 38us/step - loss: 0.7058 - acc: 0.4607 - val_loss: 0.6950 - val_acc: 0.5143
    Epoch 28/100
    560/560 [==============================] - 0s 57us/step - loss: 0.7054 - acc: 0.4607 - val_loss: 0.6947 - val_acc: 0.5143
    Epoch 29/100
    560/560 [==============================] - 0s 29us/step - loss: 0.7043 - acc: 0.4607 - val_loss: 0.6943 - val_acc: 0.5143
    Epoch 30/100
    560/560 [==============================] - 0s 27us/step - loss: 0.7035 - acc: 0.4607 - val_loss: 0.6942 - val_acc: 0.5143
    Epoch 31/100
    560/560 [==============================] - 0s 39us/step - loss: 0.7029 - acc: 0.4607 - val_loss: 0.6940 - val_acc: 0.5143
    Epoch 32/100
    560/560 [==============================] - 0s 27us/step - loss: 0.7022 - acc: 0.4607 - val_loss: 0.6937 - val_acc: 0.5143
    Epoch 33/100
    560/560 [==============================] - 0s 32us/step - loss: 0.7015 - acc: 0.4607 - val_loss: 0.6936 - val_acc: 0.5143
    Epoch 34/100
    560/560 [==============================] - 0s 38us/step - loss: 0.7010 - acc: 0.4607 - val_loss: 0.6935 - val_acc: 0.5143
    Epoch 35/100
    560/560 [==============================] - 0s 27us/step - loss: 0.7005 - acc: 0.4607 - val_loss: 0.6933 - val_acc: 0.5143
    Epoch 36/100
    560/560 [==============================] - 0s 30us/step - loss: 0.7001 - acc: 0.4607 - val_loss: 0.6933 - val_acc: 0.5143
    Epoch 37/100
    560/560 [==============================] - 0s 32us/step - loss: 0.6999 - acc: 0.4607 - val_loss: 0.6932 - val_acc: 0.5143
    Epoch 38/100
    560/560 [==============================] - 0s 36us/step - loss: 0.6993 - acc: 0.4607 - val_loss: 0.6931 - val_acc: 0.5143
    Epoch 39/100
    560/560 [==============================] - 0s 27us/step - loss: 0.6989 - acc: 0.4607 - val_loss: 0.6930 - val_acc: 0.5143
    Epoch 40/100
    560/560 [==============================] - 0s 27us/step - loss: 0.6983 - acc: 0.4607 - val_loss: 0.6929 - val_acc: 0.5143
    Epoch 41/100
    560/560 [==============================] - 0s 43us/step - loss: 0.6979 - acc: 0.4607 - val_loss: 0.6929 - val_acc: 0.5143
    Epoch 42/100
    560/560 [==============================] - 0s 29us/step - loss: 0.6975 - acc: 0.4607 - val_loss: 0.6928 - val_acc: 0.5143
    Epoch 43/100
    560/560 [==============================] - 0s 25us/step - loss: 0.6970 - acc: 0.4607 - val_loss: 0.6928 - val_acc: 0.5143
    Epoch 44/100
    560/560 [==============================] - 0s 30us/step - loss: 0.6967 - acc: 0.4607 - val_loss: 0.6928 - val_acc: 0.5143
    Epoch 45/100
    560/560 [==============================] - 0s 38us/step - loss: 0.6965 - acc: 0.4607 - val_loss: 0.6928 - val_acc: 0.5143
    Epoch 46/100
    560/560 [==============================] - 0s 27us/step - loss: 0.6963 - acc: 0.4607 - val_loss: 0.6928 - val_acc: 0.5143
    Epoch 47/100
    560/560 [==============================] - 0s 25us/step - loss: 0.6959 - acc: 0.4607 - val_loss: 0.6928 - val_acc: 0.5143
    Epoch 48/100
    560/560 [==============================] - 0s 25us/step - loss: 0.6956 - acc: 0.4607 - val_loss: 0.6928 - val_acc: 0.5143
    Epoch 49/100
    560/560 [==============================] - 0s 32us/step - loss: 0.6953 - acc: 0.4607 - val_loss: 0.6928 - val_acc: 0.5143
    Epoch 50/100
    560/560 [==============================] - 0s 41us/step - loss: 0.6951 - acc: 0.4607 - val_loss: 0.6928 - val_acc: 0.5143
    Epoch 51/100
    560/560 [==============================] - 0s 32us/step - loss: 0.6951 - acc: 0.5089 - val_loss: 0.6928 - val_acc: 0.5143
    Epoch 52/100
    560/560 [==============================] - 0s 29us/step - loss: 0.6946 - acc: 0.4607 - val_loss: 0.6929 - val_acc: 0.5143
    Epoch 53/100
    560/560 [==============================] - 0s 27us/step - loss: 0.6944 - acc: 0.4607 - val_loss: 0.6929 - val_acc: 0.5143
    Epoch 54/100
    560/560 [==============================] - 0s 29us/step - loss: 0.6942 - acc: 0.4607 - val_loss: 0.6929 - val_acc: 0.5143
    Epoch 55/100
    560/560 [==============================] - 0s 34us/step - loss: 0.6941 - acc: 0.5946 - val_loss: 0.6930 - val_acc: 0.5143
    Epoch 56/100
    560/560 [==============================] - 0s 29us/step - loss: 0.6940 - acc: 0.5393 - val_loss: 0.6930 - val_acc: 0.5143
    Epoch 57/100
    560/560 [==============================] - 0s 27us/step - loss: 0.6937 - acc: 0.6661 - val_loss: 0.6931 - val_acc: 0.7571
    Epoch 58/100
    560/560 [==============================] - 0s 27us/step - loss: 0.6935 - acc: 0.5929 - val_loss: 0.6931 - val_acc: 0.5000
    Epoch 59/100
    560/560 [==============================] - 0s 30us/step - loss: 0.6934 - acc: 0.5161 - val_loss: 0.6932 - val_acc: 0.7429
    Epoch 60/100
    560/560 [==============================] - 0s 41us/step - loss: 0.6930 - acc: 0.5250 - val_loss: 0.6933 - val_acc: 0.4857
    Epoch 61/100
    560/560 [==============================] - 0s 30us/step - loss: 0.6932 - acc: 0.4357 - val_loss: 0.6933 - val_acc: 0.2429
    Epoch 62/100
    560/560 [==============================] - 0s 30us/step - loss: 0.6929 - acc: 0.3643 - val_loss: 0.6934 - val_acc: 0.5000
    Epoch 63/100
    560/560 [==============================] - 0s 29us/step - loss: 0.6925 - acc: 0.4071 - val_loss: 0.6934 - val_acc: 0.2429
    Epoch 64/100
    560/560 [==============================] - 0s 29us/step - loss: 0.6926 - acc: 0.3821 - val_loss: 0.6935 - val_acc: 0.2429
    Epoch 65/100
    560/560 [==============================] - 0s 29us/step - loss: 0.6925 - acc: 0.3982 - val_loss: 0.6935 - val_acc: 0.4857
    Epoch 66/100
    560/560 [==============================] - 0s 25us/step - loss: 0.6923 - acc: 0.5393 - val_loss: 0.6936 - val_acc: 0.4857
    Epoch 67/100
    560/560 [==============================] - 0s 48us/step - loss: 0.6922 - acc: 0.5393 - val_loss: 0.6937 - val_acc: 0.5000
    Epoch 68/100
    560/560 [==============================] - 0s 30us/step - loss: 0.6922 - acc: 0.5161 - val_loss: 0.6937 - val_acc: 0.4857
    Epoch 69/100
    560/560 [==============================] - 0s 25us/step - loss: 0.6921 - acc: 0.5143 - val_loss: 0.6938 - val_acc: 0.4857
    Epoch 70/100
    560/560 [==============================] - 0s 29us/step - loss: 0.6918 - acc: 0.5393 - val_loss: 0.6938 - val_acc: 0.4857
    Epoch 71/100
    560/560 [==============================] - 0s 29us/step - loss: 0.6917 - acc: 0.5393 - val_loss: 0.6939 - val_acc: 0.4857
    Epoch 72/100
    560/560 [==============================] - 0s 27us/step - loss: 0.6916 - acc: 0.5393 - val_loss: 0.6939 - val_acc: 0.4857
    Epoch 73/100
    560/560 [==============================] - 0s 45us/step - loss: 0.6917 - acc: 0.5393 - val_loss: 0.6940 - val_acc: 0.4857
    Epoch 74/100
    560/560 [==============================] - 0s 46us/step - loss: 0.6914 - acc: 0.5393 - val_loss: 0.6941 - val_acc: 0.4857
    Epoch 75/100
    560/560 [==============================] - 0s 38us/step - loss: 0.6916 - acc: 0.5393 - val_loss: 0.6942 - val_acc: 0.4857
    Epoch 76/100
    560/560 [==============================] - 0s 39us/step - loss: 0.6913 - acc: 0.5393 - val_loss: 0.6942 - val_acc: 0.4857
    Epoch 77/100
    560/560 [==============================] - 0s 41us/step - loss: 0.6912 - acc: 0.5393 - val_loss: 0.6943 - val_acc: 0.4857
    Epoch 78/100
    560/560 [==============================] - 0s 38us/step - loss: 0.6911 - acc: 0.5393 - val_loss: 0.6944 - val_acc: 0.4857
    Epoch 79/100
    560/560 [==============================] - 0s 29us/step - loss: 0.6911 - acc: 0.6000 - val_loss: 0.6945 - val_acc: 0.4857
    Epoch 80/100
    560/560 [==============================] - 0s 32us/step - loss: 0.6910 - acc: 0.5393 - val_loss: 0.6947 - val_acc: 0.4857
    Epoch 81/100
    560/560 [==============================] - 0s 36us/step - loss: 0.6908 - acc: 0.5875 - val_loss: 0.6948 - val_acc: 0.7429
    Epoch 82/100
    560/560 [==============================] - 0s 27us/step - loss: 0.6908 - acc: 0.7196 - val_loss: 0.6948 - val_acc: 0.4857
    Epoch 83/100
    560/560 [==============================] - 0s 43us/step - loss: 0.6908 - acc: 0.5393 - val_loss: 0.6949 - val_acc: 0.7429
    Epoch 84/100
    560/560 [==============================] - 0s 29us/step - loss: 0.6908 - acc: 0.6179 - val_loss: 0.6948 - val_acc: 0.4857
    Epoch 85/100
    560/560 [==============================] - 0s 29us/step - loss: 0.6909 - acc: 0.6500 - val_loss: 0.6949 - val_acc: 0.4857
    Epoch 86/100
    560/560 [==============================] - 0s 29us/step - loss: 0.6905 - acc: 0.5393 - val_loss: 0.6949 - val_acc: 0.4857
    Epoch 87/100
    560/560 [==============================] - 0s 27us/step - loss: 0.6908 - acc: 0.5393 - val_loss: 0.6950 - val_acc: 0.4857
    Epoch 88/100
    560/560 [==============================] - 0s 36us/step - loss: 0.6906 - acc: 0.5393 - val_loss: 0.6951 - val_acc: 0.4857
    Epoch 89/100
    560/560 [==============================] - 0s 27us/step - loss: 0.6904 - acc: 0.5393 - val_loss: 0.6952 - val_acc: 0.7429
    Epoch 90/100
    560/560 [==============================] - 0s 34us/step - loss: 0.6905 - acc: 0.6929 - val_loss: 0.6954 - val_acc: 0.7429
    Epoch 91/100
    560/560 [==============================] - 0s 27us/step - loss: 0.6905 - acc: 0.7839 - val_loss: 0.6955 - val_acc: 0.7429
    Epoch 92/100
    560/560 [==============================] - 0s 32us/step - loss: 0.6904 - acc: 0.7839 - val_loss: 0.6955 - val_acc: 0.7429
    Epoch 93/100
    560/560 [==============================] - 0s 30us/step - loss: 0.6905 - acc: 0.7286 - val_loss: 0.6954 - val_acc: 0.7429
    Epoch 94/100
    560/560 [==============================] - 0s 32us/step - loss: 0.6902 - acc: 0.6429 - val_loss: 0.6955 - val_acc: 0.7429
    Epoch 95/100
    560/560 [==============================] - 0s 30us/step - loss: 0.6904 - acc: 0.7018 - val_loss: 0.6957 - val_acc: 0.7429
    Epoch 96/100
    560/560 [==============================] - 0s 27us/step - loss: 0.6902 - acc: 0.7839 - val_loss: 0.6957 - val_acc: 0.7429
    Epoch 97/100
    560/560 [==============================] - 0s 36us/step - loss: 0.6904 - acc: 0.7839 - val_loss: 0.6956 - val_acc: 0.4857
    Epoch 98/100
    560/560 [==============================] - 0s 27us/step - loss: 0.6901 - acc: 0.5393 - val_loss: 0.6956 - val_acc: 0.4857
    Epoch 99/100
    560/560 [==============================] - 0s 43us/step - loss: 0.6900 - acc: 0.5964 - val_loss: 0.6957 - val_acc: 0.4857
    Epoch 100/100
    560/560 [==============================] - 0s 27us/step - loss: 0.6901 - acc: 0.5393 - val_loss: 0.6957 - val_acc: 0.4857
    

## 3.4 Example: XOR problem with 2-layer Net\

We implement a 2-layer Feedforward network.


```python
model = Sequential()
model.add(Dense(2, input_shape = (2,)))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_21 (Dense)             (None, 2)                 6         
    _________________________________________________________________
    activation_21 (Activation)   (None, 2)                 0         
    _________________________________________________________________
    dense_22 (Dense)             (None, 1)                 3         
    _________________________________________________________________
    activation_22 (Activation)   (None, 1)                 0         
    =================================================================
    Total params: 9
    Trainable params: 9
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model.compile(loss='binary_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=100, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
```

    Train on 560 samples, validate on 140 samples
    Epoch 1/100
    560/560 [==============================] - 1s 2ms/step - loss: 0.7036 - acc: 0.5161 - val_loss: 0.7012 - val_acc: 0.5000
    Epoch 2/100
    560/560 [==============================] - 0s 30us/step - loss: 0.7017 - acc: 0.5161 - val_loss: 0.6987 - val_acc: 0.5000
    Epoch 3/100
    560/560 [==============================] - 0s 32us/step - loss: 0.6999 - acc: 0.5161 - val_loss: 0.6974 - val_acc: 0.5000
    Epoch 4/100
    560/560 [==============================] - 0s 50us/step - loss: 0.6989 - acc: 0.5161 - val_loss: 0.6960 - val_acc: 0.5000
    Epoch 5/100
    560/560 [==============================] - 0s 45us/step - loss: 0.6981 - acc: 0.5161 - val_loss: 0.6955 - val_acc: 0.5000
    Epoch 6/100
    560/560 [==============================] - 0s 34us/step - loss: 0.6978 - acc: 0.5607 - val_loss: 0.6950 - val_acc: 0.5000
    Epoch 7/100
    560/560 [==============================] - 0s 36us/step - loss: 0.6975 - acc: 0.6768 - val_loss: 0.6946 - val_acc: 0.7571
    Epoch 8/100
    560/560 [==============================] - 0s 32us/step - loss: 0.6972 - acc: 0.7321 - val_loss: 0.6943 - val_acc: 0.7571
    Epoch 9/100
    560/560 [==============================] - 0s 41us/step - loss: 0.6971 - acc: 0.7321 - val_loss: 0.6943 - val_acc: 0.7571
    Epoch 10/100
    560/560 [==============================] - 0s 29us/step - loss: 0.6970 - acc: 0.7321 - val_loss: 0.6942 - val_acc: 0.7571
    Epoch 11/100
    560/560 [==============================] - 0s 32us/step - loss: 0.6969 - acc: 0.7321 - val_loss: 0.6940 - val_acc: 0.7571
    Epoch 12/100
    560/560 [==============================] - 0s 63us/step - loss: 0.6968 - acc: 0.7321 - val_loss: 0.6939 - val_acc: 0.7571
    Epoch 13/100
    560/560 [==============================] - 0s 27us/step - loss: 0.6968 - acc: 0.7321 - val_loss: 0.6938 - val_acc: 0.7571
    Epoch 14/100
    560/560 [==============================] - 0s 29us/step - loss: 0.6966 - acc: 0.7321 - val_loss: 0.6935 - val_acc: 0.7571
    Epoch 15/100
    560/560 [==============================] - 0s 27us/step - loss: 0.6966 - acc: 0.7321 - val_loss: 0.6934 - val_acc: 0.7571
    Epoch 16/100
    560/560 [==============================] - 0s 23us/step - loss: 0.6965 - acc: 0.7321 - val_loss: 0.6934 - val_acc: 0.7571
    Epoch 17/100
    560/560 [==============================] - 0s 30us/step - loss: 0.6966 - acc: 0.7321 - val_loss: 0.6933 - val_acc: 0.7571
    Epoch 18/100
    560/560 [==============================] - 0s 27us/step - loss: 0.6964 - acc: 0.7321 - val_loss: 0.6933 - val_acc: 0.7571
    Epoch 19/100
    560/560 [==============================] - 0s 23us/step - loss: 0.6964 - acc: 0.7321 - val_loss: 0.6933 - val_acc: 0.7571
    Epoch 20/100
    560/560 [==============================] - 0s 27us/step - loss: 0.6964 - acc: 0.7321 - val_loss: 0.6933 - val_acc: 0.7571
    Epoch 21/100
    560/560 [==============================] - 0s 25us/step - loss: 0.6964 - acc: 0.7321 - val_loss: 0.6933 - val_acc: 0.7571
    Epoch 22/100
    560/560 [==============================] - 0s 25us/step - loss: 0.6964 - acc: 0.7321 - val_loss: 0.6936 - val_acc: 0.7571
    Epoch 23/100
    560/560 [==============================] - 0s 23us/step - loss: 0.6965 - acc: 0.7321 - val_loss: 0.6935 - val_acc: 0.7571
    Epoch 24/100
    560/560 [==============================] - 0s 43us/step - loss: 0.6963 - acc: 0.7321 - val_loss: 0.6934 - val_acc: 0.7571
    Epoch 25/100
    560/560 [==============================] - 0s 27us/step - loss: 0.6963 - acc: 0.7321 - val_loss: 0.6934 - val_acc: 0.7571
    Epoch 26/100
    560/560 [==============================] - 0s 43us/step - loss: 0.6962 - acc: 0.7321 - val_loss: 0.6935 - val_acc: 0.7571
    Epoch 27/100
    560/560 [==============================] - 0s 29us/step - loss: 0.6962 - acc: 0.7321 - val_loss: 0.6933 - val_acc: 0.7571
    Epoch 28/100
    560/560 [==============================] - 0s 23us/step - loss: 0.6962 - acc: 0.7321 - val_loss: 0.6933 - val_acc: 0.7571
    Epoch 29/100
    560/560 [==============================] - 0s 25us/step - loss: 0.6961 - acc: 0.7321 - val_loss: 0.6932 - val_acc: 0.7571
    Epoch 30/100
    560/560 [==============================] - 0s 30us/step - loss: 0.6961 - acc: 0.7321 - val_loss: 0.6933 - val_acc: 0.7571
    Epoch 31/100
    560/560 [==============================] - 0s 21us/step - loss: 0.6963 - acc: 0.7321 - val_loss: 0.6934 - val_acc: 0.7571
    Epoch 32/100
    560/560 [==============================] - 0s 30us/step - loss: 0.6961 - acc: 0.7321 - val_loss: 0.6934 - val_acc: 0.7571
    Epoch 33/100
    560/560 [==============================] - 0s 21us/step - loss: 0.6961 - acc: 0.7321 - val_loss: 0.6935 - val_acc: 0.7571
    Epoch 34/100
    560/560 [==============================] - 0s 23us/step - loss: 0.6960 - acc: 0.7321 - val_loss: 0.6935 - val_acc: 0.7571
    Epoch 35/100
    560/560 [==============================] - 0s 36us/step - loss: 0.6960 - acc: 0.7321 - val_loss: 0.6934 - val_acc: 0.7571
    Epoch 36/100
    560/560 [==============================] - 0s 25us/step - loss: 0.6960 - acc: 0.7321 - val_loss: 0.6933 - val_acc: 0.7571
    Epoch 37/100
    560/560 [==============================] - 0s 21us/step - loss: 0.6960 - acc: 0.7321 - val_loss: 0.6932 - val_acc: 0.7571
    Epoch 38/100
    560/560 [==============================] - 0s 32us/step - loss: 0.6959 - acc: 0.7321 - val_loss: 0.6932 - val_acc: 0.7571
    Epoch 39/100
    560/560 [==============================] - 0s 27us/step - loss: 0.6959 - acc: 0.7321 - val_loss: 0.6932 - val_acc: 0.7571
    Epoch 40/100
    560/560 [==============================] - 0s 25us/step - loss: 0.6960 - acc: 0.7321 - val_loss: 0.6935 - val_acc: 0.7571
    Epoch 41/100
    560/560 [==============================] - 0s 23us/step - loss: 0.6960 - acc: 0.7321 - val_loss: 0.6935 - val_acc: 0.7571
    Epoch 42/100
    560/560 [==============================] - 0s 21us/step - loss: 0.6958 - acc: 0.7321 - val_loss: 0.6935 - val_acc: 0.7571
    Epoch 43/100
    560/560 [==============================] - 0s 36us/step - loss: 0.6961 - acc: 0.7321 - val_loss: 0.6934 - val_acc: 0.7571
    Epoch 44/100
    560/560 [==============================] - 0s 25us/step - loss: 0.6958 - acc: 0.7321 - val_loss: 0.6934 - val_acc: 0.7571
    Epoch 45/100
    560/560 [==============================] - 0s 27us/step - loss: 0.6957 - acc: 0.7321 - val_loss: 0.6934 - val_acc: 0.7571
    Epoch 46/100
    560/560 [==============================] - 0s 23us/step - loss: 0.6957 - acc: 0.7321 - val_loss: 0.6934 - val_acc: 0.7571
    Epoch 47/100
    560/560 [==============================] - 0s 32us/step - loss: 0.6956 - acc: 0.7321 - val_loss: 0.6932 - val_acc: 0.7571
    Epoch 48/100
    560/560 [==============================] - 0s 25us/step - loss: 0.6956 - acc: 0.7321 - val_loss: 0.6933 - val_acc: 0.7571
    Epoch 49/100
    560/560 [==============================] - 0s 21us/step - loss: 0.6955 - acc: 0.7321 - val_loss: 0.6932 - val_acc: 0.7571
    Epoch 50/100
    560/560 [==============================] - 0s 23us/step - loss: 0.6956 - acc: 0.7321 - val_loss: 0.6932 - val_acc: 0.7571
    Epoch 51/100
    560/560 [==============================] - 0s 38us/step - loss: 0.6956 - acc: 0.7321 - val_loss: 0.6933 - val_acc: 0.7571
    Epoch 52/100
    560/560 [==============================] - 0s 27us/step - loss: 0.6954 - acc: 0.7321 - val_loss: 0.6933 - val_acc: 0.7571
    Epoch 53/100
    560/560 [==============================] - 0s 25us/step - loss: 0.6955 - acc: 0.7321 - val_loss: 0.6932 - val_acc: 0.7571
    Epoch 54/100
    560/560 [==============================] - 0s 25us/step - loss: 0.6954 - acc: 0.7321 - val_loss: 0.6931 - val_acc: 0.7571
    Epoch 55/100
    560/560 [==============================] - 0s 30us/step - loss: 0.6953 - acc: 0.7321 - val_loss: 0.6931 - val_acc: 0.7571
    Epoch 56/100
    560/560 [==============================] - 0s 20us/step - loss: 0.6955 - acc: 0.7321 - val_loss: 0.6932 - val_acc: 0.7571
    Epoch 57/100
    560/560 [==============================] - 0s 20us/step - loss: 0.6953 - acc: 0.7321 - val_loss: 0.6931 - val_acc: 0.7571
    Epoch 58/100
    560/560 [==============================] - 0s 21us/step - loss: 0.6953 - acc: 0.7321 - val_loss: 0.6932 - val_acc: 0.7571
    Epoch 59/100
    560/560 [==============================] - 0s 29us/step - loss: 0.6956 - acc: 0.7321 - val_loss: 0.6932 - val_acc: 0.7571
    Epoch 60/100
    560/560 [==============================] - 0s 25us/step - loss: 0.6952 - acc: 0.7321 - val_loss: 0.6934 - val_acc: 0.7571
    Epoch 61/100
    560/560 [==============================] - 0s 25us/step - loss: 0.6952 - acc: 0.7321 - val_loss: 0.6933 - val_acc: 0.7571
    Epoch 62/100
    560/560 [==============================] - 0s 23us/step - loss: 0.6953 - acc: 0.7321 - val_loss: 0.6931 - val_acc: 0.7571
    Epoch 63/100
    560/560 [==============================] - 0s 20us/step - loss: 0.6951 - acc: 0.7321 - val_loss: 0.6931 - val_acc: 0.7571
    Epoch 64/100
    560/560 [==============================] - 0s 21us/step - loss: 0.6951 - acc: 0.7321 - val_loss: 0.6932 - val_acc: 0.7571
    Epoch 65/100
    560/560 [==============================] - 0s 23us/step - loss: 0.6952 - acc: 0.7321 - val_loss: 0.6934 - val_acc: 0.7571
    Epoch 66/100
    560/560 [==============================] - 0s 27us/step - loss: 0.6953 - acc: 0.7321 - val_loss: 0.6934 - val_acc: 0.7571
    Epoch 67/100
    560/560 [==============================] - 0s 20us/step - loss: 0.6953 - acc: 0.7321 - val_loss: 0.6936 - val_acc: 0.7571
    Epoch 68/100
    560/560 [==============================] - 0s 20us/step - loss: 0.6951 - acc: 0.7321 - val_loss: 0.6935 - val_acc: 0.7571
    Epoch 69/100
    560/560 [==============================] - 0s 25us/step - loss: 0.6954 - acc: 0.6839 - val_loss: 0.6931 - val_acc: 0.7571
    Epoch 70/100
    560/560 [==============================] - 0s 27us/step - loss: 0.6953 - acc: 0.7321 - val_loss: 0.6930 - val_acc: 0.7571
    Epoch 71/100
    560/560 [==============================] - 0s 27us/step - loss: 0.6950 - acc: 0.7321 - val_loss: 0.6931 - val_acc: 0.7571
    Epoch 72/100
    560/560 [==============================] - 0s 21us/step - loss: 0.6949 - acc: 0.7321 - val_loss: 0.6930 - val_acc: 0.7571
    Epoch 73/100
    560/560 [==============================] - 0s 23us/step - loss: 0.6950 - acc: 0.7321 - val_loss: 0.6931 - val_acc: 0.7571
    Epoch 74/100
    560/560 [==============================] - 0s 29us/step - loss: 0.6949 - acc: 0.7321 - val_loss: 0.6931 - val_acc: 0.7571
    Epoch 75/100
    560/560 [==============================] - 0s 27us/step - loss: 0.6949 - acc: 0.7321 - val_loss: 0.6932 - val_acc: 0.7571
    Epoch 76/100
    560/560 [==============================] - 0s 23us/step - loss: 0.6951 - acc: 0.7321 - val_loss: 0.6931 - val_acc: 0.7571
    Epoch 77/100
    560/560 [==============================] - 0s 27us/step - loss: 0.6948 - acc: 0.7321 - val_loss: 0.6932 - val_acc: 0.7571
    Epoch 78/100
    560/560 [==============================] - 0s 25us/step - loss: 0.6948 - acc: 0.7321 - val_loss: 0.6933 - val_acc: 0.7571
    Epoch 79/100
    560/560 [==============================] - 0s 25us/step - loss: 0.6947 - acc: 0.7321 - val_loss: 0.6932 - val_acc: 0.7571
    Epoch 80/100
    560/560 [==============================] - 0s 25us/step - loss: 0.6950 - acc: 0.7321 - val_loss: 0.6933 - val_acc: 0.7571
    Epoch 81/100
    560/560 [==============================] - 0s 23us/step - loss: 0.6949 - acc: 0.7321 - val_loss: 0.6932 - val_acc: 0.7571
    Epoch 82/100
    560/560 [==============================] - 0s 25us/step - loss: 0.6947 - acc: 0.7321 - val_loss: 0.6932 - val_acc: 0.7571
    Epoch 83/100
    560/560 [==============================] - 0s 23us/step - loss: 0.6947 - acc: 0.7321 - val_loss: 0.6933 - val_acc: 0.7571
    Epoch 84/100
    560/560 [==============================] - 0s 21us/step - loss: 0.6947 - acc: 0.7321 - val_loss: 0.6932 - val_acc: 0.7571
    Epoch 85/100
    560/560 [==============================] - 0s 29us/step - loss: 0.6946 - acc: 0.7321 - val_loss: 0.6933 - val_acc: 0.7571
    Epoch 86/100
    560/560 [==============================] - 0s 23us/step - loss: 0.6946 - acc: 0.7321 - val_loss: 0.6933 - val_acc: 0.7571
    Epoch 87/100
    560/560 [==============================] - 0s 30us/step - loss: 0.6949 - acc: 0.7321 - val_loss: 0.6933 - val_acc: 0.7571
    Epoch 88/100
    560/560 [==============================] - 0s 21us/step - loss: 0.6946 - acc: 0.7321 - val_loss: 0.6933 - val_acc: 0.7571
    Epoch 89/100
    560/560 [==============================] - 0s 27us/step - loss: 0.6946 - acc: 0.7321 - val_loss: 0.6931 - val_acc: 0.7571
    Epoch 90/100
    560/560 [==============================] - 0s 23us/step - loss: 0.6946 - acc: 0.7321 - val_loss: 0.6930 - val_acc: 0.7571
    Epoch 91/100
    560/560 [==============================] - 0s 27us/step - loss: 0.6944 - acc: 0.7321 - val_loss: 0.6930 - val_acc: 0.7571
    Epoch 92/100
    560/560 [==============================] - 0s 27us/step - loss: 0.6945 - acc: 0.7321 - val_loss: 0.6931 - val_acc: 0.7571
    Epoch 93/100
    560/560 [==============================] - 0s 27us/step - loss: 0.6945 - acc: 0.7321 - val_loss: 0.6934 - val_acc: 0.7571
    Epoch 94/100
    560/560 [==============================] - 0s 27us/step - loss: 0.6947 - acc: 0.7071 - val_loss: 0.6932 - val_acc: 0.7571
    Epoch 95/100
    560/560 [==============================] - 0s 25us/step - loss: 0.6946 - acc: 0.6732 - val_loss: 0.6933 - val_acc: 0.7571
    Epoch 96/100
    560/560 [==============================] - 0s 25us/step - loss: 0.6945 - acc: 0.7321 - val_loss: 0.6935 - val_acc: 0.5000
    Epoch 97/100
    560/560 [==============================] - 0s 23us/step - loss: 0.6947 - acc: 0.6232 - val_loss: 0.6935 - val_acc: 0.5000
    Epoch 98/100
    560/560 [==============================] - 0s 21us/step - loss: 0.6945 - acc: 0.5161 - val_loss: 0.6933 - val_acc: 0.7571
    Epoch 99/100
    560/560 [==============================] - 0s 29us/step - loss: 0.6943 - acc: 0.7321 - val_loss: 0.6932 - val_acc: 0.7571
    Epoch 100/100
    560/560 [==============================] - 0s 21us/step - loss: 0.6944 - acc: 0.7321 - val_loss: 0.6931 - val_acc: 0.7571
    

We can see that the algorithm stucks at some local extremum. In fact, if we initialize with better weights and bias, we can get a better performance.


```python
model = Sequential()
model.add(Dense(2, input_shape = (2,), kernel_initializer=RandomUniform(-5, 5), bias_initializer=RandomUniform(-5, 5)))
model.add(Activation('sigmoid'))
model.add(Dense(1, kernel_initializer=RandomUniform(-5, 5), bias_initializer=RandomUniform(-5, 5)))
model.add(Activation('sigmoid'))
model.get_layer(index=0).set_weights([np.array([[10, -10], [10, -10]]), np.array([-5, 15])])
model.get_layer(index=2).set_weights([np.array([[10], [10]]), np.array([-15])])
```


```python
model.compile(loss='binary_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=100, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
```

    Train on 560 samples, validate on 140 samples
    Epoch 1/100
    560/560 [==============================] - 1s 2ms/step - loss: 0.0074 - acc: 1.0000 - val_loss: 0.0074 - val_acc: 1.0000
    Epoch 2/100
    560/560 [==============================] - 0s 38us/step - loss: 0.0074 - acc: 1.0000 - val_loss: 0.0074 - val_acc: 1.0000
    Epoch 3/100
    560/560 [==============================] - 0s 38us/step - loss: 0.0074 - acc: 1.0000 - val_loss: 0.0074 - val_acc: 1.0000
    Epoch 4/100
    560/560 [==============================] - 0s 38us/step - loss: 0.0074 - acc: 1.0000 - val_loss: 0.0074 - val_acc: 1.0000
    Epoch 5/100
    560/560 [==============================] - 0s 30us/step - loss: 0.0074 - acc: 1.0000 - val_loss: 0.0074 - val_acc: 1.0000
    Epoch 6/100
    560/560 [==============================] - 0s 38us/step - loss: 0.0074 - acc: 1.0000 - val_loss: 0.0074 - val_acc: 1.0000
    Epoch 7/100
    560/560 [==============================] - 0s 30us/step - loss: 0.0074 - acc: 1.0000 - val_loss: 0.0074 - val_acc: 1.0000
    Epoch 8/100
    560/560 [==============================] - 0s 39us/step - loss: 0.0074 - acc: 1.0000 - val_loss: 0.0074 - val_acc: 1.0000
    Epoch 9/100
    560/560 [==============================] - 0s 38us/step - loss: 0.0074 - acc: 1.0000 - val_loss: 0.0074 - val_acc: 1.0000
    Epoch 10/100
    560/560 [==============================] - 0s 68us/step - loss: 0.0074 - acc: 1.0000 - val_loss: 0.0074 - val_acc: 1.0000
    Epoch 11/100
    560/560 [==============================] - 0s 38us/step - loss: 0.0074 - acc: 1.0000 - val_loss: 0.0074 - val_acc: 1.0000
    Epoch 12/100
    560/560 [==============================] - 0s 25us/step - loss: 0.0074 - acc: 1.0000 - val_loss: 0.0074 - val_acc: 1.0000
    Epoch 13/100
    560/560 [==============================] - 0s 36us/step - loss: 0.0074 - acc: 1.0000 - val_loss: 0.0074 - val_acc: 1.0000
    Epoch 14/100
    560/560 [==============================] - 0s 23us/step - loss: 0.0074 - acc: 1.0000 - val_loss: 0.0074 - val_acc: 1.0000
    Epoch 15/100
    560/560 [==============================] - 0s 23us/step - loss: 0.0074 - acc: 1.0000 - val_loss: 0.0074 - val_acc: 1.0000
    Epoch 16/100
    560/560 [==============================] - 0s 29us/step - loss: 0.0074 - acc: 1.0000 - val_loss: 0.0074 - val_acc: 1.0000
    Epoch 17/100
    560/560 [==============================] - 0s 29us/step - loss: 0.0074 - acc: 1.0000 - val_loss: 0.0074 - val_acc: 1.0000
    Epoch 18/100
    560/560 [==============================] - 0s 20us/step - loss: 0.0074 - acc: 1.0000 - val_loss: 0.0074 - val_acc: 1.0000
    Epoch 19/100
    560/560 [==============================] - 0s 38us/step - loss: 0.0074 - acc: 1.0000 - val_loss: 0.0074 - val_acc: 1.0000
    Epoch 20/100
    560/560 [==============================] - 0s 23us/step - loss: 0.0073 - acc: 1.0000 - val_loss: 0.0073 - val_acc: 1.0000
    Epoch 21/100
    560/560 [==============================] - 0s 21us/step - loss: 0.0073 - acc: 1.0000 - val_loss: 0.0073 - val_acc: 1.0000
    Epoch 22/100
    560/560 [==============================] - 0s 34us/step - loss: 0.0073 - acc: 1.0000 - val_loss: 0.0073 - val_acc: 1.0000
    Epoch 23/100
    560/560 [==============================] - 0s 23us/step - loss: 0.0073 - acc: 1.0000 - val_loss: 0.0073 - val_acc: 1.0000
    Epoch 24/100
    560/560 [==============================] - 0s 21us/step - loss: 0.0073 - acc: 1.0000 - val_loss: 0.0073 - val_acc: 1.0000
    Epoch 25/100
    560/560 [==============================] - 0s 29us/step - loss: 0.0073 - acc: 1.0000 - val_loss: 0.0073 - val_acc: 1.0000
    Epoch 26/100
    560/560 [==============================] - 0s 25us/step - loss: 0.0073 - acc: 1.0000 - val_loss: 0.0073 - val_acc: 1.0000
    Epoch 27/100
    560/560 [==============================] - 0s 21us/step - loss: 0.0073 - acc: 1.0000 - val_loss: 0.0073 - val_acc: 1.0000
    Epoch 28/100
    560/560 [==============================] - 0s 36us/step - loss: 0.0073 - acc: 1.0000 - val_loss: 0.0073 - val_acc: 1.0000
    Epoch 29/100
    560/560 [==============================] - 0s 23us/step - loss: 0.0073 - acc: 1.0000 - val_loss: 0.0073 - val_acc: 1.0000
    Epoch 30/100
    560/560 [==============================] - 0s 21us/step - loss: 0.0073 - acc: 1.0000 - val_loss: 0.0073 - val_acc: 1.0000
    Epoch 31/100
    560/560 [==============================] - 0s 43us/step - loss: 0.0073 - acc: 1.0000 - val_loss: 0.0073 - val_acc: 1.0000
    Epoch 32/100
    560/560 [==============================] - 0s 21us/step - loss: 0.0073 - acc: 1.0000 - val_loss: 0.0073 - val_acc: 1.0000
    Epoch 33/100
    560/560 [==============================] - 0s 21us/step - loss: 0.0073 - acc: 1.0000 - val_loss: 0.0073 - val_acc: 1.0000
    Epoch 34/100
    560/560 [==============================] - 0s 36us/step - loss: 0.0073 - acc: 1.0000 - val_loss: 0.0073 - val_acc: 1.0000
    Epoch 35/100
    560/560 [==============================] - 0s 27us/step - loss: 0.0073 - acc: 1.0000 - val_loss: 0.0073 - val_acc: 1.0000
    Epoch 36/100
    560/560 [==============================] - 0s 20us/step - loss: 0.0073 - acc: 1.0000 - val_loss: 0.0073 - val_acc: 1.0000
    Epoch 37/100
    560/560 [==============================] - 0s 36us/step - loss: 0.0073 - acc: 1.0000 - val_loss: 0.0073 - val_acc: 1.0000
    Epoch 38/100
    560/560 [==============================] - 0s 27us/step - loss: 0.0073 - acc: 1.0000 - val_loss: 0.0073 - val_acc: 1.0000
    Epoch 39/100
    560/560 [==============================] - 0s 21us/step - loss: 0.0073 - acc: 1.0000 - val_loss: 0.0073 - val_acc: 1.0000
    Epoch 40/100
    560/560 [==============================] - 0s 34us/step - loss: 0.0073 - acc: 1.0000 - val_loss: 0.0073 - val_acc: 1.0000
    Epoch 41/100
    560/560 [==============================] - 0s 27us/step - loss: 0.0073 - acc: 1.0000 - val_loss: 0.0073 - val_acc: 1.0000
    Epoch 42/100
    560/560 [==============================] - 0s 21us/step - loss: 0.0073 - acc: 1.0000 - val_loss: 0.0073 - val_acc: 1.0000
    Epoch 43/100
    560/560 [==============================] - 0s 20us/step - loss: 0.0073 - acc: 1.0000 - val_loss: 0.0073 - val_acc: 1.0000
    Epoch 44/100
    560/560 [==============================] - 0s 43us/step - loss: 0.0073 - acc: 1.0000 - val_loss: 0.0073 - val_acc: 1.0000
    Epoch 45/100
    560/560 [==============================] - 0s 23us/step - loss: 0.0073 - acc: 1.0000 - val_loss: 0.0073 - val_acc: 1.0000
    Epoch 46/100
    560/560 [==============================] - 0s 23us/step - loss: 0.0073 - acc: 1.0000 - val_loss: 0.0073 - val_acc: 1.0000
    Epoch 47/100
    560/560 [==============================] - 0s 20us/step - loss: 0.0073 - acc: 1.0000 - val_loss: 0.0073 - val_acc: 1.0000
    Epoch 48/100
    560/560 [==============================] - 0s 27us/step - loss: 0.0073 - acc: 1.0000 - val_loss: 0.0073 - val_acc: 1.0000
    Epoch 49/100
    560/560 [==============================] - 0s 20us/step - loss: 0.0072 - acc: 1.0000 - val_loss: 0.0073 - val_acc: 1.0000
    Epoch 50/100
    560/560 [==============================] - 0s 20us/step - loss: 0.0072 - acc: 1.0000 - val_loss: 0.0073 - val_acc: 1.0000
    Epoch 51/100
    560/560 [==============================] - 0s 32us/step - loss: 0.0072 - acc: 1.0000 - val_loss: 0.0073 - val_acc: 1.0000
    Epoch 52/100
    560/560 [==============================] - 0s 25us/step - loss: 0.0072 - acc: 1.0000 - val_loss: 0.0073 - val_acc: 1.0000
    Epoch 53/100
    560/560 [==============================] - 0s 21us/step - loss: 0.0072 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 54/100
    560/560 [==============================] - 0s 21us/step - loss: 0.0072 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 55/100
    560/560 [==============================] - 0s 39us/step - loss: 0.0072 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 56/100
    560/560 [==============================] - 0s 23us/step - loss: 0.0072 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 57/100
    560/560 [==============================] - 0s 23us/step - loss: 0.0072 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 58/100
    560/560 [==============================] - 0s 20us/step - loss: 0.0072 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 59/100
    560/560 [==============================] - 0s 20us/step - loss: 0.0072 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 60/100
    560/560 [==============================] - 0s 32us/step - loss: 0.0072 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 61/100
    560/560 [==============================] - 0s 25us/step - loss: 0.0072 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 62/100
    560/560 [==============================] - 0s 20us/step - loss: 0.0072 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 63/100
    560/560 [==============================] - 0s 20us/step - loss: 0.0072 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 64/100
    560/560 [==============================] - 0s 21us/step - loss: 0.0072 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 65/100
    560/560 [==============================] - 0s 46us/step - loss: 0.0072 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 66/100
    560/560 [==============================] - 0s 23us/step - loss: 0.0072 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 67/100
    560/560 [==============================] - 0s 21us/step - loss: 0.0072 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 68/100
    560/560 [==============================] - 0s 34us/step - loss: 0.0072 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 69/100
    560/560 [==============================] - 0s 21us/step - loss: 0.0072 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 70/100
    560/560 [==============================] - 0s 32us/step - loss: 0.0072 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 71/100
    560/560 [==============================] - 0s 20us/step - loss: 0.0072 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 72/100
    560/560 [==============================] - 0s 30us/step - loss: 0.0072 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 73/100
    560/560 [==============================] - 0s 30us/step - loss: 0.0072 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 74/100
    560/560 [==============================] - 0s 23us/step - loss: 0.0072 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 75/100
    560/560 [==============================] - 0s 27us/step - loss: 0.0072 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 76/100
    560/560 [==============================] - 0s 21us/step - loss: 0.0072 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 77/100
    560/560 [==============================] - 0s 21us/step - loss: 0.0072 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 78/100
    560/560 [==============================] - 0s 25us/step - loss: 0.0072 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 79/100
    560/560 [==============================] - 0s 21us/step - loss: 0.0072 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 80/100
    560/560 [==============================] - 0s 25us/step - loss: 0.0072 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 81/100
    560/560 [==============================] - 0s 30us/step - loss: 0.0072 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 82/100
    560/560 [==============================] - 0s 21us/step - loss: 0.0072 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 83/100
    560/560 [==============================] - 0s 29us/step - loss: 0.0072 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 84/100
    560/560 [==============================] - 0s 30us/step - loss: 0.0072 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 85/100
    560/560 [==============================] - 0s 36us/step - loss: 0.0072 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 86/100
    560/560 [==============================] - 0s 25us/step - loss: 0.0072 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 87/100
    560/560 [==============================] - 0s 25us/step - loss: 0.0071 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 88/100
    560/560 [==============================] - 0s 20us/step - loss: 0.0071 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 89/100
    560/560 [==============================] - 0s 21us/step - loss: 0.0071 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 90/100
    560/560 [==============================] - 0s 23us/step - loss: 0.0071 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 91/100
    560/560 [==============================] - 0s 21us/step - loss: 0.0071 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 92/100
    560/560 [==============================] - 0s 30us/step - loss: 0.0071 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 93/100
    560/560 [==============================] - 0s 21us/step - loss: 0.0071 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 94/100
    560/560 [==============================] - 0s 27us/step - loss: 0.0071 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 95/100
    560/560 [==============================] - 0s 23us/step - loss: 0.0071 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 96/100
    560/560 [==============================] - 0s 29us/step - loss: 0.0071 - acc: 1.0000 - val_loss: 0.0072 - val_acc: 1.0000
    Epoch 97/100
    560/560 [==============================] - 0s 21us/step - loss: 0.0071 - acc: 1.0000 - val_loss: 0.0071 - val_acc: 1.0000
    Epoch 98/100
    560/560 [==============================] - 0s 27us/step - loss: 0.0071 - acc: 1.0000 - val_loss: 0.0071 - val_acc: 1.0000
    Epoch 99/100
    560/560 [==============================] - 0s 21us/step - loss: 0.0071 - acc: 1.0000 - val_loss: 0.0071 - val_acc: 1.0000
    Epoch 100/100
    560/560 [==============================] - 0s 30us/step - loss: 0.0071 - acc: 1.0000 - val_loss: 0.0071 - val_acc: 1.0000
    


```python
model.get_layer(index = 0).get_weights()
```




    [array([[ 10.006285, -10.000407],
            [ 10.005641, -10.001053]], dtype=float32),
     array([-4.993556 , 15.0052595], dtype=float32)]




```python
model.get_layer(index = 2).get_weights()
```




    [array([[10.078076],
            [10.096514]], dtype=float32), array([-15.001636], dtype=float32)]



Although it is not a local minimum, it suffices to classify the result. In general, we don't know which initialization is good. We often increase the number of units in the hidden layer to obtain the chance to get a better local minimum.


```python
model = Sequential()
model.add(Dense(6, input_shape = (2,), kernel_initializer=RandomUniform(-5, 5), bias_initializer=RandomUniform(-5, 5)))
model.add(Activation('sigmoid'))
model.add(Dense(1, kernel_initializer=RandomUniform(-5, 5), bias_initializer=RandomUniform(-5, 5)))
model.add(Activation('sigmoid'))
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_47 (Dense)             (None, 6)                 18        
    _________________________________________________________________
    activation_47 (Activation)   (None, 6)                 0         
    _________________________________________________________________
    dense_48 (Dense)             (None, 1)                 7         
    _________________________________________________________________
    activation_48 (Activation)   (None, 1)                 0         
    =================================================================
    Total params: 25
    Trainable params: 25
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model.compile(loss='binary_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=100, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
```

    Train on 560 samples, validate on 140 samples
    Epoch 1/100
    560/560 [==============================] - 2s 4ms/step - loss: 3.9807 - acc: 0.4839 - val_loss: 3.3641 - val_acc: 0.5000
    Epoch 2/100
    560/560 [==============================] - 0s 30us/step - loss: 3.3646 - acc: 0.4839 - val_loss: 2.7736 - val_acc: 0.5000
    Epoch 3/100
    560/560 [==============================] - 0s 32us/step - loss: 2.7599 - acc: 0.4839 - val_loss: 2.2065 - val_acc: 0.5000
    Epoch 4/100
    560/560 [==============================] - 0s 41us/step - loss: 2.1936 - acc: 0.4839 - val_loss: 1.7251 - val_acc: 0.5000
    Epoch 5/100
    560/560 [==============================] - 0s 29us/step - loss: 1.7372 - acc: 0.4839 - val_loss: 1.3918 - val_acc: 0.5000
    Epoch 6/100
    560/560 [==============================] - 0s 50us/step - loss: 1.4317 - acc: 0.5643 - val_loss: 1.1686 - val_acc: 0.7571
    Epoch 7/100
    560/560 [==============================] - 0s 36us/step - loss: 1.2254 - acc: 0.5143 - val_loss: 1.0276 - val_acc: 0.5143
    Epoch 8/100
    560/560 [==============================] - 0s 52us/step - loss: 1.0882 - acc: 0.4607 - val_loss: 0.9280 - val_acc: 0.5143
    Epoch 9/100
    560/560 [==============================] - 0s 29us/step - loss: 0.9868 - acc: 0.4607 - val_loss: 0.8441 - val_acc: 0.5143
    Epoch 10/100
    560/560 [==============================] - 0s 43us/step - loss: 0.9033 - acc: 0.4607 - val_loss: 0.7831 - val_acc: 0.5143
    Epoch 11/100
    560/560 [==============================] - 0s 27us/step - loss: 0.8403 - acc: 0.4607 - val_loss: 0.7332 - val_acc: 0.5143
    Epoch 12/100
    560/560 [==============================] - 0s 46us/step - loss: 0.7880 - acc: 0.4607 - val_loss: 0.6892 - val_acc: 0.5143
    Epoch 13/100
    560/560 [==============================] - 0s 30us/step - loss: 0.7427 - acc: 0.4607 - val_loss: 0.6552 - val_acc: 0.5143
    Epoch 14/100
    560/560 [==============================] - 0s 30us/step - loss: 0.7074 - acc: 0.4607 - val_loss: 0.6269 - val_acc: 0.5143
    Epoch 15/100
    560/560 [==============================] - 0s 27us/step - loss: 0.6773 - acc: 0.4607 - val_loss: 0.6031 - val_acc: 0.5143
    Epoch 16/100
    560/560 [==============================] - 0s 45us/step - loss: 0.6521 - acc: 0.4607 - val_loss: 0.5823 - val_acc: 0.5143
    Epoch 17/100
    560/560 [==============================] - 0s 30us/step - loss: 0.6303 - acc: 0.4607 - val_loss: 0.5629 - val_acc: 0.5143
    Epoch 18/100
    560/560 [==============================] - 0s 34us/step - loss: 0.6086 - acc: 0.4607 - val_loss: 0.5464 - val_acc: 0.5143
    Epoch 19/100
    560/560 [==============================] - 0s 30us/step - loss: 0.5913 - acc: 0.4607 - val_loss: 0.5312 - val_acc: 0.5143
    Epoch 20/100
    560/560 [==============================] - 0s 34us/step - loss: 0.5749 - acc: 0.4607 - val_loss: 0.5173 - val_acc: 0.5143
    Epoch 21/100
    560/560 [==============================] - 0s 32us/step - loss: 0.5595 - acc: 0.4607 - val_loss: 0.5049 - val_acc: 0.5143
    Epoch 22/100
    560/560 [==============================] - 0s 27us/step - loss: 0.5462 - acc: 0.4607 - val_loss: 0.4930 - val_acc: 0.5143
    Epoch 23/100
    560/560 [==============================] - 0s 39us/step - loss: 0.5330 - acc: 0.4607 - val_loss: 0.4821 - val_acc: 0.5143
    Epoch 24/100
    560/560 [==============================] - 0s 30us/step - loss: 0.5212 - acc: 0.4607 - val_loss: 0.4721 - val_acc: 0.5143
    Epoch 25/100
    560/560 [==============================] - 0s 30us/step - loss: 0.5111 - acc: 0.4607 - val_loss: 0.4627 - val_acc: 0.5143
    Epoch 26/100
    560/560 [==============================] - 0s 36us/step - loss: 0.5003 - acc: 0.4607 - val_loss: 0.4538 - val_acc: 0.5143
    Epoch 27/100
    560/560 [==============================] - 0s 30us/step - loss: 0.4906 - acc: 0.4607 - val_loss: 0.4454 - val_acc: 0.5143
    Epoch 28/100
    560/560 [==============================] - 0s 25us/step - loss: 0.4815 - acc: 0.4607 - val_loss: 0.4378 - val_acc: 0.5143
    Epoch 29/100
    560/560 [==============================] - 0s 30us/step - loss: 0.4734 - acc: 0.6071 - val_loss: 0.4300 - val_acc: 0.5143
    Epoch 30/100
    560/560 [==============================] - 0s 43us/step - loss: 0.4647 - acc: 0.4786 - val_loss: 0.4228 - val_acc: 0.5143
    Epoch 31/100
    560/560 [==============================] - 0s 30us/step - loss: 0.4566 - acc: 0.4839 - val_loss: 0.4160 - val_acc: 0.7571
    Epoch 32/100
    560/560 [==============================] - 0s 29us/step - loss: 0.4494 - acc: 0.6661 - val_loss: 0.4095 - val_acc: 0.7571
    Epoch 33/100
    560/560 [==============================] - 0s 38us/step - loss: 0.4420 - acc: 0.7321 - val_loss: 0.4032 - val_acc: 0.7571
    Epoch 34/100
    560/560 [==============================] - 0s 30us/step - loss: 0.4351 - acc: 0.7321 - val_loss: 0.3973 - val_acc: 0.7571
    Epoch 35/100
    560/560 [==============================] - 0s 29us/step - loss: 0.4289 - acc: 0.7321 - val_loss: 0.3917 - val_acc: 0.7571
    Epoch 36/100
    560/560 [==============================] - 0s 34us/step - loss: 0.4228 - acc: 0.7321 - val_loss: 0.3861 - val_acc: 0.7571
    Epoch 37/100
    560/560 [==============================] - 0s 27us/step - loss: 0.4163 - acc: 0.7321 - val_loss: 0.3808 - val_acc: 0.7571
    Epoch 38/100
    560/560 [==============================] - 0s 29us/step - loss: 0.4105 - acc: 0.7321 - val_loss: 0.3757 - val_acc: 0.7571
    Epoch 39/100
    560/560 [==============================] - 0s 36us/step - loss: 0.4053 - acc: 0.8375 - val_loss: 0.3706 - val_acc: 0.7571
    Epoch 40/100
    560/560 [==============================] - 0s 29us/step - loss: 0.3992 - acc: 0.8107 - val_loss: 0.3658 - val_acc: 0.7571
    Epoch 41/100
    560/560 [==============================] - 0s 29us/step - loss: 0.3944 - acc: 0.8393 - val_loss: 0.3612 - val_acc: 0.7571
    Epoch 42/100
    560/560 [==============================] - 0s 48us/step - loss: 0.3889 - acc: 0.8714 - val_loss: 0.3568 - val_acc: 1.0000
    Epoch 43/100
    560/560 [==============================] - 0s 32us/step - loss: 0.3841 - acc: 0.9304 - val_loss: 0.3525 - val_acc: 1.0000
    Epoch 44/100
    560/560 [==============================] - 0s 29us/step - loss: 0.3793 - acc: 1.0000 - val_loss: 0.3483 - val_acc: 1.0000
    Epoch 45/100
    560/560 [==============================] - 0s 38us/step - loss: 0.3747 - acc: 1.0000 - val_loss: 0.3441 - val_acc: 1.0000
    Epoch 46/100
    560/560 [==============================] - 0s 29us/step - loss: 0.3701 - acc: 1.0000 - val_loss: 0.3401 - val_acc: 1.0000
    Epoch 47/100
    560/560 [==============================] - 0s 27us/step - loss: 0.3659 - acc: 1.0000 - val_loss: 0.3364 - val_acc: 1.0000
    Epoch 48/100
    560/560 [==============================] - 0s 27us/step - loss: 0.3616 - acc: 1.0000 - val_loss: 0.3328 - val_acc: 1.0000
    Epoch 49/100
    560/560 [==============================] - 0s 38us/step - loss: 0.3575 - acc: 1.0000 - val_loss: 0.3287 - val_acc: 1.0000
    Epoch 50/100
    560/560 [==============================] - 0s 29us/step - loss: 0.3530 - acc: 1.0000 - val_loss: 0.3251 - val_acc: 1.0000
    Epoch 51/100
    560/560 [==============================] - 0s 30us/step - loss: 0.3491 - acc: 1.0000 - val_loss: 0.3217 - val_acc: 1.0000
    Epoch 52/100
    560/560 [==============================] - 0s 30us/step - loss: 0.3455 - acc: 1.0000 - val_loss: 0.3183 - val_acc: 1.0000
    Epoch 53/100
    560/560 [==============================] - 0s 34us/step - loss: 0.3416 - acc: 1.0000 - val_loss: 0.3149 - val_acc: 1.0000
    Epoch 54/100
    560/560 [==============================] - 0s 34us/step - loss: 0.3377 - acc: 1.0000 - val_loss: 0.3115 - val_acc: 1.0000
    Epoch 55/100
    560/560 [==============================] - 0s 30us/step - loss: 0.3339 - acc: 1.0000 - val_loss: 0.3082 - val_acc: 1.0000
    Epoch 56/100
    560/560 [==============================] - 0s 38us/step - loss: 0.3303 - acc: 1.0000 - val_loss: 0.3051 - val_acc: 1.0000
    Epoch 57/100
    560/560 [==============================] - 0s 32us/step - loss: 0.3272 - acc: 1.0000 - val_loss: 0.3021 - val_acc: 1.0000
    Epoch 58/100
    560/560 [==============================] - 0s 29us/step - loss: 0.3234 - acc: 1.0000 - val_loss: 0.2990 - val_acc: 1.0000
    Epoch 59/100
    560/560 [==============================] - 0s 27us/step - loss: 0.3199 - acc: 1.0000 - val_loss: 0.2960 - val_acc: 1.0000
    Epoch 60/100
    560/560 [==============================] - 0s 36us/step - loss: 0.3166 - acc: 1.0000 - val_loss: 0.2929 - val_acc: 1.0000
    Epoch 61/100
    560/560 [==============================] - 0s 50us/step - loss: 0.3133 - acc: 1.0000 - val_loss: 0.2900 - val_acc: 1.0000
    Epoch 62/100
    560/560 [==============================] - 0s 25us/step - loss: 0.3103 - acc: 1.0000 - val_loss: 0.2872 - val_acc: 1.0000
    Epoch 63/100
    560/560 [==============================] - 0s 29us/step - loss: 0.3071 - acc: 1.0000 - val_loss: 0.2843 - val_acc: 1.0000
    Epoch 64/100
    560/560 [==============================] - 0s 29us/step - loss: 0.3040 - acc: 1.0000 - val_loss: 0.2817 - val_acc: 1.0000
    Epoch 65/100
    560/560 [==============================] - 0s 30us/step - loss: 0.3009 - acc: 1.0000 - val_loss: 0.2788 - val_acc: 1.0000
    Epoch 66/100
    560/560 [==============================] - 0s 29us/step - loss: 0.2977 - acc: 1.0000 - val_loss: 0.2761 - val_acc: 1.0000
    Epoch 67/100
    560/560 [==============================] - 0s 48us/step - loss: 0.2947 - acc: 1.0000 - val_loss: 0.2735 - val_acc: 1.0000
    Epoch 68/100
    560/560 [==============================] - 0s 29us/step - loss: 0.2920 - acc: 1.0000 - val_loss: 0.2709 - val_acc: 1.0000
    Epoch 69/100
    560/560 [==============================] - 0s 25us/step - loss: 0.2890 - acc: 1.0000 - val_loss: 0.2682 - val_acc: 1.0000
    Epoch 70/100
    560/560 [==============================] - 0s 29us/step - loss: 0.2860 - acc: 1.0000 - val_loss: 0.2658 - val_acc: 1.0000
    Epoch 71/100
    560/560 [==============================] - 0s 55us/step - loss: 0.2832 - acc: 1.0000 - val_loss: 0.2633 - val_acc: 1.0000
    Epoch 72/100
    560/560 [==============================] - 0s 32us/step - loss: 0.2804 - acc: 1.0000 - val_loss: 0.2609 - val_acc: 1.0000
    Epoch 73/100
    560/560 [==============================] - 0s 34us/step - loss: 0.2778 - acc: 1.0000 - val_loss: 0.2585 - val_acc: 1.0000
    Epoch 74/100
    560/560 [==============================] - 0s 36us/step - loss: 0.2751 - acc: 1.0000 - val_loss: 0.2561 - val_acc: 1.0000
    Epoch 75/100
    560/560 [==============================] - 0s 32us/step - loss: 0.2725 - acc: 1.0000 - val_loss: 0.2538 - val_acc: 1.0000
    Epoch 76/100
    560/560 [==============================] - 0s 39us/step - loss: 0.2700 - acc: 1.0000 - val_loss: 0.2515 - val_acc: 1.0000
    Epoch 77/100
    560/560 [==============================] - 0s 36us/step - loss: 0.2676 - acc: 1.0000 - val_loss: 0.2493 - val_acc: 1.0000
    Epoch 78/100
    560/560 [==============================] - 0s 36us/step - loss: 0.2649 - acc: 1.0000 - val_loss: 0.2471 - val_acc: 1.0000
    Epoch 79/100
    560/560 [==============================] - 0s 39us/step - loss: 0.2626 - acc: 1.0000 - val_loss: 0.2450 - val_acc: 1.0000
    Epoch 80/100
    560/560 [==============================] - 0s 27us/step - loss: 0.2602 - acc: 1.0000 - val_loss: 0.2429 - val_acc: 1.0000
    Epoch 81/100
    560/560 [==============================] - 0s 36us/step - loss: 0.2580 - acc: 1.0000 - val_loss: 0.2411 - val_acc: 1.0000
    Epoch 82/100
    560/560 [==============================] - 0s 38us/step - loss: 0.2557 - acc: 1.0000 - val_loss: 0.2388 - val_acc: 1.0000
    Epoch 83/100
    560/560 [==============================] - 0s 29us/step - loss: 0.2536 - acc: 1.0000 - val_loss: 0.2367 - val_acc: 1.0000
    Epoch 84/100
    560/560 [==============================] - 0s 38us/step - loss: 0.2511 - acc: 1.0000 - val_loss: 0.2347 - val_acc: 1.0000
    Epoch 85/100
    560/560 [==============================] - 0s 29us/step - loss: 0.2489 - acc: 1.0000 - val_loss: 0.2327 - val_acc: 1.0000
    Epoch 86/100
    560/560 [==============================] - 0s 32us/step - loss: 0.2467 - acc: 1.0000 - val_loss: 0.2308 - val_acc: 1.0000
    Epoch 87/100
    560/560 [==============================] - 0s 34us/step - loss: 0.2445 - acc: 1.0000 - val_loss: 0.2288 - val_acc: 1.0000
    Epoch 88/100
    560/560 [==============================] - 0s 30us/step - loss: 0.2424 - acc: 1.0000 - val_loss: 0.2268 - val_acc: 1.0000
    Epoch 89/100
    560/560 [==============================] - 0s 34us/step - loss: 0.2403 - acc: 1.0000 - val_loss: 0.2249 - val_acc: 1.0000
    Epoch 90/100
    560/560 [==============================] - 0s 30us/step - loss: 0.2381 - acc: 1.0000 - val_loss: 0.2232 - val_acc: 1.0000
    Epoch 91/100
    560/560 [==============================] - 0s 32us/step - loss: 0.2362 - acc: 1.0000 - val_loss: 0.2215 - val_acc: 1.0000
    Epoch 92/100
    560/560 [==============================] - 0s 36us/step - loss: 0.2343 - acc: 1.0000 - val_loss: 0.2195 - val_acc: 1.0000
    Epoch 93/100
    560/560 [==============================] - 0s 34us/step - loss: 0.2321 - acc: 1.0000 - val_loss: 0.2179 - val_acc: 1.0000
    Epoch 94/100
    560/560 [==============================] - 0s 30us/step - loss: 0.2303 - acc: 1.0000 - val_loss: 0.2161 - val_acc: 1.0000
    Epoch 95/100
    560/560 [==============================] - 0s 32us/step - loss: 0.2283 - acc: 1.0000 - val_loss: 0.2144 - val_acc: 1.0000
    Epoch 96/100
    560/560 [==============================] - 0s 38us/step - loss: 0.2264 - acc: 1.0000 - val_loss: 0.2126 - val_acc: 1.0000
    Epoch 97/100
    560/560 [==============================] - 0s 29us/step - loss: 0.2245 - acc: 1.0000 - val_loss: 0.2109 - val_acc: 1.0000
    Epoch 98/100
    560/560 [==============================] - 0s 23us/step - loss: 0.2226 - acc: 1.0000 - val_loss: 0.2092 - val_acc: 1.0000
    Epoch 99/100
    560/560 [==============================] - 0s 25us/step - loss: 0.2209 - acc: 1.0000 - val_loss: 0.2076 - val_acc: 1.0000
    Epoch 100/100
    560/560 [==============================] - 0s 30us/step - loss: 0.2190 - acc: 1.0000 - val_loss: 0.2060 - val_acc: 1.0000
    


```python
model.get_layer(index = 0).get_weights(), model.get_layer(index = 2).get_weights()
```




    ([array([[ 0.6345892 ,  0.21349257,  5.190718  ,  2.6648445 ,  5.9560175 ,
               3.2525434 ],
             [-2.5391288 ,  0.7896347 , -3.460859  , -2.393607  ,  3.5305624 ,
              -2.6143186 ]], dtype=float32),
      array([ 0.27963892,  4.7800684 ,  1.5024713 , -6.227105  , -1.5551796 ,
              2.260877  ], dtype=float32)],
     [array([[ 4.3241935],
             [ 1.5521915],
             [-2.3014896],
             [-3.4141047],
             [ 4.4846277],
             [-3.6434932]], dtype=float32), array([-1.9597361], dtype=float32)])



# 4. Elements of Feedforward Networks

## 4.1 Cost Function

The cost function $L(\mathbf x_{(1)}, \ldots, \mathbf t_{(N)}, t_{(1)}, \ldots, t_{(N)})$ can be defined as the sum/mean of the loss on the $N$ training values

$$
L(\mathbf x_{(1)}, \ldots, \mathbf t_{(N)}, t_{(1)}, \ldots, t_{(N)}) = \sum_{n=1}^N l(f(\mathbf x_n), t_n)) = \sum_{n=1}^N l(y_n, t_n))
$$

where $y$ calculated from $\mathbf x$ and the model architecture.

Like basic regression and classification problem models, we can choose:

- Mean Squared Error for regression
$$
l(t, y) = \frac12 |t-y|^2
$$

- Binary Cross-entropy for binary classification
$$
l(t, y) = -t\log y - (1-t)\log(1-y)
$$
where $y\in [0,1]$ predicted as a probability.

- Cross-entropy for for multiclass classification
$$
l(t, y) = -\sum_{k=1}^K t_k \log y_k
$$
where $y$ predicted as $(y_1, \ldots, y_K)$, $0 \leq y_k \leq 1$, $\sum_{k=1}^K y_k = 1.$

We know that Mean Squared Error is obtained when we suppose that $t$ is a random variable following Gaussian distribution of mean $y=f(\mathbf x)$ and some variance; Binary Cross-entropy is obtained when we suppose that $t$ follows a Bernoulli distribution of parameter $y$; while general Cross-entropy is obtained with supposition that $t$ follows a multinomial distribution of paramter $y = (y_1, \ldots, y_K)$.

If we change, for example in case of regression, $t$ follows a symmetric exponential distribution, then the loss become 
$$
l(t, y) = \Vert t - y \Vert_1
$$
It is an example to show that we can change the loss function by changing the hypothesis of the probability distribution $p(t|\mathbf x)$.

## 4.2 The Output Layer

For regression, there is no constraint of the output layer, it can be any real number, so the simplest activation function can be chosen is the identity function.

For binary classification, if we choose binary cross-entropy as the loss function, we expect the output to be some number between $[0,1]$, so the activation function at the output layer un should be a function $\mathbf R \to [0,1]$. The sigmoid function is one possible choice and widely used. Note that if we choose another loss function (like $[yt]_{+}$), we can use another activation function (like the identity).

For multiclass classification, if we choose the general cross-entropy as the loss function, we also expect the output to be a vector summed to 1. A possible choice of the activation function at the output layer is the softmax.

## 4.3 Hidden Layers

Choice of hidden layers does not depend on the cost function. The most popular are:

### 4.3.1 Rectified Linear Unit (ReLU)
$$
h(a) = [a]_{+} = \max(0, a)
$$


```python
def relu(a):
    return np.max([0, a])

import matplotlib.pyplot as plt
%matplotlib inline

interval = np.linspace(-5, 5, 101)
plt.plot(interval, [relu(x) for x in interval])
```




    [<matplotlib.lines.Line2D at 0x1d579a58>]




![png](https://raw.githubusercontent.com/riduan91/DSC111/master/Lesson1/Amphi/output_37_1.png)


### 4.3.2 Logistic Sigmoid
$$
\sigma(a) = \frac1{1+ \exp(-a)}
$$


```python
def sigmoid(a):
    return 1/(1+np.exp(-a))

plt.plot(interval, sigmoid(interval))
```




    [<matplotlib.lines.Line2D at 0x1d4f6390>]




![png](https://raw.githubusercontent.com/riduan91/DSC111/master/Lesson1/Amphi/output_39_1.png)


Sigmoidal units saturate across most of their domain (i.e., their gradient is nearly zero across most of their domain). This will make gradient-based learning very difficult. Their use as hidden units is usually discouraged.

### 4.3.3 tanh Function
$$
h(a) = \tanh (a)
$$


```python
plt.plot(interval, np.tanh(interval))
```




    [<matplotlib.lines.Line2D at 0x1d51afd0>]




![png](https://raw.githubusercontent.com/riduan91/DSC111/master/Lesson1/Amphi/output_42_1.png)


### 4.4 Universal Approximation Properties

The **universal approximation theorem** states that a feedforward network with a linear output layer and at least one hidden layer with 
- logistic sigmoid activation function
- tanh activation function
- rectified linear unit activation function

can approximately any Borel measureable function from one finite-dimensional space to another, with any desired non-zero amount of errors like:
- MSE
- (binary, multiclass) cross-entropy

provided that the network is given **enough** hidden units.

So, a feedforward network with single layer is sufficient to represent any function, but the layer may be very large and may fail to learn correctly with gradient-based methods.

**Example: Approximation of the function $x^2$**


```python
X = np.random.uniform(-5, 5, size=(1000, 1))
y = (X ** 2).flatten()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
pd.DataFrame([X[:,0], y])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>990</th>
      <th>991</th>
      <th>992</th>
      <th>993</th>
      <th>994</th>
      <th>995</th>
      <th>996</th>
      <th>997</th>
      <th>998</th>
      <th>999</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-3.636879</td>
      <td>-3.266407</td>
      <td>-1.131050</td>
      <td>1.954376</td>
      <td>-0.192230</td>
      <td>-3.321515</td>
      <td>-4.689132</td>
      <td>-1.377835</td>
      <td>-4.627739</td>
      <td>4.002532</td>
      <td>...</td>
      <td>4.33169</td>
      <td>-3.581613</td>
      <td>-3.325948</td>
      <td>3.811807</td>
      <td>-0.006702</td>
      <td>-1.193012</td>
      <td>4.599222</td>
      <td>-2.518922</td>
      <td>1.616815</td>
      <td>-4.772549</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13.226886</td>
      <td>10.669415</td>
      <td>1.279275</td>
      <td>3.819586</td>
      <td>0.036952</td>
      <td>11.032459</td>
      <td>21.987963</td>
      <td>1.898430</td>
      <td>21.415967</td>
      <td>16.020263</td>
      <td>...</td>
      <td>18.76354</td>
      <td>12.827953</td>
      <td>11.061933</td>
      <td>14.529872</td>
      <td>0.000045</td>
      <td>1.423278</td>
      <td>21.152844</td>
      <td>6.344969</td>
      <td>2.614092</td>
      <td>22.777221</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 1000 columns</p>
</div>



**Using ReLU**


```python
model = Sequential()
model.add(Dense(4, input_shape = (1,), kernel_initializer=RandomUniform(-2, 2), bias_initializer=RandomUniform(-2, 2)))
model.add(Activation('relu'))
model.add(Dense(1, kernel_initializer=RandomUniform(-2, 2), bias_initializer=RandomUniform(-2, 2)))
model.add(Activation('linear'))
model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01))
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=100, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
```

    Train on 560 samples, validate on 140 samples
    Epoch 1/100
    560/560 [==============================] - 2s 4ms/step - loss: 67.0041 - val_loss: 26.4730
    Epoch 2/100
    560/560 [==============================] - 0s 27us/step - loss: 22.1227 - val_loss: 21.2005
    Epoch 3/100
    560/560 [==============================] - 0s 46us/step - loss: 18.3511 - val_loss: 18.5854
    Epoch 4/100
    560/560 [==============================] - 0s 38us/step - loss: 16.1558 - val_loss: 16.4015
    Epoch 5/100
    560/560 [==============================] - 0s 45us/step - loss: 14.1631 - val_loss: 14.4693
    Epoch 6/100
    560/560 [==============================] - 0s 29us/step - loss: 13.0349 - val_loss: 13.6919
    Epoch 7/100
    560/560 [==============================] - 0s 43us/step - loss: 11.8046 - val_loss: 11.9613
    Epoch 8/100
    560/560 [==============================] - 0s 27us/step - loss: 10.8738 - val_loss: 11.0355
    Epoch 9/100
    560/560 [==============================] - 0s 50us/step - loss: 9.9145 - val_loss: 10.6021
    Epoch 10/100
    560/560 [==============================] - 0s 32us/step - loss: 9.2531 - val_loss: 9.4695
    Epoch 11/100
    560/560 [==============================] - 0s 27us/step - loss: 8.5979 - val_loss: 8.8155
    Epoch 12/100
    560/560 [==============================] - 0s 29us/step - loss: 8.0236 - val_loss: 8.2725
    Epoch 13/100
    560/560 [==============================] - 0s 27us/step - loss: 7.3837 - val_loss: 7.7706
    Epoch 14/100
    560/560 [==============================] - 0s 50us/step - loss: 6.9683 - val_loss: 7.3715
    Epoch 15/100
    560/560 [==============================] - 0s 27us/step - loss: 6.7191 - val_loss: 6.8272
    Epoch 16/100
    560/560 [==============================] - 0s 43us/step - loss: 6.3348 - val_loss: 6.5492
    Epoch 17/100
    560/560 [==============================] - 0s 30us/step - loss: 6.0335 - val_loss: 6.1675
    Epoch 18/100
    560/560 [==============================] - 0s 25us/step - loss: 5.5795 - val_loss: 6.1818
    Epoch 19/100
    560/560 [==============================] - 0s 45us/step - loss: 5.5234 - val_loss: 5.5472
    Epoch 20/100
    560/560 [==============================] - 0s 34us/step - loss: 5.1633 - val_loss: 6.0941
    Epoch 21/100
    560/560 [==============================] - 0s 25us/step - loss: 5.0551 - val_loss: 5.0999
    Epoch 22/100
    560/560 [==============================] - 0s 43us/step - loss: 4.6822 - val_loss: 4.9782
    Epoch 23/100
    560/560 [==============================] - 0s 32us/step - loss: 4.5000 - val_loss: 5.2731
    Epoch 24/100
    560/560 [==============================] - 0s 32us/step - loss: 4.5159 - val_loss: 4.6112
    Epoch 25/100
    560/560 [==============================] - 0s 39us/step - loss: 4.2222 - val_loss: 4.4328
    Epoch 26/100
    560/560 [==============================] - 0s 27us/step - loss: 4.0597 - val_loss: 4.2244
    Epoch 27/100
    560/560 [==============================] - 0s 23us/step - loss: 3.9272 - val_loss: 4.0786
    Epoch 28/100
    560/560 [==============================] - 0s 36us/step - loss: 3.8184 - val_loss: 4.0048
    Epoch 29/100
    560/560 [==============================] - 0s 34us/step - loss: 3.7184 - val_loss: 3.8354
    Epoch 30/100
    560/560 [==============================] - 0s 25us/step - loss: 3.5941 - val_loss: 3.7904
    Epoch 31/100
    560/560 [==============================] - 0s 27us/step - loss: 3.5919 - val_loss: 3.8272
    Epoch 32/100
    560/560 [==============================] - 0s 38us/step - loss: 3.3515 - val_loss: 3.5750
    Epoch 33/100
    560/560 [==============================] - 0s 29us/step - loss: 3.3348 - val_loss: 3.8271
    Epoch 34/100
    560/560 [==============================] - 0s 27us/step - loss: 3.3035 - val_loss: 3.6798
    Epoch 35/100
    560/560 [==============================] - 0s 46us/step - loss: 3.2402 - val_loss: 3.2357
    Epoch 36/100
    560/560 [==============================] - 0s 29us/step - loss: 3.0690 - val_loss: 3.4253
    Epoch 37/100
    560/560 [==============================] - 0s 30us/step - loss: 3.1368 - val_loss: 3.1710
    Epoch 38/100
    560/560 [==============================] - 0s 32us/step - loss: 2.9811 - val_loss: 3.0751
    Epoch 39/100
    560/560 [==============================] - 0s 29us/step - loss: 2.9140 - val_loss: 3.0500
    Epoch 40/100
    560/560 [==============================] - 0s 23us/step - loss: 2.8949 - val_loss: 3.0578
    Epoch 41/100
    560/560 [==============================] - 0s 23us/step - loss: 2.8152 - val_loss: 3.3445
    Epoch 42/100
    560/560 [==============================] - 0s 36us/step - loss: 2.9023 - val_loss: 2.8449
    Epoch 43/100
    560/560 [==============================] - 0s 21us/step - loss: 2.7077 - val_loss: 2.8572
    Epoch 44/100
    560/560 [==============================] - 0s 21us/step - loss: 2.6904 - val_loss: 2.8667
    Epoch 45/100
    560/560 [==============================] - 0s 21us/step - loss: 2.7667 - val_loss: 2.6935
    Epoch 46/100
    560/560 [==============================] - 0s 34us/step - loss: 2.5441 - val_loss: 2.6485
    Epoch 47/100
    560/560 [==============================] - 0s 21us/step - loss: 2.5224 - val_loss: 2.6874
    Epoch 48/100
    560/560 [==============================] - 0s 21us/step - loss: 2.5714 - val_loss: 2.5300
    Epoch 49/100
    560/560 [==============================] - 0s 21us/step - loss: 2.4591 - val_loss: 2.4919
    Epoch 50/100
    560/560 [==============================] - 0s 34us/step - loss: 2.4856 - val_loss: 2.4772
    Epoch 51/100
    560/560 [==============================] - 0s 27us/step - loss: 2.4545 - val_loss: 2.4694
    Epoch 52/100
    560/560 [==============================] - 0s 25us/step - loss: 2.3867 - val_loss: 2.3925
    Epoch 53/100
    560/560 [==============================] - 0s 20us/step - loss: 2.3603 - val_loss: 2.5830
    Epoch 54/100
    560/560 [==============================] - 0s 34us/step - loss: 2.3781 - val_loss: 2.4637
    Epoch 55/100
    560/560 [==============================] - 0s 29us/step - loss: 2.3747 - val_loss: 2.3557
    Epoch 56/100
    560/560 [==============================] - 0s 23us/step - loss: 2.2595 - val_loss: 2.3843
    Epoch 57/100
    560/560 [==============================] - 0s 27us/step - loss: 2.2269 - val_loss: 2.4703
    Epoch 58/100
    560/560 [==============================] - 0s 21us/step - loss: 2.2744 - val_loss: 2.4079
    Epoch 59/100
    560/560 [==============================] - 0s 38us/step - loss: 2.3043 - val_loss: 2.2396
    Epoch 60/100
    560/560 [==============================] - 0s 23us/step - loss: 2.1694 - val_loss: 2.2479
    Epoch 61/100
    560/560 [==============================] - 0s 21us/step - loss: 2.1584 - val_loss: 2.1805
    Epoch 62/100
    560/560 [==============================] - 0s 23us/step - loss: 2.1328 - val_loss: 2.3013
    Epoch 63/100
    560/560 [==============================] - 0s 21us/step - loss: 2.1784 - val_loss: 2.3265
    Epoch 64/100
    560/560 [==============================] - 0s 20us/step - loss: 2.1171 - val_loss: 2.2274
    Epoch 65/100
    560/560 [==============================] - 0s 41us/step - loss: 2.0871 - val_loss: 2.2418
    Epoch 66/100
    560/560 [==============================] - 0s 21us/step - loss: 2.0561 - val_loss: 2.0816
    Epoch 67/100
    560/560 [==============================] - 0s 21us/step - loss: 1.9915 - val_loss: 2.0492
    Epoch 68/100
    560/560 [==============================] - 0s 32us/step - loss: 1.9851 - val_loss: 2.0241
    Epoch 69/100
    560/560 [==============================] - 0s 21us/step - loss: 1.9702 - val_loss: 2.0714
    Epoch 70/100
    560/560 [==============================] - 0s 23us/step - loss: 2.0101 - val_loss: 2.0170
    Epoch 71/100
    560/560 [==============================] - 0s 34us/step - loss: 1.9074 - val_loss: 1.9364
    Epoch 72/100
    560/560 [==============================] - 0s 23us/step - loss: 1.8833 - val_loss: 1.8980
    Epoch 73/100
    560/560 [==============================] - 0s 20us/step - loss: 1.8683 - val_loss: 1.9440
    Epoch 74/100
    560/560 [==============================] - 0s 20us/step - loss: 1.8851 - val_loss: 2.0647
    Epoch 75/100
    560/560 [==============================] - 0s 29us/step - loss: 1.8810 - val_loss: 1.8473
    Epoch 76/100
    560/560 [==============================] - 0s 21us/step - loss: 1.8315 - val_loss: 1.8998
    Epoch 77/100
    560/560 [==============================] - 0s 21us/step - loss: 1.8094 - val_loss: 1.8222
    Epoch 78/100
    560/560 [==============================] - 0s 41us/step - loss: 1.8207 - val_loss: 1.8414
    Epoch 79/100
    560/560 [==============================] - 0s 23us/step - loss: 1.7619 - val_loss: 1.8160
    Epoch 80/100
    560/560 [==============================] - 0s 48us/step - loss: 1.7393 - val_loss: 1.7731
    Epoch 81/100
    560/560 [==============================] - 0s 21us/step - loss: 1.7405 - val_loss: 1.7591
    Epoch 82/100
    560/560 [==============================] - 0s 20us/step - loss: 1.7267 - val_loss: 1.8874
    Epoch 83/100
    560/560 [==============================] - 0s 30us/step - loss: 1.7781 - val_loss: 1.8381
    Epoch 84/100
    560/560 [==============================] - 0s 23us/step - loss: 1.6987 - val_loss: 1.7002
    Epoch 85/100
    560/560 [==============================] - 0s 25us/step - loss: 1.6803 - val_loss: 1.7603
    Epoch 86/100
    560/560 [==============================] - 0s 18us/step - loss: 1.6924 - val_loss: 1.8362
    Epoch 87/100
    560/560 [==============================] - 0s 29us/step - loss: 1.6364 - val_loss: 1.6814
    Epoch 88/100
    560/560 [==============================] - 0s 27us/step - loss: 1.6062 - val_loss: 1.6793
    Epoch 89/100
    560/560 [==============================] - 0s 25us/step - loss: 1.6423 - val_loss: 1.7415
    Epoch 90/100
    560/560 [==============================] - 0s 25us/step - loss: 1.6081 - val_loss: 1.6661
    Epoch 91/100
    560/560 [==============================] - 0s 25us/step - loss: 1.6303 - val_loss: 1.6271
    Epoch 92/100
    560/560 [==============================] - 0s 25us/step - loss: 1.5268 - val_loss: 1.6519
    Epoch 93/100
    560/560 [==============================] - 0s 25us/step - loss: 1.6022 - val_loss: 1.7651
    Epoch 94/100
    560/560 [==============================] - 0s 29us/step - loss: 1.5626 - val_loss: 1.6491
    Epoch 95/100
    560/560 [==============================] - 0s 25us/step - loss: 1.5631 - val_loss: 1.6602
    Epoch 96/100
    560/560 [==============================] - 0s 21us/step - loss: 1.5351 - val_loss: 1.5712
    Epoch 97/100
    560/560 [==============================] - 0s 23us/step - loss: 1.4791 - val_loss: 1.5486
    Epoch 98/100
    560/560 [==============================] - 0s 27us/step - loss: 1.4728 - val_loss: 1.6777
    Epoch 99/100
    560/560 [==============================] - 0s 20us/step - loss: 1.5091 - val_loss: 1.6052
    Epoch 100/100
    560/560 [==============================] - 0s 27us/step - loss: 1.4672 - val_loss: 1.6242
    


```python
A = model.get_layer(index=0).get_weights()
a, b = A[0][0], A[1]
B = model.get_layer(index=2).get_weights()
c, d = B[0][:, 0], B[1][0]

def approximate(x):
    logit = np.zeros((len(a)))
    for i in range(len(a)):
        weighted_sum = a[i] * x + b[i]
        logit[i] = (weighted_sum + abs(weighted_sum))/2
    answer = sum([logit[i] * c[i] for i in range(len(c))]) + d
    return answer

interval = np.linspace(-5, 5, 101)
squared_interval = [approximate(x) for x in interval]
plt.plot(interval, squared_interval, 'g', interval, interval ** 2, 'r')
```




    [<matplotlib.lines.Line2D at 0x1ebb0668>,
     <matplotlib.lines.Line2D at 0x1ebb07b8>]




![png](https://raw.githubusercontent.com/riduan91/DSC111/master/Lesson1/Amphi/output_48_1.png)


**Using tanh function**


```python
model = Sequential()
model.add(Dense(12, input_shape = (1,), kernel_initializer=RandomUniform(-2, 2), bias_initializer=RandomUniform(-2, 2)))
model.add(Activation('tanh'))
model.add(Dense(1, kernel_initializer=RandomUniform(-2, 2), bias_initializer=RandomUniform(-2, 2)))
model.add(Activation('linear'))
model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01))
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=100, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
```

    Train on 560 samples, validate on 140 samples
    Epoch 1/100
    560/560 [==============================] - 2s 3ms/step - loss: 110.0394 - val_loss: 86.5305
    Epoch 2/100
    560/560 [==============================] - 0s 39us/step - loss: 73.0932 - val_loss: 63.6668
    Epoch 3/100
    560/560 [==============================] - 0s 50us/step - loss: 59.6977 - val_loss: 58.4199
    Epoch 4/100
    560/560 [==============================] - 0s 43us/step - loss: 53.9511 - val_loss: 53.5884
    Epoch 5/100
    560/560 [==============================] - 0s 80us/step - loss: 49.9636 - val_loss: 50.1778
    Epoch 6/100
    560/560 [==============================] - 0s 39us/step - loss: 46.5741 - val_loss: 46.5179
    Epoch 7/100
    560/560 [==============================] - 0s 43us/step - loss: 43.2523 - val_loss: 43.3287
    Epoch 8/100
    560/560 [==============================] - 0s 68us/step - loss: 40.4494 - val_loss: 40.5961
    Epoch 9/100
    560/560 [==============================] - 0s 30us/step - loss: 37.5799 - val_loss: 38.5105
    Epoch 10/100
    560/560 [==============================] - 0s 52us/step - loss: 35.3633 - val_loss: 35.7305
    Epoch 11/100
    560/560 [==============================] - 0s 34us/step - loss: 33.0787 - val_loss: 33.4582
    Epoch 12/100
    560/560 [==============================] - 0s 36us/step - loss: 30.8465 - val_loss: 31.4378
    Epoch 13/100
    560/560 [==============================] - 0s 32us/step - loss: 28.8071 - val_loss: 29.6073
    Epoch 14/100
    560/560 [==============================] - 0s 34us/step - loss: 26.7555 - val_loss: 27.7218
    Epoch 15/100
    560/560 [==============================] - 0s 54us/step - loss: 24.7998 - val_loss: 25.1130
    Epoch 16/100
    560/560 [==============================] - 0s 38us/step - loss: 22.8827 - val_loss: 23.4280
    Epoch 17/100
    560/560 [==============================] - 0s 41us/step - loss: 20.8998 - val_loss: 21.4451
    Epoch 18/100
    560/560 [==============================] - 0s 27us/step - loss: 19.0926 - val_loss: 19.5617
    Epoch 19/100
    560/560 [==============================] - 0s 25us/step - loss: 17.5095 - val_loss: 18.1781
    Epoch 20/100
    560/560 [==============================] - 0s 36us/step - loss: 16.2600 - val_loss: 16.8964
    Epoch 21/100
    560/560 [==============================] - 0s 29us/step - loss: 14.9885 - val_loss: 15.7119
    Epoch 22/100
    560/560 [==============================] - 0s 34us/step - loss: 13.8704 - val_loss: 14.8805
    Epoch 23/100
    560/560 [==============================] - 0s 41us/step - loss: 12.8733 - val_loss: 13.8503
    Epoch 24/100
    560/560 [==============================] - 0s 34us/step - loss: 11.9958 - val_loss: 12.9330
    Epoch 25/100
    560/560 [==============================] - 0s 29us/step - loss: 11.1555 - val_loss: 12.0145
    Epoch 26/100
    560/560 [==============================] - 0s 66us/step - loss: 10.3257 - val_loss: 11.3014
    Epoch 27/100
    560/560 [==============================] - 0s 34us/step - loss: 9.6058 - val_loss: 10.6630
    Epoch 28/100
    560/560 [==============================] - 0s 34us/step - loss: 8.9875 - val_loss: 9.9667
    Epoch 29/100
    560/560 [==============================] - 0s 30us/step - loss: 8.3680 - val_loss: 9.3476
    Epoch 30/100
    560/560 [==============================] - 0s 29us/step - loss: 7.8168 - val_loss: 8.7566
    Epoch 31/100
    560/560 [==============================] - 0s 32us/step - loss: 7.3757 - val_loss: 8.2614
    Epoch 32/100
    560/560 [==============================] - 0s 36us/step - loss: 6.8404 - val_loss: 8.0052
    Epoch 33/100
    560/560 [==============================] - 0s 34us/step - loss: 6.5734 - val_loss: 7.3856
    Epoch 34/100
    560/560 [==============================] - 0s 23us/step - loss: 6.0397 - val_loss: 7.1956
    Epoch 35/100
    560/560 [==============================] - 0s 38us/step - loss: 5.7567 - val_loss: 6.7886
    Epoch 36/100
    560/560 [==============================] - 0s 21us/step - loss: 5.4259 - val_loss: 6.3796
    Epoch 37/100
    560/560 [==============================] - 0s 23us/step - loss: 5.0920 - val_loss: 6.1057
    Epoch 38/100
    560/560 [==============================] - 0s 23us/step - loss: 4.8267 - val_loss: 5.7810
    Epoch 39/100
    560/560 [==============================] - 0s 30us/step - loss: 4.6344 - val_loss: 5.6356
    Epoch 40/100
    560/560 [==============================] - 0s 25us/step - loss: 4.3696 - val_loss: 5.2874
    Epoch 41/100
    560/560 [==============================] - 0s 25us/step - loss: 4.1399 - val_loss: 5.0562
    Epoch 42/100
    560/560 [==============================] - 0s 41us/step - loss: 3.9315 - val_loss: 4.9299
    Epoch 43/100
    560/560 [==============================] - 0s 23us/step - loss: 3.8046 - val_loss: 4.7003
    Epoch 44/100
    560/560 [==============================] - 0s 39us/step - loss: 3.5493 - val_loss: 4.4187
    Epoch 45/100
    560/560 [==============================] - 0s 38us/step - loss: 3.4053 - val_loss: 4.2661
    Epoch 46/100
    560/560 [==============================] - 0s 25us/step - loss: 3.2965 - val_loss: 4.0590
    Epoch 47/100
    560/560 [==============================] - 0s 21us/step - loss: 3.0976 - val_loss: 3.9188
    Epoch 48/100
    560/560 [==============================] - 0s 20us/step - loss: 2.9868 - val_loss: 3.9174
    Epoch 49/100
    560/560 [==============================] - 0s 32us/step - loss: 2.8952 - val_loss: 3.6447
    Epoch 50/100
    560/560 [==============================] - 0s 21us/step - loss: 2.8071 - val_loss: 3.5084
    Epoch 51/100
    560/560 [==============================] - 0s 23us/step - loss: 2.6576 - val_loss: 3.4458
    Epoch 52/100
    560/560 [==============================] - 0s 21us/step - loss: 2.5698 - val_loss: 3.3077
    Epoch 53/100
    560/560 [==============================] - 0s 25us/step - loss: 2.4527 - val_loss: 3.2147
    Epoch 54/100
    560/560 [==============================] - 0s 32us/step - loss: 2.4204 - val_loss: 3.1016
    Epoch 55/100
    560/560 [==============================] - 0s 23us/step - loss: 2.2754 - val_loss: 3.0979
    Epoch 56/100
    560/560 [==============================] - 0s 21us/step - loss: 2.2468 - val_loss: 2.9836
    Epoch 57/100
    560/560 [==============================] - 0s 21us/step - loss: 2.1863 - val_loss: 3.0163
    Epoch 58/100
    560/560 [==============================] - 0s 32us/step - loss: 2.1740 - val_loss: 2.8800
    Epoch 59/100
    560/560 [==============================] - 0s 27us/step - loss: 2.0896 - val_loss: 2.6460
    Epoch 60/100
    560/560 [==============================] - 0s 21us/step - loss: 2.0690 - val_loss: 2.7578
    Epoch 61/100
    560/560 [==============================] - 0s 21us/step - loss: 2.1290 - val_loss: 2.5498
    Epoch 62/100
    560/560 [==============================] - 0s 23us/step - loss: 1.8630 - val_loss: 2.5478
    Epoch 63/100
    560/560 [==============================] - 0s 29us/step - loss: 1.8118 - val_loss: 2.4953
    Epoch 64/100
    560/560 [==============================] - 0s 43us/step - loss: 1.9455 - val_loss: 2.9399
    Epoch 65/100
    560/560 [==============================] - 0s 25us/step - loss: 2.0665 - val_loss: 2.3253
    Epoch 66/100
    560/560 [==============================] - 0s 29us/step - loss: 1.6773 - val_loss: 2.2254
    Epoch 67/100
    560/560 [==============================] - 0s 23us/step - loss: 1.6695 - val_loss: 2.1638
    Epoch 68/100
    560/560 [==============================] - 0s 23us/step - loss: 1.6755 - val_loss: 3.1131
    Epoch 69/100
    560/560 [==============================] - 0s 34us/step - loss: 2.3310 - val_loss: 2.1421
    Epoch 70/100
    560/560 [==============================] - 0s 30us/step - loss: 1.5173 - val_loss: 2.0520
    Epoch 71/100
    560/560 [==============================] - 0s 30us/step - loss: 1.4787 - val_loss: 2.0450
    Epoch 72/100
    560/560 [==============================] - 0s 46us/step - loss: 1.4659 - val_loss: 1.9519
    Epoch 73/100
    560/560 [==============================] - 0s 23us/step - loss: 1.4142 - val_loss: 1.9759
    Epoch 74/100
    560/560 [==============================] - 0s 29us/step - loss: 1.3884 - val_loss: 1.9101
    Epoch 75/100
    560/560 [==============================] - 0s 25us/step - loss: 1.4044 - val_loss: 1.8590
    Epoch 76/100
    560/560 [==============================] - 0s 20us/step - loss: 1.5029 - val_loss: 1.8635
    Epoch 77/100
    560/560 [==============================] - 0s 21us/step - loss: 1.3653 - val_loss: 2.0549
    Epoch 78/100
    560/560 [==============================] - 0s 30us/step - loss: 1.4674 - val_loss: 1.9123
    Epoch 79/100
    560/560 [==============================] - 0s 21us/step - loss: 1.4495 - val_loss: 1.8487
    Epoch 80/100
    560/560 [==============================] - 0s 21us/step - loss: 1.5829 - val_loss: 2.3325
    Epoch 81/100
    560/560 [==============================] - 0s 30us/step - loss: 1.7693 - val_loss: 2.7065
    Epoch 82/100
    560/560 [==============================] - 0s 25us/step - loss: 2.3319 - val_loss: 1.6166
    Epoch 83/100
    560/560 [==============================] - 0s 18us/step - loss: 1.1822 - val_loss: 1.7620
    Epoch 84/100
    560/560 [==============================] - 0s 32us/step - loss: 1.1724 - val_loss: 1.7143
    Epoch 85/100
    560/560 [==============================] - 0s 29us/step - loss: 1.3063 - val_loss: 1.6526
    Epoch 86/100
    560/560 [==============================] - 0s 29us/step - loss: 1.7289 - val_loss: 1.7275
    Epoch 87/100
    560/560 [==============================] - 0s 30us/step - loss: 1.3503 - val_loss: 1.6254
    Epoch 88/100
    560/560 [==============================] - 0s 23us/step - loss: 1.5487 - val_loss: 1.4794
    Epoch 89/100
    560/560 [==============================] - 0s 29us/step - loss: 1.0794 - val_loss: 1.5054
    Epoch 90/100
    560/560 [==============================] - 0s 30us/step - loss: 1.1145 - val_loss: 1.4246
    Epoch 91/100
    560/560 [==============================] - 0s 25us/step - loss: 1.1069 - val_loss: 1.5510
    Epoch 92/100
    560/560 [==============================] - 0s 30us/step - loss: 1.0826 - val_loss: 1.5525
    Epoch 93/100
    560/560 [==============================] - 0s 36us/step - loss: 1.2846 - val_loss: 1.7323
    Epoch 94/100
    560/560 [==============================] - 0s 29us/step - loss: 1.0796 - val_loss: 1.4276
    Epoch 95/100
    560/560 [==============================] - 0s 27us/step - loss: 1.0051 - val_loss: 1.5081
    Epoch 96/100
    560/560 [==============================] - 0s 21us/step - loss: 0.9864 - val_loss: 1.3673
    Epoch 97/100
    560/560 [==============================] - 0s 21us/step - loss: 1.0620 - val_loss: 1.5531
    Epoch 98/100
    560/560 [==============================] - 0s 25us/step - loss: 1.0757 - val_loss: 1.3313
    Epoch 99/100
    560/560 [==============================] - 0s 20us/step - loss: 0.9661 - val_loss: 1.2292
    Epoch 100/100
    560/560 [==============================] - 0s 29us/step - loss: 0.9626 - val_loss: 1.2537
    


```python
A = model.get_layer(index=0).get_weights()
a, b = A[0][0], A[1]
B = model.get_layer(index=2).get_weights()
c, d = B[0][:, 0], B[1][0]

def approximate(x):
    logit = np.zeros((len(a)))
    for i in range(len(a)):
        weighted_sum = a[i] * x + b[i]
        logit[i] = np.tanh(weighted_sum)
    answer = sum([logit[i] * c[i] for i in range(len(c))]) + d
    return answer

interval = np.linspace(-5, 5, 101)
squared_interval = [approximate(x) for x in interval]
plt.plot(interval, squared_interval, 'g', interval, interval ** 2, 'r')
```




    [<matplotlib.lines.Line2D at 0x1ed7f6d8>,
     <matplotlib.lines.Line2D at 0x1ed7f828>]




![png](https://raw.githubusercontent.com/riduan91/DSC111/master/Lesson1/Amphi/output_51_1.png)


# 5. Gradient-based Method for Training

Many iterative methods can be used to solve the optimization problem, like Gradient Descent and Schochastic Gradient Descent (SGD). Some of them needs the first order derivative (gradient) of the cost w.r.t. the paramters (i.e., weights and bias).

These derivatives can be calcuated from the higher layers, then back to lower layers. This technique is called **backpropagation**.

**Example for 2-layer feedforward network for regression**

<img src="https://raw.githubusercontent.com/riduan91/DSC111/master/Lesson1/Amphi/F3.png" width=800></img>

- Cost function: 
$$
L(\mathbf x_{(1)}, \ldots, \mathbf x_{(N)}, t_{(1)}, \ldots, t_{(N)}) = \sum_{n} E_n := \sum_{n} \frac12 |y_{(n)} - t_{(n)}|^2
$$

We ignore the index $n$ for short.

- We need to calculate 

$$\frac{\partial E}{\partial v_{k,m}}$$ for $k=1, \ldots, K; m=0, \ldots, M$ 
and 
$$\frac{\partial E}{\partial w_{m,d}}$$ for $m = 1, \ldots, M; d = 0, \ldots, D$.

**Second layer**

We have
$$
\frac{\partial E}{\partial v_{k,m}}(\mathbf w, \mathbf v)  = \frac{\partial E}{\partial b_k} (\mathbf y) \frac{\partial b_k}{\partial v_{k,m}}(\mathbf w, \mathbf v) 
$$

Note that

$$
\frac{\partial E}{\partial b_k} = b_k - t_k = y_k - t_k
$$

and

$$
\frac{\partial b_k}{\partial v_{k,m}} = z_m
$$

So:
$$
\frac{\partial E}{\partial v_{k,m}}(\mathbf w, \mathbf v) = (y_k - t_k)z_m
$$

**First layer**

We have
$$
\frac{\partial E}{\partial w_{m,d}} = \frac{\partial E}{\partial a_m} \frac{\partial a_m}{\partial w_{m,d}}
$$

Note that
$$
\frac{\partial a_m}{\partial w_{m,d}} = x_d
$$

and
$$
\frac{\partial E}{\partial a_m} =  \frac{\partial z_m}{\partial a_m} \frac{\partial E}{\partial z_m} = h' \cdot \sum_{k=1}^K v_{k,m} \frac{\partial E}{\partial b_k}
$$

So
$$
\frac{\partial E}{\partial w_{m,d}} = x_d h' \cdot \sum_{k=1}^K v_{k,m} \frac{\partial E}{\partial b_k}
$$

We see that the values $\frac{\partial E}{\partial b_k}$ of the second layer are first calculated and then propagate to the first layer.

**Complexity**

The algorithm requires *$O(iW)$* where $i$ is the number of iterations, $W$ is the total number of paramaters.

# References

[1] C. Bishop, *Pattern Recognition and Machine Learning*  
[2] I. Goodfellow, Y. Bengio, A.Courville, *Deep Learning*
