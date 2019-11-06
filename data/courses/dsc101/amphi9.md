
# Amphi 9 - Support Vector Machines. Introduction to Kernel Method

# 1. Support Vector Machine for Binary Classification - Linear Separable Case

## 1.1. Remind: Logistic Regression

In Logistic Regression for binary classification, we suppose that the boundary is a hyperplan of form $\mathbf w \cdot \mathbf x + b = 0$ and try to find $\mathbf w$, $b$ minimizes:

$$
L(\mathbf w, b, \mathbf X, \mathbf y) = \sum_{n=1}^N l(\mathbf w, b, \mathbf x_n, y_n) 
$$

where $\mathbf X = (\mathbf x_1, \ldots, \mathbf x_N)$, $\mathbf y=(y_1, \ldots, y_N)$ denote the training set, and

$$
l(\mathbf w, b, \mathbf x, y) = y \log \left(\frac1{1 + \exp(-\mathbf w\cdot \mathbf x - b)}\right) + (1-y) \log \left(\frac1{1 + \exp(\mathbf w \cdot \mathbf x + b)}\right)
$$

Remind that when $\frac1{1 + \exp(-\mathbf w\cdot x - b)}$ is the probability that the output is 1 predicted by the logistic regression classifier, and $\frac1{1 + \exp(\mathbf w \cdot x + b)} = 1 - \frac1{1 + \exp(-\mathbf w\cdot x - b)}$ is the probability of output 0. The minus log logit function $-\log\left(\frac1{1+\exp(-z)}\right)$ is used as a "favorizer" of positivity of the argument $z$ because it will attain a value near 0 if $z >> 0$ and very large if $z << 0$, while $-\log\left( \frac1{1+\exp(z)}\right)$ is used as a "favorizer" of negativity of the argument.


```python
%matplotlib inline
import numpy as np, matplotlib.pyplot as plt

def minusloglogit(z):
    return -np.log(1 / (1 + np.exp(-z)))

def minusloglogit0(z):
    return -np.log(1 / (1 + np.exp(z)))

X = np.linspace(-5, 5, 100)
plt.plot(X, minusloglogit(X), 'r')
plt.plot(X, minusloglogit0(X), 'b')
```




    [<matplotlib.lines.Line2D at 0x549bf28>]




![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/Amphi/output_3_1.png)



```python
import pandas as pd

data = pd.read_csv("Example2.txt", sep=";", header=None)
red_data = data[data[2] == 0]
blue_data = data[data[2] == 1]

plt.scatter(red_data[0], red_data[1], color='red', marker='o')
plt.scatter(blue_data[0], blue_data[1], color='blue', marker='x')
```




    <matplotlib.collections.PathCollection at 0x92dbc18>




![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/Amphi/output_4_1.png)



```python
from sklearn.linear_model import LogisticRegression
X = data.values[:,:2]
y = data.values[:,2]

clf = LogisticRegression()
clf.fit(X, y)

plt.scatter(red_data[0], red_data[1], color='red', marker='o')
plt.scatter(blue_data[0], blue_data[1], color='blue', marker='x')
my_range = np.arange(-2, 2, 0.1)
w = clf.coef_
print(w)
b = clf.intercept_
plt.plot(np.arange(-2, 2, 0.1), -w[0,0]/w[0,1]*my_range - b/w[0,1], 'g')
```

    [[2.76362881 1.98566284]]
    




    [<matplotlib.lines.Line2D at 0xb2d66d8>]




![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/Amphi/output_5_2.png)


## 1.2 Margin

We are interested in the notion of "margin". Margin is distance between the nearest point to the boundary. In the first picture (logistic regression), the margin is large. In the second one, it is much smaller. In general, we usually prefer classifier possessing large margins. For example, if a new orange point appears that is near to the red cluster, we would rather classify it as a red point than a blue one. This is true if the margin is large in the first case and false in the other case. 

<img src="https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/Amphi/F1.png" width=600></img>

We would like to find a classifier that maximizes the sum of margins of 2 classes.

## 1.3 The Optimization Problem

If a hyperplan $\mathbf w \cdot \mathbf x + b = 0$ separates correctly two classes of the training set, the distance from $\mathbf x$ (in the training set) to this hyperplan is

$$
\frac{\vert \mathbf w \cdot \mathbf x + b\vert}{\Vert w \Vert} = \begin{cases}
\frac{\left( \mathbf w \cdot \mathbf x + b\right)}{\Vert w \Vert}, \qquad y = 1\\
-\frac{\left( \mathbf w \cdot \mathbf x + b\right)}{\Vert w \Vert}, \qquad y = 0
\end{cases}\\
= (2y-1)\frac{\left( \mathbf w \cdot \mathbf x + b\right)}{\Vert w \Vert}
$$

Denote $2y_n - 1 =: \tilde y_n$ for short (which is $\pm 1$)

So the maximum margin solution is found by solving
$$
\arg\max_{\mathbf w, b} \left( \frac1{\Vert \mathbf w\Vert} \min_n \tilde y_n(\mathbf w \cdot \mathbf x_n + b)\right)
$$

where $\min_n$ denotes minimum on the training set.

This problem is complex to solve. Note that the hyperplan doesn't change if we scale $\mathbf w, b$ by a coefficient, we can set a constraint on $\mathbf w, b$ that $\Vert w \Vert$ equals the margin, i.e:

$$
\tilde y_n (\mathbf w \cdot \mathbf x_n + b) = 1
$$

for closest point to the hyperplan. Now all data points in the training set will satisfy:

$$
\tilde y_n (\mathbf w \cdot \mathbf x_n + b) \geq 1,\qquad n = 1, \ldots N
$$

The maximization problem now becomes:

$$
\arg\max_{\mathbf w, b} \frac1{\Vert \mathbf w\Vert}
$$

where $$\tilde y_n (\mathbf w \cdot \mathbf x_n + b) \geq 1,\qquad n = 1, \ldots N $$ and the equality holds for at least one data point. We see that the requirement that equality holds will always automatically satisfy (otherwise a smaller $\Vert \mathbf w \Vert$ still satisfies all the constraints), hence we ignore it from the optimization problem. Besides, maximizing $\frac1{\Vert \mathbf w \Vert}$ is equivalent to minimizing $\Vert \mathbf w \Vert^2$. The problem becomes:

$$
\arg\min_{\mathbf w, b} \frac12 \Vert \mathbf w \Vert^2
$$

subject to
$$
\tilde y_n (\mathbf w \cdot \mathbf x_n + b) \geq 1,\qquad n = 1, \ldots N
$$

(coefficient $\frac12$ will be convenient for later uses).

## 1.4 Lagrange Method and The Dual Representation

By Lagrange method, we can introduce Lagrange multipliers $a_n \geq 0$

$$
L(\mathbf w, b, \mathbf a) = \frac12 \Vert \mathbf w \Vert^2 - \sum_{n=1}^N a_n \left( \tilde y_n (\mathbf w \cdot \mathbf x_n + b) - 1 \right)
$$

Setting the derivatives of $L(\mathbf w, b, \mathbf a)$ w.r.t $\mathbf w, b$ equal to 0, we get:

$$
\mathbf w = \sum_{n=1}^N a_n \tilde y_n \mathbf x_n, 
$$

$$
\sum_{n=1}^N a_n \tilde y_n = 0
$$

We get
$$
\Vert \mathbf w \Vert^2 = \sum_{n=1}^N\sum_{m=1}^N a_n a_m \tilde y_n \tilde y_m \mathbf x_n \cdot \mathbf x_m
$$

The dual representation of the maximum margin problem becomes:
$$
\arg\max_{\mathbf a}\tilde L (\mathbf a) = \arg\max_{a}\sum_{n=1}^N a_n - \frac12 \sum_{n=1}^N \sum_{m=1}^N a_n a_m \tilde y_n \tilde y_m \mathbf x_n \cdot \mathbf x_m
$$
s.t.
$$
a_n \geq 0, \qquad n = 1, \ldots, N \\
\sum_{n=1}^N a_n \tilde y_n = 0
$$

This is a quadratic programming problem and can be solved in closed form. (https://en.wikipedia.org/wiki/Quadratic_programming)

## 1.5 Support Vectors. Prediction

Note that the solution of this satisfies the KKT conditions, which requires for each $n = 1, \ldots, N$:

$$
a_n \geq 0 \\
\tilde y_n (\mathbf w \cdot \mathbf x_n + b) - 1\geq 0 \\
a_n (\tilde y_n (\mathbf w \cdot \mathbf x_n + b) - 1) = 0
$$

The points satisfying $\tilde y_n (\mathbf w \cdot \mathbf x_n + b) - 1 = 0$ are the nearest points to the boundary and called **support vectors**. The points where $a_n = 0$ are not support vectors and have no role in later prediction. Indeed, prediction can be done by calculating

$$
\mathbf w \cdot \mathbf x + b = \sum_{n=1}^N a_n \tilde y_n \mathbf x_n \cdot \mathbf x + b = \sum_{m \in \mathcal S} a_n \tilde y_m \mathbf x_m \cdot \mathbf x + b
$$

where $\mathcal S$ denotes the set of indices of support vectors, and classification is based on sign of this quantity.

To determine $b$, use
$$
\tilde y_n \left( \sum_{m \in \mathcal S} a_m \tilde y_m \mathbf x_m \cdot \mathbf x_n + b \right) = 1
$$

for all $n \in \mathcal S$. Multiplying both side of these equations by $\tilde y_n^2$, we have

$$
 \sum_{m \in \mathcal S} a_m \tilde y_m \mathbf x_m \cdot \mathbf x_n + b  = \tilde y_n
$$



We finally find out:
$$
b = \frac1{|\mathcal S|} \sum_{n \in \mathcal S} \left( \tilde y_n - \sum_{m \in \mathcal S} a_m \tilde y_m \mathbf x_m \cdot \mathbf x_n \right)
$$

After solving the problem and find out $\mathbf w, b$, the hyperplan $\mathbf w \cdot \mathbf x + b = 0$ will be called the **decision boundary** while $\mathbf w \cdot \mathbf x + b = \pm 1$ will be called **margin boundaries**.

## 1.6 ERM Representation

Support vector machine for linear separable case is represented by the optimization problem:

$$
\arg\min_{\mathbf w, b} \Vert \mathbf w \Vert^2
$$

s.t
$$
\tilde y_n (\mathbf w \cdot \mathbf x_n + b) \geq 1
$$

This problem is equivalent to:
$$
\arg\min_{\mathbf w, b} E_{\infty} \left( \tilde y_n (\mathbf w \cdot \mathbf x_n + b) - 1 \right) + \lambda \Vert w\Vert^2
$$

where $E_{\infty}(z) = 0$ if $z \geq 0$ and $+\infty$ if $z < 0$.

As an ERM representation, SVM optimization problem can be expressed by

$$
\hat y = \mathbf w \cdot \mathbf x + b
$$

$$
l(y, \hat y) = E_{\infty}(y \hat y - 1)
$$

and regularization $\Vert \mathbf w \Vert^2$. 


```python
%matplotlib inline
import numpy as np, matplotlib.pyplot as plt, sys

INFINITY = 10

def infinityLoss(z):
    if z >= 1:
        return 0
    return INFINITY

def minusInfinityLoss(z):
    if z <= -1:
        return 0
    return INFINITY

myspace = np.linspace(-5, 5, 100)
plt.plot(myspace, minusloglogit(myspace), 'r--', label="logistic")
plt.plot(myspace, minusloglogit0(myspace), 'g--', label="logistic")
plt.plot(myspace, [infinityLoss(z) for z in myspace], 'r-', label="SVM")
plt.plot(myspace, [minusInfinityLoss(z) for z in myspace], 'g-', label="SVM")
plt.legend()
```




    <matplotlib.legend.Legend at 0xb354710>




![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/Amphi/output_18_1.png)


## 1.7 Implementation

** In case of linear separability, use large C for sklearn implementation **


```python
from sklearn.svm import SVC
clf = SVC(kernel='linear', C=1000)
clf.fit(X, y)
```




    SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)



**Boundary**


```python
w = clf.coef_
w
```




    array([[1.65488693, 1.85627221]])




```python
b = clf.intercept_
b
```




    array([0.1118497])




```python
plt.scatter(red_data[0], red_data[1], color='red', marker='o')
plt.scatter(blue_data[0], blue_data[1], color='blue', marker='x')
my_range = np.arange(-2, 2, 0.1)
plt.plot(np.arange(-2, 2, 0.1), -w[0,0]/w[0,1]*my_range - b/w[0,1], 'g')
```




    [<matplotlib.lines.Line2D at 0xb3c1978>]




![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/Amphi/output_25_1.png)


**Support vectors**


```python
clf.support_
```




    array([  0, 116, 153])




```python
clf.support_vectors_
```




    array([[-0.6135, -0.0521],
           [-0.1585,  0.6198],
           [ 0.7992, -0.234 ]])




```python
plt.scatter(red_data[0], red_data[1], color='red', marker='o')
plt.scatter(blue_data[0], blue_data[1], color='blue', marker='x')
my_range = np.arange(-2, 2, 0.1)
# Decision boundary wx + b = 0
plt.plot(np.arange(-2, 2, 0.1), -w[0,0]/w[0,1]*my_range - b/w[0,1], 'g')
# Margin boundary wx + b = -1
plt.plot(np.arange(-2, 2, 0.1), -w[0,0]/w[0,1]*my_range - (b + 1)/w[0,1], 'orange')
# Margin boundary wx + b = 1
plt.plot(np.arange(-2, 2, 0.1), -w[0,0]/w[0,1]*my_range - (b - 1)/w[0,1], 'skyblue')
plt.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1], color='green', marker='o')
```




    <matplotlib.collections.PathCollection at 0xb447400>




![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/Amphi/output_29_1.png)


**Dual coefficients ($a_n$)**


```python
clf.dual_coef_
```




    array([[-3.09188504,  2.83284856,  0.25903648]])



# 2. Support Vector Machine for Binary Classification. Non Linear-Separable Case

## 2.1 The Optimization Problem

Using the same minimization problem as in the case of linear-separability, a mis-classified point will lead to infinity loss for ERM representation in 1.6. A not linearly separable dataset will give no solution to the constraint satisfaction problem (CSP) in 1.3. Hence, if we are dealing with data that are not linearly separable, we must modify the CSP (hence modify the loss function in ERM representation).

We will introduce the variables $\xi_n$, $n = 1, \ldots, N$ that modelize penalty for miss classified data points. If the data point is on the right side and inside the supported zone (i.e, has greater distance to the boundary than the support vectors), we will have $\xi_n  = 0$, otherwise $\xi_n$ will be proportional to the distance to the margin boundary of the correct side. $\xi_n$ are called slack variables. The linear-separable case is also called **soft margin** case.

We would prefer $\xi_n$ not too large, so we want $\sum_{n=1}^N \xi_n$ to be small. The problem now becomes:

$$
\arg\min_{\mathbf w, b} \frac12 \Vert w\Vert^2 + C\sum_{n=1}^N \xi_n
$$

s.t.

$$
\xi_n \geq 0, \qquad n = 1, \ldots, N \\
\tilde y_n (\mathbf w \cdot \mathbf x + b) \geq 1 - \xi_n
$$

where $C>0$ is a penalization coefficient that controls the trade-off between the slack variables and the margin.

## 2.2 The Dual Representation

Similarly to the linearly-separable case, we can introduce the Lagrangian

$$
L(\mathbf w, b, \mathbf a, \mathbf \mu) = \frac12 \Vert w\Vert^2 + C\sum_{n=1}^N \xi_n - \sum_{n=1}^N a_n(\tilde y_n (\mathbf w \cdot \mathbf x + b) - 1 + \xi_n) - \sum_{n=1}^N \mu_n \xi_n
$$.

The KKT conditions become:
$$
a_n \geq 0 \\
\tilde y_n (\mathbf w \cdot \mathbf x + b) - 1 + \xi_n \geq 0 \\
a_n(\tilde y_n (\mathbf w \cdot \mathbf x + b) - 1 + \xi_n) = 0 \\
\mu_n \geq 0 \\
\xi_n \geq 0 \\
\mu_n \xi_n = 0 \\
$$

where $n = 1, \ldots, N$.

Let $\partial L/\partial \mathbf w, \partial L/\partial b, \partial L/\partial \xi_n = 0 $ we have

$$
\mathbf w = \sum_{n=1}^N a_n \tilde y_n \mathbf x_n\\
\sum_{n=1}^N a_n \tilde y_n = 0 \\
a_n = C - \mu_n
$$

The dual representation becomes:

$$
\arg\max_{\mathbf a, \mathbf \mu} \sum_{n=1}^N a_n - \frac12 \sum_{n=1}^N \sum_{m=1}^N a_n a_m \tilde y_n \tilde y_m \mathbf x_n \cdot \mathbf x_m
$$

s.t.
$$
a_n \geq 0 \\
\mu_n \geq 0 \\
\sum_{n=1}^N a_n \tilde y_n = 0 
$$

Using $a_n = C - \mu_n$, we get the dual representation depending only on $\mathbf a$:

$$
\arg\max_{\mathbf a} \sum_{n=1}^N a_n - \frac12 \sum_{n=1}^N \sum_{m=1}^N a_n a_m \tilde y_n \tilde y_m \mathbf x_n \cdot \mathbf x_m
$$

s.t.
$$
0 \leq a_n \leq C \\
\sum_{n=1}^N a_n \tilde y_n = 0 
$$

The conditions $0 \leq a_n \leq C$ are called the box constraints. This is a quadratic programming problem.

## 2.3 Solution. Prediction

- Points corresponding to $a_n = 0$ can be discarded, they have no role to the prediction.
- Points corresponding to $a_n = C$ can have $\xi_n > 0$. They can lie inside the margin and can either be correctly classified if $\xi_n \leq 1$ or mis-classified if $\xi_n > 1$.
- Points corresponding to $0 < a_n < C$ have $\xi_n = 0$, they belong to $\tilde y_n(\mathbf w \cdot \mathbf x_n + b) = 1$ (the margin).

<img src="https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/Amphi/F2.png"></img>

Let $\mathcal S$ be the set of indices of points corresponding to $a_n \geq 0$ (type 2, 3). They are called support vectors.

Let $\mathcal M$ be the set of indices of points corresponding to $0 < a_n < C$ (type 3). They lie on the margins.

$b$ can be calculated using
$$
\mathbf w \cdot \mathbf x_n + b = \tilde y_n, \qquad n \in \mathcal M
$$

$$
\Leftrightarrow y\mathbf \sum_{m \in \mathcal S} a_m \tilde y_m \mathbf x_m\cdot \mathbf x_n + b = \tilde y_n, \qquad n \in \mathcal M
$$

Sum up to give
$$
b = \frac1{|\mathcal M|}\sum_{n \in \mathcal M} \left( \tilde y_n - \sum_{m \in \mathcal S} a_m \tilde y_m \mathbf x_m \cdot \mathbf x_n \right)
$$


**Prediction** can be done by calculating

$$
\mathbf w \cdot \mathbf x + b = \sum_{n=1}^N a_n \tilde y_n \mathbf x_n \cdot \mathbf x + b = \sum_{m \in \mathcal S} a_n \tilde y_m \mathbf x_m \cdot \mathbf x + b
$$

and classification is made using this quantity's sign.

## 2.4 ERM Representation

Recall the CSP representation of the problem:

$$
\arg\min_{\mathbf w, b} \frac12 \Vert w\Vert^2 + C\sum_{n=1}^N \xi_n
$$

s.t.

$$
\xi_n \geq 0, \qquad n = 1, \ldots, N \\
\tilde y_n (\mathbf w \cdot \mathbf x + b) \geq 1 - \xi_n
$$

The problem can be represented as

$$
\arg\min_{\mathbf w, b} \frac1{2C} \Vert w\Vert^2 + \sum_{n=1}^N \left[1 - \tilde y_n(\mathbf w \cdot \mathbf x + b) \right]_+
$$

As an ERM representation, SVM optimization problem can be expressed by

$$
\hat y = \mathbf w \cdot \mathbf x + b
$$

$$
l(y, \hat y) = E_{SV} (y\hat y) = \left[1 - y\hat y \right]_+
$$

and regularization $\Vert \mathbf w \Vert^2$, where $a_+$ denotes the positive part of $a$. $E_{SV}$ is called the **hinge** error function.

To compare with Logistic Regression.


```python
%matplotlib inline
import numpy as np, matplotlib.pyplot as plt, sys

INFINITY = 10

def HingeLoss(z):
    return ((1 - z) + abs(1 - z)) / 2

def HingeLossMinus(z):
    return ((1 + z) + abs(1 + z)) / 2    

myspace = np.linspace(-5, 5, 100)
plt.plot(myspace, minusloglogit(myspace), 'r--', label="logistic")
plt.plot(myspace, minusloglogit0(myspace), 'g--', label="logistic")
plt.plot(myspace, [HingeLoss(z) for z in myspace], 'r-', label="SVM")
plt.plot(myspace, [HingeLossMinus(z) for z in myspace], 'g-', label="SVM")
plt.legend()
```




    <matplotlib.legend.Legend at 0xb4ac438>




![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/Amphi/output_37_1.png)


## 2.5 Implementation


```python
data = pd.read_csv("Example1.txt", sep=";", header=None)
red_data = data[data[2] == 0]
blue_data = data[data[2] == 1]

plt.scatter(red_data[0], red_data[1], color='red', marker='o')
plt.scatter(blue_data[0], blue_data[1], color='blue', marker='x')

X = data.values[:,:2]
y = data.values[:,2]
```


![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/Amphi/output_39_0.png)



```python
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
clf = SVC(kernel='linear', C=1)
clf.fit(X, y)
```




    SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)




```python
w = clf.coef_
w
```




    array([[0.5194304 , 1.02593164]])




```python
b = clf.intercept_
b
```




    array([-0.20002904])




```python
clf.support_
```




    array([  0,   1,  15,  16,  17,  18,  20,  21,  22,  25,  26,  29,  31,
            32,  33,  35,  40,  43,  44,  51,  52,  55,  58,  59,  60,  65,
            67,  70,  72,  73,  75,  82,  86,  92,  96,  99, 100, 107, 110,
           111, 112, 115, 116, 118, 124, 125, 128, 130, 131, 139, 141, 145,
           147, 153, 157, 158, 161, 165, 166, 172, 175, 176, 183, 184, 189,
           190, 191, 192, 193, 196, 197, 198])




```python
y_pred = clf.predict(X)
y_pred
```




    array([1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           1., 1., 1., 1., 1., 0., 1., 1., 0., 1., 1., 1., 1., 0., 0., 1., 0.,
           1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1.,
           1., 1., 1., 0., 1., 0., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1.,
           0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1.,
           1., 1., 0., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1.,
           1., 1., 0., 1., 0., 1., 1., 1., 1., 0., 1., 0., 1.])




```python
np.array([i for i in range(len(X)) if y_pred[i] != y[i]])
# Should be a subset of support vectors
```




    array([  0,   1,  15,  21,  33,  40,  52,  70,  73,  75, 100, 107, 110,
           115, 116, 118, 125, 130, 139, 141, 145, 153, 166, 172, 175, 183,
           184, 189, 191, 196, 198])




```python
clf.decision_function(X)
```




    array([ 1.53772801,  0.47617821, -2.29633056, -2.83132447, -4.06124059,
           -1.74790965, -3.39517212, -2.49745817, -2.93063795, -1.95013572,
           -2.04154003, -3.43219497, -1.01055419, -1.68308215, -2.70938442,
            0.03345135, -0.23281741, -0.67082176, -0.98362899, -3.70268763,
           -0.53029246,  0.42999986, -0.99973763, -2.5039206 , -1.34404744,
           -0.66292178, -0.99479446, -2.38888744, -1.75999987, -0.37073443,
           -1.88325732, -0.67408313, -0.84526404,  0.44406388, -1.3959584 ,
           -0.82246806, -1.3227039 , -3.97166017, -2.67079346, -1.21924389,
            1.31522999, -1.6917555 , -1.64995968, -0.78624153, -0.33062044,
           -1.96688394, -1.4443731 , -2.21304322, -2.48888593, -4.2664121 ,
           -2.00722918, -0.41430772,  0.73103757, -1.79798438, -2.27343442,
           -0.43733609, -1.5501279 , -2.73682815, -0.9165661 , -0.0245055 ,
           -0.76973515, -1.76670486, -1.94928495, -1.77202989, -3.59490234,
           -0.12990927, -4.17351346, -0.53505659, -1.96913542, -1.56965186,
            0.05257333, -2.14842858, -0.88974494,  0.59455567, -1.25898892,
            0.36363509, -2.28476857, -2.41545394, -4.6933418 , -1.65295754,
           -1.37270267, -3.22557081, -0.15375878, -4.70643279, -4.86903536,
           -2.23597936, -0.89539217, -4.33059599, -2.25211855, -1.61851867,
           -2.32027194, -1.11788939, -0.91971892, -2.15046658, -1.07214582,
           -2.58868021, -0.08539911, -1.98915278, -3.10455168, -0.99973813,
           -0.02227702,  4.30194125,  2.59532316,  1.839658  ,  3.29497589,
            2.05245642,  1.87814753, -0.60649121,  2.38985247,  1.39749333,
           -0.03868324,  0.86712011,  0.999476  ,  3.67544809,  3.54113484,
           -0.53205916, -2.63851302,  3.30390469, -0.31554492,  2.02138996,
            1.90095005,  1.97884494,  1.97817193,  1.55972441,  0.86589122,
           -0.24571988,  3.83156806,  1.50271162,  0.19016602,  4.71097202,
           -0.76717506,  0.4288247 ,  4.4553455 ,  2.2271149 ,  2.38472791,
            1.74153613,  2.4794733 ,  1.22644753,  3.94293955, -2.12593954,
            2.39523422, -1.76661568,  2.01720001,  3.70400397,  2.66847241,
           -0.26936632,  1.34600356,  0.50067506,  2.38880381,  1.53890282,
            1.04054817,  3.5390866 ,  3.48350373, -2.34107034,  2.40997628,
            1.00025182,  1.06518578,  0.30508267,  0.30426859,  1.31609393,
            2.46447271,  0.54638711,  2.08101346,  2.09818355,  5.15163027,
            0.62679182, -0.01463805,  1.17740262,  2.4428449 ,  1.69771357,
            2.05914576,  3.57317548, -0.6592966 ,  3.59436732,  1.10716492,
           -0.39406022,  0.82138202,  2.29092998,  2.05143649,  2.89718351,
            2.33374123,  2.22594262,  2.94725938, -0.11277424, -1.31025776,
            1.93154451,  1.21974879,  1.87829452,  1.91218892, -0.40312404,
            0.85977376, -0.07999896,  0.6887286 ,  0.45242605,  2.76629405,
            2.97627593, -0.3231674 ,  0.0752545 , -0.40936704,  3.67441854])




```python
plt.scatter(red_data[0], red_data[1], color='red', marker='o')
plt.scatter(blue_data[0], blue_data[1], color='blue', marker='x')
my_range = np.arange(-6, 6, 0.1)
# Decision boundary wx + b = 0
plt.plot(my_range, -w[0,0]/w[0,1]*my_range - b/w[0,1], 'g')
# Margin boundary wx + b = -1
plt.plot(my_range, -w[0,0]/w[0,1]*my_range - (b + 1)/w[0,1], 'orange')
# Margin boundary wx + b = 1
plt.plot(my_range, -w[0,0]/w[0,1]*my_range - (b - 1)/w[0,1], 'skyblue')
```




    [<matplotlib.lines.Line2D at 0xb58ce10>]




![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/Amphi/output_47_1.png)


# 3. Support Vector Machine for Multiclass Classification

SVM can be generalized to multiclass classification by one of the known strategies:
- OVR (One vs Rest)
- OVO (One vs One)
- Hierarchical Strategies

# 4. Support Vector Machine for Regression



## 4.1 The $\epsilon$-insensitive Error Function

In linear regression, the loss function being used is:

$$
\mathbf l(y, y') = \mathbf (y - y')^2 
$$

In regularization case, this becomes:
$$
L(y, y') = \sum_{n=1}^N |y_n - y_n'|^2 + C\Vert \mathbf w \Vert^2
$$

We introduce a new error function that allows some insensitive error $\epsilon$ as follows: if the difference between the real and the prediction value does not exceed $\epsilon$, we accept that the loss can be ignored.

$$
l_{\epsilon} (y, y') =
\begin{cases}
0, if \qquad |y - y'| < \epsilon \\
|y - y'| - \epsilon, if \qquad |y - y'| \geq \epsilon
\end{cases}
$$

So, by introducing the $\epsilon-$insensitive error function, we want to find $\mathbf w, b$ that minimize
$$
L_{\epsilon}(\mathbf w, b) = \sum_{n=1}^N l_{\epsilon} (y_n, \mathbf w \cdot \mathbf x_n + b) + \lambda \Vert w \Vert^2 = \sum_{n=1}^N [|\mathbf w \cdot \mathbf x_n + b - y_n| - \epsilon]_+ + \lambda \Vert w \Vert^2
$$

This is the ERM-representation of the support vector machine method for regression (**SVR**).


```python
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

epsilon = 0.25
X = np.linspace(-2, 2, 100)

def square_loss(x):
    return x**2

def epsilon_insensitive_loss(x, epsilon):
    return np.maximum(abs(x) - epsilon, 0)

plt.plot(X, square_loss(X), 'r', label="square loss")
plt.plot(X, epsilon_insensitive_loss(X, epsilon), 'b', label='epsilon-insensitive loss')
plt.legend()
```




    <matplotlib.legend.Legend at 0xb5f9390>




![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/Amphi/output_51_1.png)


## 4.2 The Optimization Problem

For each data point $(\mathbf x_n, y_n)$, we introduce a slack variable $\xi_n \geq 0$ and a slack variable $\hat \xi_n \geq 0$ where

- $\xi_n > 0$ corresponds to a point where $y_n > \mathbf w \cdot \mathbf x_n + b + \epsilon$
- $\hat \xi_n > 0$ corresponds to a point where $y_n < \mathbf w \cdot \mathbf x_n + b - \epsilon$
- $\xi_n = 0$ and $\hat \xi_n = 0$ corresponds to a point where $\mathbf w \cdot \mathbf x_n + b - \epsilon \leq y_n \leq \mathbf w \cdot \mathbf x_n + b + \epsilon$

The constraints favorizing $y_n$ between $(\mathbf w \cdot \mathbf x_n) + b \pm \epsilon$, as in the classification problem, can now be reformulated as:
$$
\mathbf w \cdot \mathbf x_n - \epsilon - \hat \xi_n \leq y_n \leq \mathbf w \cdot \mathbf x_n + \epsilon
$$

And the error function becomes:
$$
\sum_{n=1}^N (\xi + \hat \xi) + \lambda \Vert w \Vert^2
$$

By scaling, it is equivalent to minimizing the function:
$$
\frac12 \Vert w \Vert^2 + C\sum_{n=1}^N(\xi_n + \hat \xi_n)
$$

s.t
$$
\xi_n \geq 0 \\
\hat \xi_n \geq 0 \\
\mathbf w \cdot \mathbf x_n + b + \xi_n + \epsilon - y_n \geq 0\\
\mathbf w \cdot \mathbf x_n + b - \xi_n - \epsilon - y_n \leq 0\\
$$

This is the CSP problem for SVR.

## 4.3 Solution to the Optimization Problem

By introducing the Lagrange coefficients, as in classification case, $a_n \geq 0$, $\hat a_n \geq 0$, $\mu_n \geq 0$, $\hat \mu_n \geq 0$, we have the Lagrangian

$$
L = C\sum_{n=1}^N (\xi_n + \hat \xi_n) + \frac12 \Vert w \Vert^2 - \sum_{n=1}^N (\mu_n \xi_n + \hat \mu_n \hat \xi_n) - \sum_{n=1}^N a_n (\mathbf w\cdot \mathbf x_n + b + \xi_n + \epsilon - y_n) - \sum_{n=1}^N a_n (-\mathbf w\cdot \mathbf x_n + b - \xi_n - \epsilon - y_n)
$$

by taking derivatives w.r.t $\mathbf w, b, \xi_n, \hat \xi_n$:

$$
\partial L / \partial \mathbf w = 0 \Leftrightarrow \mathbf w = \sum_{n=1}^N(a_n - \hat a_n)\mathbf x_n
$$

$$
\partial L / \partial b = 0 \Leftrightarrow \sum_{n=1}^N (a_n - \hat a_n) = 0
$$

$$
\partial L / \xi_n = 0 \Leftrightarrow a_n + \mu_n = 0 
$$

$$
\partial L / \hat \xi_n = 0 \Leftrightarrow \hat a_n + \hat \mu_n = 0
$$

Finally, the dual problem can be written as:

$$
\tilde L(\mathbf a, \hat{\mathbf a}) = -\frac12 \sum_{n=1}^N \sum_{m=1}^N (a_n - \hat a_n)(a_m - \hat a_m)\mathbf x_n \cdot \mathbf x_m - \epsilon \sum_{n=1}^N (a_n + \hat a_n) - \sum_{n=1}^N(a_n - \hat a_n) y_n
$$

s.t.
$$
0 \leq a_n \leq C\\
0 \leq \hat a_n \leq C\\
\sum_{n=1}^N (a_n - \hat a_n) = 0
$$

This is, again, a quadratic programming problem. After having found $a_n, \hat a_n$, the prediction can be done by
$$
f(\mathbf x) = \mathbf w \cdot \mathbf x + b= \sum_{n=1}^N (a_n - \hat a_n)\mathbf x_n \cdot \mathbf x + b
$$

where 
$$
b = y_n - \epsilon - \mathbf w \cdot \mathbf x_n = y_n - \epsilon - \sum_{m=1}^N(a_m - \hat a_m) \mathbf x_m \cdot \mathbf x_n
$$

for some $n$.

## 4.4. Support Vectors

By KKT theorem, the following equalities hold:
$$
(C-a_n) \xi_n = 0 \\
(C-\hat a_n)\hat \xi_n = 0 \\
a_n (\mathbf w \cdot \mathbf x_n + b + \xi_n + \epsilon - y_n) = 0\\
\hat a_n(\mathbf w \cdot \mathbf x_n + b - \xi_n - \epsilon - y_n) = 0\\
$$

We can classify the points into the following categories:

- Points with $a_n$ and $\hat a_n = 0$ have no role in prediction, they are outside the good margin boundary.
- Points with $0 < a_n < C$ or $0 < \hat a_n < C$ will have $\xi_n = 0$, they are inside the zone limited by 2 margin boundaries.
- Points with $a_n = C$ lie on the margin boundary.

Points of type 2 and 3 are, again, called support vectors. 



## 4.5 Implementation


```python
import pandas as pd

data = pd.read_csv("SVRData.csv", sep=",", header=None)
data.head()
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.593647</td>
      <td>-3.752560</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.103280</td>
      <td>-7.461048</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.244790</td>
      <td>0.750907</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.381638</td>
      <td>0.757323</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.732329</td>
      <td>2.963808</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_train = data.iloc[:100, 0:1].values
y_train = data.iloc[:100, 1].values
X_test = data.iloc[100:, 0:1].values
y_test = data.iloc[100:, 1].values
```


```python
plt.scatter(X_train, y_train)
```




    <matplotlib.collections.PathCollection at 0xb679978>




![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/Amphi/output_59_1.png)



```python
from sklearn.svm import SVR
import numpy as np

epsilon = 2
clf = SVR(epsilon = epsilon, kernel = "linear")
clf.fit(X_train, y_train)
```




    SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=2, gamma='auto',
      kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=False)



**Learnt coefficients w, b**


```python
clf.coef_
```




    array([[-1.44982737]])




```python
clf.intercept_
```




    array([1.34571846])



**Support vectors**


```python
clf.support_
```




    array([ 1,  8, 19, 21, 26, 30, 31, 34, 35, 43, 44, 45, 47, 50, 51, 55, 58,
           60, 62, 66, 67, 69, 70, 82, 87, 96, 98])




```python
clf.support_vectors_
```




    array([[ 2.10328028],
           [-2.58664379],
           [ 0.37974713],
           [-1.46826374],
           [-2.56602415],
           [-4.91518346],
           [ 4.10076614],
           [-4.59564666],
           [-4.11667506],
           [ 0.41453653],
           [-4.6987654 ],
           [ 3.45552055],
           [ 4.8689181 ],
           [ 1.71143378],
           [ 4.92958697],
           [-1.25048812],
           [ 2.16441719],
           [ 0.98549722],
           [ 3.97096418],
           [ 0.50204385],
           [-0.02730656],
           [ 0.8030457 ],
           [ 3.8785509 ],
           [ 4.86031016],
           [ 1.72356086],
           [-3.48165119],
           [-2.99013291]])




```python
plt.scatter(X_train, y_train)
plt.scatter(X_train[clf.support_], y_train[clf.support_])

interval = np.linspace(-5, 5, 100)
w = clf.coef_[0, 0]
b = clf.intercept_[0]
plt.plot(interval, w * interval + b, 'g', label = "Approximation line")
plt.plot(interval, w * interval + b + epsilon, 'r', label = "Margins")
plt.plot(interval, w * interval + b - epsilon, 'r')
plt.legend()
```




    <matplotlib.legend.Legend at 0xb6edcf8>




![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/Amphi/output_67_1.png)


**Prediction**


```python
y_pred = clf.predict(X_test)
y_pred
```




    array([ 7.75572104e+00, -5.74096280e+00,  1.57570534e+00, -1.84326392e+00,
           -3.56472014e-01,  8.16042154e+00, -1.70242790e+00,  2.66865390e+00,
           -4.76706642e+00, -1.85123809e-01,  5.72938470e+00, -1.43652899e+00,
            5.46224894e-01, -4.47926963e+00,  4.01760455e+00, -4.51364133e+00,
            5.89589406e+00, -1.05384754e+00, -2.84892227e+00,  7.36803409e+00,
           -1.03644560e-01,  1.02975349e+00, -5.36417058e+00, -5.35664592e+00,
           -5.16292746e+00,  2.02906971e+00,  9.74425411e-01, -2.87367509e+00,
            1.92075029e+00, -2.05171454e+00, -3.65073586e+00, -1.89544337e+00,
            6.35747520e+00,  8.13800342e-01,  7.82562503e-01,  2.09607545e+00,
           -7.30696122e-01, -1.65667889e+00, -7.79312060e-01,  5.87516676e+00,
            4.75510333e-01, -2.81929755e+00,  1.66976821e+00, -2.57959146e+00,
            1.70488340e+00,  7.38662780e-01,  3.69366447e+00,  4.68207941e+00,
           -8.49898925e-01,  7.52832645e+00,  4.66755219e+00,  4.67743554e+00,
            2.96136482e+00, -1.76671921e-01,  7.62691051e+00,  2.68651447e+00,
            2.79155812e+00,  6.20249256e+00, -4.54406523e+00,  3.12607243e+00,
            3.05527975e-01,  7.20599969e+00,  2.61820327e+00, -4.36628365e+00,
            7.58849152e+00,  1.10657862e+00,  4.67189866e+00, -2.40673076e+00,
            8.44658419e+00, -6.18671704e-01, -5.84746854e+00,  8.43996182e-01,
           -4.88578649e+00, -2.52518671e+00,  7.06342977e-01, -5.42222686e-01,
            1.96091878e+00,  4.26670637e+00,  4.38661273e+00,  4.40372538e+00,
            5.86045089e+00,  8.40390482e+00, -1.07446397e+00,  4.95858925e+00,
           -5.62209591e+00, -4.74270691e+00, -3.57842924e+00, -4.98747537e+00,
           -1.76574511e+00, -4.36671515e-01,  5.14674606e-02,  6.48279354e+00,
            5.49978891e+00, -1.45818729e+00,  5.35892543e+00,  6.73307754e-01,
            2.52295756e+00, -3.48919319e+00,  1.75762990e+00,  5.24006141e+00,
           -1.69417403e+00, -2.49566251e-02, -2.97720600e+00,  6.99245063e+00,
            3.57798649e+00, -5.27303765e+00, -2.48356000e+00, -3.62875900e+00,
            5.92227038e+00,  3.95521895e-01,  7.68735831e+00, -3.29367282e+00,
            3.22713893e+00, -2.30185981e-01,  2.32163792e+00,  5.28764708e+00,
            2.63554256e+00, -3.93959479e+00,  7.27250417e+00, -1.74279109e+00,
            8.46088439e+00,  4.90762363e+00,  1.79745107e+00, -4.73861244e+00,
            3.96639199e+00,  1.33589578e+00, -3.81489341e+00,  7.36280845e+00,
            3.33355590e+00, -4.02388441e-01,  2.78317779e+00,  1.64805236e+00,
           -2.54722367e+00, -3.99454744e+00, -1.57699567e+00,  7.22632440e+00,
           -2.45684162e+00,  3.23298109e+00, -3.03199809e+00, -1.98860430e+00,
            2.12070288e+00, -3.16879099e+00,  2.04062857e+00,  2.14269996e+00,
           -3.26519672e+00,  1.78020309e+00,  2.55859009e+00, -4.00762188e+00,
           -1.09502365e+00, -2.86212068e+00,  6.36750199e+00,  3.06025591e+00,
           -8.07008061e-01, -1.58274363e+00, -2.77070450e+00,  4.74218507e+00,
           -2.59198242e+00,  2.30967320e+00,  5.37455819e+00, -4.69145193e+00,
            2.03474840e+00,  6.11364693e+00,  4.52036256e+00,  4.50696442e+00,
            1.97433103e+00,  6.95345637e+00,  6.86347096e+00, -2.21455411e-01,
           -2.84546422e-01,  3.91906061e+00,  2.60066656e+00,  2.23497137e+00,
           -1.48300124e+00,  3.57638115e+00, -1.52419543e+00,  1.41748841e+00,
            1.82968538e+00, -4.79310771e+00,  7.70874764e+00,  1.71219178e+00,
            5.88375899e+00,  3.36538004e+00,  3.55788733e+00,  6.14082652e+00,
            5.90117075e+00, -3.85573483e-01,  1.73412084e+00,  4.22914674e+00,
           -4.82136132e+00, -3.03585800e+00, -2.94914336e+00, -4.74621188e+00,
            4.04704737e+00,  2.65062235e+00, -5.64114394e-01,  4.24311213e+00,
           -3.90988044e-01, -4.69175241e+00,  4.80323505e+00,  4.58085488e+00,
           -2.30041178e+00,  7.48776223e-01,  4.36632059e+00, -2.79527192e+00,
            7.47630661e-01,  6.63452715e+00,  5.17962702e+00, -8.70874283e-01,
            2.52533072e+00,  9.54109435e-01,  6.50604992e+00, -5.27147960e+00,
           -5.80504726e+00, -5.80156396e+00, -3.95343593e+00, -9.32108002e-01,
           -5.22668283e+00, -2.08491695e+00,  4.30095636e-01, -2.38616078e+00,
            1.63366828e+00,  4.12867629e+00, -5.29357409e+00,  5.78378775e+00,
           -2.60593379e-01,  6.29662429e+00,  4.94335594e+00, -2.97774477e+00,
            7.31577446e+00,  4.32835048e+00, -3.01538883e+00,  2.32420553e+00,
           -2.96903030e+00,  8.28361642e+00,  2.30317979e+00,  4.05732012e+00,
           -3.76702222e+00, -4.27990786e+00,  6.62946333e+00,  6.34151649e+00,
           -3.47575874e+00,  2.03736311e+00,  2.20610661e+00,  5.85343289e+00,
            7.47030266e+00,  4.98948200e+00,  2.10741981e+00,  7.44216718e+00,
            1.38674974e+00,  3.46126281e+00, -4.71981054e+00,  3.05894298e+00,
            5.96981102e-01,  4.42168664e+00, -4.78867861e+00,  2.42457598e+00,
            7.88943302e+00, -7.02599859e-01,  1.34158383e+00,  6.33243790e+00,
            4.40822970e+00, -2.01055466e+00,  2.07758553e+00,  2.83597116e+00,
            8.69126194e-01, -3.90955118e-01,  4.97349574e+00,  7.56524533e+00,
           -3.63072606e+00,  1.91285440e+00, -5.80601864e+00,  7.60658985e+00,
           -1.41846853e+00,  5.99353624e+00,  7.66875132e+00, -1.08525523e+00,
           -5.55983434e+00, -5.40496307e+00,  8.21556804e+00, -1.82772618e+00,
            9.67252178e-01,  5.85157082e+00,  1.00538377e+00,  2.60936027e+00,
           -2.67590029e+00,  4.75709814e+00,  2.71291839e+00, -5.38290091e+00,
            5.91959198e+00,  7.29657301e+00,  6.95057750e+00,  7.39003352e+00,
            6.22334049e+00, -1.95267848e+00,  7.50697538e+00,  3.61223719e+00,
            3.30606598e+00,  2.77378468e-01,  5.84745335e+00,  4.93177230e+00,
           -3.51313026e+00,  2.55754504e+00,  6.67473280e+00, -2.17644121e+00,
            4.63489631e+00,  6.18468358e+00,  4.58430571e-01,  4.06073550e+00,
           -5.71914975e+00, -4.57324502e-01,  6.16230369e+00,  4.63619622e+00,
           -1.19642346e+00,  1.92529213e+00, -1.95970631e+00, -4.64426046e+00,
            8.93608551e-01,  7.48273187e+00,  5.61519272e+00, -4.04685606e+00,
            7.63292178e+00,  1.83693282e+00,  7.61410464e-01,  1.76512399e+00,
            8.57158741e+00,  6.65752083e+00,  5.81441165e+00,  3.70936612e+00,
            6.42300634e+00, -3.99910247e+00, -4.93215208e+00, -3.94913021e+00,
           -3.28434675e-01,  3.35813917e+00,  2.14388472e-01, -4.41754398e+00,
            5.61315748e+00, -1.78668035e+00,  9.58778566e-01, -9.01163108e-01,
            3.27862771e+00, -4.32517923e+00, -2.96145204e+00, -2.54208398e-01,
           -4.67917665e+00,  1.96648438e+00,  2.69008807e+00, -3.49547106e+00,
            4.64394137e+00, -4.07113295e+00, -1.86210934e-01,  2.73708907e-01,
           -4.18776066e+00,  1.23334871e+00,  1.09158730e+00, -1.44448513e+00,
            5.61100193e+00, -4.56491414e+00, -4.81010760e+00,  6.02149690e+00,
            5.12012278e+00,  4.72309937e+00,  9.67789235e-01, -2.07534694e+00,
           -2.70456035e+00,  6.72608097e+00,  2.77638735e+00, -1.44820976e+00,
           -1.53389873e-01, -1.62216042e-01,  4.58268932e+00,  5.91779726e+00,
            4.25125988e+00,  2.31996267e+00, -2.55126252e+00,  6.28577299e+00,
            2.53565551e+00, -3.03311252e+00,  1.72622922e+00,  3.17229058e+00,
           -3.38928315e+00, -3.53062915e+00,  3.07434517e+00, -5.66333878e+00,
           -5.42263294e+00, -5.01378077e+00,  8.45383685e+00, -5.54358570e+00,
            3.18542551e+00, -5.03479285e+00,  7.42354401e+00, -4.63741021e+00,
           -1.67231296e+00,  4.57204861e+00,  5.06829382e+00, -4.55357104e+00,
            7.69090367e+00, -5.83470565e+00, -5.72447670e+00,  8.49513921e+00,
            1.37314516e+00,  7.96024637e+00, -5.53780313e+00,  2.06707876e+00,
           -4.09882502e+00,  7.23715781e+00,  6.32016098e+00, -1.00864853e-01,
           -1.04304029e+00,  1.12713477e+00, -5.03490259e+00,  7.62901321e+00,
           -2.25391839e+00,  1.73930483e-01,  4.49775171e-01,  7.11501005e+00,
            4.85164401e+00, -5.58456576e-01, -1.23698797e+00, -4.61659528e+00,
            4.46249964e+00, -5.36408813e+00, -1.49755900e+00,  8.32292802e+00,
           -3.51915375e-01, -5.65124691e+00,  6.19699110e+00,  6.31349584e+00,
           -7.99602804e-01,  6.38179575e+00, -4.36099169e+00, -4.95611162e+00,
            2.38366441e+00, -1.14410875e+00,  5.05699553e+00,  7.34520247e+00,
           -5.04333704e+00,  2.18331824e+00, -2.49281456e+00, -4.50126841e+00,
           -4.51679310e+00,  6.91306544e+00, -6.88319496e-01, -3.22477743e+00,
           -7.38070649e-01, -5.04725014e-01,  8.15590254e+00, -2.50933102e+00,
            4.92886969e+00, -5.50629827e+00,  7.92517015e+00,  1.45745619e+00,
           -1.30099629e+00, -7.99116421e-01,  1.05906550e+00,  4.44815657e-01,
            2.94108755e-03,  4.77279131e+00,  4.66423378e-01,  8.45177851e+00,
           -4.58083965e-01,  4.23801243e+00,  7.60840607e+00, -5.14170982e+00,
            6.41488114e+00, -5.57942839e+00, -2.42139674e+00, -1.37982225e-01,
            3.43931192e+00,  1.14546906e+00, -2.93794308e+00, -3.30855216e+00,
           -5.03983318e+00, -5.00199463e+00,  5.29474076e+00, -2.02811380e+00,
            1.47858259e+00, -5.68735434e+00,  4.37883940e+00,  4.07896756e+00,
            7.57340014e-01, -3.18034065e+00,  6.08510840e+00, -5.49937483e+00,
           -1.69816947e+00,  7.13573895e+00, -4.59793353e+00, -1.02974471e+00,
            1.89704809e+00, -1.71792148e+00, -2.18472531e+00,  7.49159268e+00,
            7.77154606e-01,  7.12653002e+00,  5.07415611e-01,  4.00997158e+00,
            5.10154331e+00, -1.38949682e+00,  2.74095602e+00, -1.09176201e+00,
            4.98545177e+00,  2.82884113e+00, -2.77934917e+00,  1.01535580e+00,
           -3.75884310e+00,  1.76925866e+00,  3.67593215e+00,  3.39277373e+00,
           -5.32428067e+00,  5.14568321e-02,  7.55641597e+00,  3.92372407e+00,
           -2.97864074e+00, -1.83501970e+00,  7.53067429e+00,  5.35144225e+00,
            6.77344778e+00, -5.14710723e+00,  6.35323084e+00,  4.22063973e+00,
            8.50847413e+00, -1.76645288e+00, -1.60476847e+00,  3.86750116e+00,
            1.77918681e+00,  4.36491366e+00, -2.52042465e+00, -3.40171665e+00,
            1.71822580e+00,  4.22874269e-01, -3.24821320e+00,  3.07365750e+00,
            2.50853225e+00,  2.65970943e+00, -2.80980458e+00,  6.95140552e+00,
           -4.08089152e+00,  5.30463895e+00, -3.84288146e+00,  6.39737345e+00,
            4.15178507e+00, -2.68218533e+00,  3.81115301e+00,  1.39642871e+00,
            7.94417624e+00,  3.42967690e+00,  8.53617558e+00,  7.06290699e-01,
            5.12372531e+00,  1.81466065e+00,  5.45182444e+00,  6.61951149e+00,
           -4.41532606e+00, -2.51824197e+00,  8.29176469e+00, -2.94474313e-01,
            3.55412917e+00,  1.58024738e+00, -5.17574310e+00,  4.75068611e+00,
           -6.58605084e-01,  8.27949549e+00, -1.09241568e+00,  5.17680729e+00,
            2.23732933e+00, -4.84042202e+00, -5.48504819e+00, -4.71914234e+00,
            8.10931922e+00, -3.60848576e+00, -2.48673640e+00, -1.84461303e+00,
            3.79264810e+00, -7.07843192e-01,  4.26581870e+00,  7.69396071e+00,
            5.70801673e-01,  8.36617368e+00, -3.92010578e+00,  5.64530842e+00,
            3.55200242e+00,  1.56562009e+00, -3.61795246e-01, -4.49106049e+00,
           -1.52703186e-02,  7.72934972e+00, -3.01099649e-01, -1.08858780e+00,
           -1.48487517e+00,  3.64293130e+00,  5.14827740e+00,  2.12435335e+00,
           -4.07931122e+00,  8.02882336e+00, -3.57567783e+00,  3.45387393e+00,
           -1.70795718e+00, -2.17937436e+00,  1.30167440e+00,  3.88619898e+00,
           -2.52632229e+00, -4.93615781e+00, -5.77754198e+00, -4.43201011e+00,
           -1.56898123e+00,  2.87134677e+00, -3.12679083e+00,  3.82552118e+00,
           -4.38686199e+00,  3.19448234e+00,  1.70303051e+00,  7.58064964e+00,
           -4.74892466e+00,  5.70568332e+00,  6.85670205e+00,  4.78650749e+00,
            3.23444037e+00,  3.74478205e+00, -1.46603870e+00, -3.72443596e+00,
           -2.62700312e+00, -1.09423770e+00, -5.13977857e+00,  8.50143223e+00,
            6.26427277e-02,  4.17594882e+00,  7.85841981e+00,  1.87006685e+00,
           -5.69659274e+00, -2.66313791e+00, -2.98219569e+00,  4.16911759e+00,
            1.51943132e+00, -4.62017012e+00, -3.75104258e+00, -2.20880940e+00,
           -9.67653293e-01,  1.71512166e+00, -4.78020991e+00,  5.51325949e-02,
            8.57868751e+00, -2.65407197e+00, -3.04458498e+00, -5.70788170e+00,
            7.45683958e+00,  3.71024562e+00,  2.88703101e+00, -2.30678977e+00,
            2.13833042e+00,  7.13620181e+00,  1.29557672e+00,  5.65210242e+00,
            6.81149749e+00,  3.10033052e-01, -1.85312124e+00,  4.90764502e+00,
            8.56210999e+00, -2.25722221e+00, -4.96005414e+00, -1.38292533e+00,
           -2.24552732e+00,  1.73885220e+00,  4.61496850e+00, -1.82491887e+00,
            2.68785848e+00,  1.40205099e+00,  4.58957063e-01, -4.10303628e+00,
           -3.10597305e+00, -5.63435280e+00, -2.32711312e+00, -3.06289252e-02,
            4.19954295e+00, -2.27620760e+00,  7.34496048e+00, -8.23437711e-01,
            3.69332326e+00, -4.45400419e+00,  5.54793555e+00,  5.67152589e+00,
            6.18412825e+00, -2.55239178e+00,  2.20824591e+00,  3.80289367e+00,
           -4.28628337e+00, -5.37577375e+00,  4.30243886e+00,  3.84911060e+00,
           -5.38338666e+00,  4.59886405e+00, -4.80344770e-02,  8.47031611e-01,
           -1.05923582e+00, -2.44798088e+00,  7.03366281e+00,  6.44914936e-01,
           -4.17719237e+00,  3.37392471e+00, -5.05279963e+00,  2.79737362e+00,
            1.02171953e+00, -3.06709892e+00,  8.15557825e+00,  8.59253732e+00,
            7.99688772e+00,  4.15307510e+00,  1.81261548e+00,  7.80924870e+00,
            4.16311131e-01, -4.48939020e-01, -4.26462312e+00,  8.07216747e+00,
           -4.94376270e+00, -2.49980789e+00,  7.72326651e+00,  4.49506296e+00,
            7.25625286e+00, -1.47067941e+00,  1.87947744e+00,  2.15759268e+00,
           -3.07998158e+00,  2.13129712e+00,  4.87070873e+00,  6.99120579e+00,
            4.52978152e+00, -4.64463566e+00, -5.00192941e+00,  1.38884452e-01,
            1.37872327e+00, -5.03003969e-01,  6.51465685e-01,  4.27675794e+00,
            1.92970344e+00,  5.53837228e+00,  7.71109437e+00, -5.76153978e+00,
           -3.63006793e+00,  4.00707690e+00, -1.67966002e+00, -2.29103419e+00,
            8.21375160e+00,  6.64598084e+00,  5.95043012e+00,  5.33424374e+00,
            5.81213354e+00,  4.38565769e-01,  5.25313020e+00, -1.54408216e+00,
           -2.23855709e+00, -1.33734634e+00, -1.01930157e+00,  4.60033435e+00,
           -5.15543858e+00, -3.76182851e+00, -5.56974319e+00,  2.98126703e+00,
            1.53170900e+00,  5.72025262e+00, -8.95229712e-01,  5.03439242e+00,
           -5.00950088e+00,  3.02390683e-02, -3.59827888e+00,  6.02160895e+00,
            5.73233309e+00, -3.91321697e-01,  3.88114650e+00, -5.88871951e+00,
            4.09427660e+00,  2.69914548e+00, -3.69145751e+00, -2.85280701e+00,
           -1.79280176e+00,  5.58424555e-01,  1.10197360e+00, -5.12796493e+00,
            4.47075254e+00,  4.51642739e+00, -1.18950765e-01, -2.64475322e+00,
            3.97077545e+00, -4.38896956e+00, -1.72316834e+00, -8.22321783e-01,
            2.19779050e+00, -4.30446536e+00, -2.62188334e+00,  3.23370188e+00,
            4.65505106e+00,  8.48326112e+00,  5.44317501e+00, -2.15006107e+00,
            7.25390004e+00, -2.66058619e+00, -6.70946536e-01,  6.19192419e+00,
           -4.06340343e+00, -1.78282416e+00,  2.17594286e+00,  2.76983260e+00,
            5.55244022e+00, -4.48177465e+00, -4.20792808e+00,  7.95641752e-01,
            6.86888212e+00,  5.39623807e+00, -4.10855013e+00, -3.39371276e+00,
           -3.54219655e+00, -2.79770963e-01,  6.87090983e+00, -1.39491381e+00,
            7.67656699e+00,  3.20924748e+00,  5.74433229e+00,  3.20708143e+00,
           -3.26199506e+00, -1.80152702e+00,  7.11095895e+00,  2.00674711e+00,
            7.02273399e+00, -1.45087755e+00,  7.48362435e+00,  8.23149109e+00,
            8.41616176e+00,  7.43115098e+00, -1.41694033e+00, -1.16212572e+00,
           -1.34382580e+00, -4.17663568e+00,  5.53766326e+00,  6.23825023e+00,
           -2.81580723e+00,  7.97623204e-01, -3.14900446e+00,  2.22864723e+00,
           -6.99131648e-01,  5.66782118e+00, -2.30695446e+00, -2.96682150e+00,
            2.30587286e+00, -8.45298987e-01,  2.70698171e+00, -4.98019262e+00,
            8.07862304e+00,  3.09727378e+00,  6.82767073e+00, -2.55573914e+00,
           -1.79625161e+00, -3.46176227e-01,  7.24908069e+00,  7.65292569e+00,
            8.42113856e+00, -7.60747499e-01, -3.01773379e-01,  2.59906129e+00,
            1.18328282e-01,  2.02971185e+00,  6.97188370e+00, -3.41155136e-01,
           -1.57342685e-01,  2.35622057e+00,  7.10964256e-01, -6.11634999e-01,
            4.78041027e+00,  3.15943593e+00,  7.48594446e-01,  7.36974112e+00,
            2.97429319e+00,  5.57804308e+00, -4.79309980e-01, -2.16993646e+00,
            7.83639029e+00, -4.94455196e-01,  5.49923794e+00,  5.89440120e-01,
            4.86756507e+00,  4.04409978e+00, -3.99566775e+00,  7.04124045e-01,
            5.13938673e+00,  2.79567950e+00, -2.25445421e+00, -1.04150122e+00,
            5.82753710e+00, -3.81129062e+00,  3.95128846e+00,  6.13616050e+00,
           -5.13002102e+00,  5.08920366e+00, -4.66556811e+00,  8.06272884e+00,
            4.30520446e+00, -5.28915058e+00,  4.03928241e+00, -4.62449148e+00,
           -2.31346942e+00, -3.40388953e+00, -3.42714442e+00, -1.00651626e+00])




```python
clf.score(X_test, y_test) #R2 score
```




    0.8108525741117112




# 5. Kernel Methods

## 5.1 Non-linearly Separable Problems

We have seen that, training a SVM is maximizing the dual problem:
$$
\tilde L(\mathbf a) = \sum_{n=1}^N a_n - \frac12 \sum_{n=1}^N \sum_{m=1}^N a_n a_m \tilde y_n \tilde y_m \mathbf x_n \cdot \mathbf x_m
$$

s.t.
$$
0 \leq a_n \leq C\\
\sum_{n=1}^N a_n \tilde y_n = 0
$$

The decision function for new vectors is given by
$$
f(\mathbf x) = \mathbf w \cdot \mathbf x + b = \sum_{\mathcal S} a_n \tilde y_n \mathbf x_n\cdot \mathbf x + \frac1{|\mathcal M|}\sum_{\mathcal M}(\tilde y_n - \sum_{\mathcal S}a_m \tilde y_m \mathcal x_m\cdot \mathcal x_n)
$$

Classification is done based on the sign of this quantity.

In both training and prediction phases, only inner products of the vectors are important, i.e, if we don't know what $\mathbf x_n$, $\mathbf x_m$ but can find out a way to compute $\mathbf x_n \cdot \mathbf x_m$ and in general for $\mathbf x \cdot \mathbf y$ for any pair $\mathbf x, \mathbf y \in \mathbf R^d$, we can reconstruct the training and prediction phase. This idea can be generalized to a strategy called **Kernel methods**.

In the following example, we see that a non-linearly separable problem can become linearly separable if we apply a transformation:

<img src = "F4.png" width=600></img>

If we put $p = (x, y)$ and $\phi(p) = (x, y, x^2 + y^2)$, then
$$
K(p_1, p_2) := \phi(p_1)\cdot \phi(p_2) = x_1 x_2 + y_1 y_2 + (x_1^2 + y_1^2)(x_2^2 + y_2^2)
$$

If we train the model by applying the dual problem:
$$
\tilde L(\mathbf a) = \sum_{n=1}^N a_n - \frac12 \sum_{n=1}^N \sum_{m=1}^N a_n a_m \tilde y_n \tilde y_m K(\mathbf x_n, \mathbf x_m)
$$

s.t.
$$
0 \leq a_n \leq C\\
\sum_{n=1}^N a_n \tilde y_n = 0
$$

and using the decision function:
$$
f(\mathbf x) = \mathbf w \cdot \mathbf x + b = \sum_{\mathcal S} a_n \tilde y_n K(\mathbf x_n, \mathbf x) + b
$$

then it is equivalent to solving the classification for $\phi(p)$ in $3D$.

We see that in bose training and prediction phases, the importance is knowing how to calculate:
$$
K(p_1, p_2) = x_1 x_2 + y_1 y_2 + (x_1^2 + y_1^2)(x_2^2 + y_2^2)
$$

The knowledge abour $\phi(p)$ (i.e., the mapping transforming the space to a linearly separable space so that the data become linearly separable), can be ignored.

## 5.2 Formalization

**Definition - Positive Definite Kernels**

*Let $X$ be a set. A mapping* $$K: \mathcal X \times X \to \mathcal R$$
$$
(x, y) \mapsto K(x, y)
$$
*is called a positive definite kernel if for any $\{ x_1, \ldots, x_L\}$, $x_i \in X$, the Gram matrix*
$$
\mathbf K = \begin{pmatrix}
K(x_1, x_1), \ldots, K(x_1, x_L)\\
\cdots \\
K(x_L, x_1), \ldots, K(x_L, x_L)
\end{pmatrix}
$$

*is semipositive definite.*

**Example**
The mapping in section 5.1 is a positive definite kernel.

**Definition - Reproducing Kernel Hilbert Space (RKHS) **

*Let $X$ be a compact subset of $\mathbf R^n$ and $\mathcal H $ be a Hilbert space $H$ of functions from $X \to \mathbf R$. Then $H$ is a **reproducting kernel Hilbert space (RKHS)** if there exists some positive definite kernel $K: X \times X \to \mathbf R$ such that

- $K$ has the reproducing property: $\langle K(\cdot, x), f \rangle = f(x)$
- $K$ spans $H$, i.e, $\mathrm{span} (\{ k(\cdot, x): x \in \mathcal X\}) = H$

*$K$ is then called a reproducing kernel of $H$*.

**Theorem (Moore-Aronszajn)**

*Let $K: X \times X \to \mathbf R$ be positive definite. Then there is a unique RKHS $\mathcal H \subset \mathbf R^X$ with reproducing kernel $K$.*

**Theorem (The Representer Theorem)**

*Let $K$ be a kernel on $X$ and let $H$ be its associated RKHS. Fix $x_1, \ldots, x_N \in X$, and consider the optimization problem*

$$
\min_{f\in H} D(f(x_1), \ldots, f(x_n)) + P(\Vert f \Vert_{H}^2)
$$

*where $P$ is a non decreasing and $D$ is a function. If the problem has a minizer, then it has a minimizer of the form*
$$
f = \sum_{i=1}^N \alpha_i K(\cdot, x_i)
$$

*where $\alpha_i \in \mathbf R$. Furthermore, if $P$ is strictly increasing, then every solution of the problem has this form*.



## 5.3 Properties of Kernels

Given positive definite kernels $K_1(\mathbf x, \mathbf x'), K_2(\mathbf x, \mathbf x')$ the following kernels are also positive definite:

$$
K(\mathbf x, \mathbf x') = \mathbf K_1(\mathbf x, \mathbf x')\\
K(\mathbf x, \mathbf x') = f(\mathbf x) K_1(\mathbf x, \mathbf x')f(\mathbf x')\\
K(\mathbf x, \mathbf x') = q(K_1(\mathbf x, \mathbf x'))\\
K(\mathbf x, \mathbf x') = \exp(K_1(\mathbf x, \mathbf x'))\\
K(\mathbf x, \mathbf x') = K_1(\mathbf x, \mathbf x') + K_2(\mathbf x, \mathbf x')\\
K(\mathbf x, \mathbf x') = K_1(\mathbf x, \mathbf x')K_2(\mathbf x, \mathbf x')\\
K(\mathbf x, \mathbf x') = \mathbf x^t \cdot \mathbf A \mathbf x'
$$
($q$ is a polynomial with nonnegative coefficients), $f$ is any function, $\mathbf A$ is a symmetric positive semidefinite matrix)

## 5.4 Mostly Used Kernels

(From Mr. Tiep VU's blog, [3])
<table>
    <tr>
        <th>Name</th>
        <th>Function</th>
        <th>Parameter in scikit learn</th>
    </tr>
    <tr>
        <td>Linear</td>
        <td>$\mathbf x \cdot \mathbf  x'$</td>
        <td>`kernel='linear'`</td>
    </tr>
    <tr>
        <td>Polynomial</td>
        <td>$(r + \gamma \mathbf x \cdot \mathbf x')^d$</td>
        <td>`kernel='poly', degree = d, gamma = g, coef0 = r`</td>
    </tr>   
    <tr>
        <td>Sigmoid</td>
        <td>$\tanh(\gamma \mathbf x \cdot \mathbf x' + r)$</td>
        <td>`kernel='poly', gamma = g, coef0 = r`</td>
    </tr>  
    <tr>
        <td>rbf</td>
        <td>$\exp(-\gamma\Vert \mathbf x - \mathbf x'\Vert^2)$</td>
        <td>`kernel='rbf', gamma = g`</td>
    </tr>
</table>

## 5.5 Implementation


```python
data = pd.read_csv("KernelData.csv", sep=",", header=None)
red_data = data[data[2] == -1]
blue_data = data[data[2] == 1]

plt.scatter(red_data[0], red_data[1], color='red', marker='o')
plt.scatter(blue_data[0], blue_data[1], color='blue', marker='x')

X = data.values[:,:2]
y = data.values[:,2]
```


![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/Amphi/output_80_0.png)



```python
X_train = X[:100]
X_test = X[100:]
y_train = y[:100]
y_test = y[100:]
clf = SVC(C = 1.0, kernel = 'linear')
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
```




    0.7




```python
clf = SVC(C = 1.0, kernel = 'poly', degree = 2)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
```




    0.98




```python
clf.support_
```




    array([21, 63, 64, 65, 18, 26, 27, 69, 91])




```python
plt.scatter(red_data[0], red_data[1], color='red', marker='o')
plt.scatter(blue_data[0], blue_data[1], color='blue', marker='x')
plt.scatter(X_train[clf.support_][:,0], X_train[clf.support_][:,1], color='green', marker='o')
```




    <matplotlib.collections.PathCollection at 0xfe5d390>




![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/Amphi/output_84_1.png)



```python
clf = SVC(C = 1.0, kernel = 'rbf', gamma = 0.1)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
```




    0.97




```python
plt.scatter(red_data[0], red_data[1], color='red', marker='o')
plt.scatter(blue_data[0], blue_data[1], color='blue', marker='x')
plt.scatter(X_train[clf.support_][:,0], X_train[clf.support_][:,1], color='green', marker='o')
```




    <matplotlib.collections.PathCollection at 0xff0fcf8>




![png](https://raw.githubusercontent.com/riduan91/DSC101/master/Lesson9/Amphi/output_86_1.png)


More on [3]

## References
[1] C. M. Bishop, Pattern Recognition and Machine Learning (chapter 7)

[2] http://web.eecs.umich.edu/~cscott/past_courses/eecs598w14/notes/13_kernel_methods.pdf

[3] Vu Huu Tiep, machinelearningcoban.com blog, https://machinelearningcoban.com/2017/04/22/kernelsmv/

[4] http://130.243.105.49/Research/Learning/courses/ml/2011/lectures/ML_2011_L04.pdf

[5] https://pdfs.semanticscholar.org/presentation/5a56/a93897162a6b473a4277c84c3047ce242264.pdf
