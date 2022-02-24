import torch
import hw1_utils as utils
import numpy as np

'''
    Important
    ========================================
    The autograder evaluates your code using FloatTensors for all computations.
    If you use DoubleTensors, your results will not match those of the autograder
    due to the higher precision.

    PyTorch constructs FloatTensors by default, so simply don't explicitly
    convert your tensors to DoubleTensors or change the default tensor.

    Be sure to modify your input matrix X in exactly the way specified. That is,
    make sure to prepend the column of ones to X and not put the column anywhere
    else, and make sure your feature-expanded matrix in Problem 4 is in the
    specified order (otherwise, your w will be ordered differently than the
    reference solution's in the autograder).
'''
# Problem 3
#def loss(w, x, y):
  #  wt = w.T
   # wtx = torch.mul(wt, x)
    #return 0.5*(wtx-y)**2

def linear_gd(X, Y, lrate=0.01, num_iter=1000):
    w = torch.zeros(X.shape[1] + 1, 1)
    n = X.shape[0]
    ones = torch.ones(n, 1)
    X = torch.cat((ones, X), 1)

    for x in range(num_iter):
        w -= lrate * X.T @ (X @ w - Y) / n



    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        num_iter (int): iterations of gradient descent to perform

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    '''
    return w

def linear_normal(X, Y):
    n = X.shape[0]
    ones = torch.ones(n, 1)
    X = torch.cat((ones, X), 1)
    PseudoX = torch.linalg.pinv(X)
    w = PseudoX @ Y
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    '''
    return w

def plot_linear():
    X, Y = utils.load_reg_data();
    return utils.contour_plot(X.min(), X.max(), Y.min(), Y.max(), linear_normal(X, Y), 100)
    '''
        Returns:
            Figure: the figure plotted with matplotlib
    '''

# Problem 4
def poly_gd(X, Y, lrate=0.01, num_iter=1000):

    n = X.shape[0]
    d = X.shape[1]
    S = X.T
    
    for i in range(d):
        for j in range(i, d):
            a = S[j] * S[i]
            a = torch.reshape(a, (1, n))
            S = torch.cat((S, a), 0)
    
    X = S.T
    ones = torch.ones(n, 1)
    X = torch.cat((ones, X), 1)
    
    w = torch.zeros(X.shape[1], 1)
        


    for x in range(num_iter):
        w -= lrate * X.T @ (X @ w - Y) / n

    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        lrate (float): the learning rate
        num_iter (int): number of iterations of gradient descent to perform

    Returns:
        (1 + d + d * (d + 1) / 2) x 1 FloatTensor: the parameters w
    '''
    return w

def poly_normal(X,Y):



    n = X.shape[0]
    d = X.shape[1]
    S = X.T
    
    for i in range(d):
        for j in range(i, d):
            a = S[j] * S[i]
            a = torch.reshape(a, (1, n))
            S = torch.cat((S, a), 0)
    
    X = S.T
    ones = torch.ones(n, 1)
    X = torch.cat((ones, X), 1)

    PseudoX = torch.linalg.pinv(X)
    w = PseudoX @ Y
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (1 + d + d * (d + 1) / 2) x 1 FloatTensor: the parameters w
    '''
    return w

def plot_poly():
    '''
    Returns:
        Figure: the figure plotted with matplotlib
    '''
    pass

def poly_xor():
    '''
    Returns:
        n x 1 FloatTensor: the linear model's predictions on the XOR dataset
        n x 1 FloatTensor: the polynomial model's predictions on the XOR dataset
    '''
    pass

# Problem 5


def logistic(X, Y, lrate=.01, num_iter=1000):
    
    n = X.shape[0]
    d = X.shape[1]
    ones = torch.ones((X.shape[0], 1))
    X1 = torch.cat((ones, X), 1)
    w = torch.zeros((X1.shape[1], 1), requires_grad = True)

    for i in range(num_iter):
        pred = X1 @ w
        loss = torch.sum(torch.log(torch.tensor(1) + torch.exp(-Y * pred))) / pred.numel()
        loss.backward()
        with torch.no_grad():
            w -= w.grad * lrate
            w.grad.zero_()

    return w.detach()
    
def logistic_vs_ols():
    '''
    Returns:
        Figure: the figure plotted with matplotlib
    '''
    pass
