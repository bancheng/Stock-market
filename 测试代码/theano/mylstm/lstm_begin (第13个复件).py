import numpy as np
import init
import theano
import theano.tensor as T
w = np.random.rand(5)
print w
w = theano.shared(value = w,name = 'w')
# out = np.array([1]*5)
# inter = T.dot(w, out)
t = T.nnet.softmax(w)
i  = theano.function(
    inputs = [],
    outputs = t
)
print t.evals()