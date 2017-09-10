# -*- coding: utf-8 -*-
"""
Created on Wed Mar 01 00:06:10 2017

@author: Administrator
"""

import numpy
import theano
import theano.tensor as T
from littleFun import makeAllShared
import six.moves.cPickle as pickle
#import matplotlib.pyplot as plt


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        # tensor variable 的数据如果要提取出来，要用eval()来提取数值
        # argmax 烦返回值是最大概率所在行的整数型值
        #例： y_pred.eval() 
        #array(2L, dtype=int64)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # T.arange(y.shape[0]) row index
        # y colum index
        # end-snippet-2
        
        
# error计算的是错误率

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # 若两者不相等则返回值为1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
            
def test_LG(trainSet,validSet,testSet,trainLabel,validLabel,testLabel,
            learning_rate=0.13, n_epochs=1000,batch_size=10):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    """
    
    # 此处dataSets为一个含有三个元素的元组，每个元组之内包含输入数据与标签
    #dataSets[0]包含两个元素，第一个元素为输入第二个为标签
    # 用这个方法可以会自动对应位置得到输入和标签
    train_set_x, valid_set_x, test_set_x,train_set_y,valid_set_y,test_set_y = makeAllShared(trainSet,validSet,testSet,trainLabel,validLabel,testLabel)

    # compute number of minibatches for training, validation and testing
    # //表示整数除法，返回商的整数部分，/表示浮点数除法
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
 #   print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    classifier = LogisticRegression(input=x, n_in=10, n_out=2)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # start-snippet-3
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    # givens代表函数的输入
    # cost是在一个batch上的cost，然后是在一个训练集上不断训练，直到全局最低值
    # train_model的输出是分类模型的cost
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    # end-snippet-3

    ###############
    # TRAIN MODEL #
    ###############
  #  print('... training the model')
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.998  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
    #n_train_batches 为batch的个数
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    done_looping = False
    epoch = 0
    minibatch_avg_cost =[]
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost1 = train_model(minibatch_index)
            minibatch_avg_cost.append(minibatch_avg_cost1)
            # 通过train_model对权重进行更新
#            print (
#                    (
#                        epoch,
#                        minibatch_index + 1,
#                        n_train_batches,
#                        minibatch_avg_cost
#                    )
#                )
            # iteration number
            # iter 记录的是使用batch的个数，如果满足iter%validation_frequency为0
            #那么相当于已经完全遍历训练集了，所以计算一下错误率，相当于这一轮训练过后的误差
            # 总而言之LG训练思想在于，受限将训练集分成小份，训练多次，然后每次通过多次训练遍历依次训练集
            # epoch记录了遍历训练集的次数，然后每次正轮训练之后输出依次错误率的判断
            iter = (epoch - 1) * n_train_batches + minibatch_index
#            print ('iter %d validation_frequencey%f n_train_batches%d n_test_batches%d n_validation_test%d'%(iter,validation_frequency,n_train_batches,n_test_batches,n_valid_batches ))
            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                # 此处validation_losses仍旧是错误率
                validation_losses = [validate_model(i)
                                     for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
 #               print(
   #                 'epoch %i, minibatch %i/%i, validation error %f %%' %
   #                 (
     #                   epoch,
     #                   minibatch_index + 1,
      #                  n_train_batches,
      #                  this_validation_loss * 100.
      #              )
      #          )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    #此处表明，如果此时错误率减少的足够小时，则扩大一倍的iter以寻找更好的降低误差
                    #直到不会再有明显降低为止
                    #此处参数人为设定以提高表现
                    # 实验可证如果patience_increase变大会有提高但会越来越不明显
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)
#                        print ('patience%d iter%d'%(patience,iter))

                    best_validation_loss = this_validation_loss
                    # test it on the test set
                    #只采用validation最好model来进行test的测试
                    test_losses = [test_model(i)
                                   for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

 #                   print(
  #                      (
  #                          '     epoch %i, minibatch %i/%i, test error of'
   #                         ' best model %f %%'
   #                     ) %
   #                     (
  #                          epoch,
  #                          minibatch_index + 1,
  #                          n_train_batches,
   #                         test_score * 100.
  #                      )
      #              )

                    # save the best model
                    with open('best_model.pkl', 'wb') as f:
                        pickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break

#    end_time = timeit.default_timer()
#    print(
#        (
#            'Optimization complete with best validation score of %f %%,'
#            'with test performance %f %%'
#        )
#        % (best_validation_loss * 100., test_score * 100.)
#    )
#  绘制每次epoch之后的cost变化
#    epoch_avg_cost =[]
#    for i in range(len(minibatch_avg_cost)/n_train_batches):
#        epoch_avg_cost.append(sum(minibatch_avg_cost[(i)*n_train_batches:(i+1)*n_train_batches]))
        
#    plt.plot(epoch_avg_cost)
#    plt.show()

#    print('The code run for %d epochs, with %f epochs/sec' % (
#        epoch, 1. * epoch / (end_time - start_time)))
#    print(('The code for file ' +
#           os.path.split(__file__)[1] +
#           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)

#非shared形式，只是值
def predict(test_set_x):
    """
    An example of how to load a trained model and use it
    to predict labels.
    """

    # load the saved model
    classifier = pickle.load(open('best_model.pkl'))

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred,allow_input_downcast=True)

    # We can test it on some examples from test test
#    dataset='mnist.pkl.gz'
#    datasetd_data()
#    test_set_x, test_set_y = datasets[2]
#    test_set_x = test_set_x.get_value()

    predicted_values = predict_model(test_set_x)
    return predicted_values
#    print("Predicted values for the first 10 examples in test set:")
#    print(predicted_values)