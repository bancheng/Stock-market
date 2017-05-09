# -*- coding:utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
import operator
import time

class LSTM_Theano:
	def __init__(self, feature_dim, output_dim, hidden_dim=128, bptt_truncate=-1):

		#初始化参数
		self.feature_dim = feature_dim
		self.output_dim = output_dim
		self.hidden_dim = hidden_dim
		self.bptt_truncate = bptt_truncate

		#E = np.random.uniform(-np.sqrt(1.0/word_dim), np.sqrt(1.0/word_dim), (hidden_dim, word_dim))
		U = np.random.uniform(-np.sqrt(1.0/hidden_dim), np.sqrt(1.0/hidden_dim), (4, hidden_dim, feature_dim))
		W = np.random.uniform(-np.sqrt(1.0/hidden_dim), np.sqrt(1.0/hidden_dim), (4, hidden_dim, hidden_dim))
		V = np.random.uniform(-np.sqrt(1.0/hidden_dim), np.sqrt(1.0/hidden_dim), (output_dim, hidden_dim))
		b = np.zeros((4, hidden_dim))
		c = np.zeros(output_dim)

		# shared variables
		#self.E = theano.shared(value=E.astype(theano.config.floatX), name='E')
		self.U = theano.shared(value=U.astype(theano.config.floatX), name='U')
		self.W = theano.shared(value=W.astype(theano.config.floatX), name='W')
		self.V = theano.shared(value=V.astype(theano.config.floatX), name='V')
		self.b = theano.shared(value=b.astype(theano.config.floatX), name='b')
		self.c = theano.shared(value=c.astype(theano.config.floatX), name='c')

		# SGD
		#self.mE = theano.shared(value=np.zeros(E.shape).astype(theano.config.floatX), name='mE')
		self.mU = theano.shared(value=np.zeros(U.shape).astype(theano.config.floatX), name='mU')
		self.mV = theano.shared(value=np.zeros(V.shape).astype(theano.config.floatX), name='mV')
		self.mW = theano.shared(value=np.zeros(W.shape).astype(theano.config.floatX), name='mW')
		self.mb = theano.shared(value=np.zeros(b.shape).astype(theano.config.floatX), name='mb')
		self.mc = theano.shared(value=np.zeros(c.shape).astype(theano.config.floatX), name='mc')
		self.theano = {}
		self.__theano_build__()


	def __theano_build__(self):
		U, W, V, b, c = self.U, self.W, self.V, self.b, self.c
		x = T.dmatrix('x')
		y = T.dmatrix('y')
		#y = T.dvector('y')

		# feed forword
		def forword_prop_step(x_t, s_prev, c_prev):

			x_e = T.transpose(x_t)

			# lstm layer
			i = T.nnet.hard_sigmoid(U[0].dot(x_e) + W[0].dot(s_prev) + b[0])
			f = T.nnet.hard_sigmoid(U[1].dot(x_e) + W[1].dot(s_prev) + b[1])
			o = T.nnet.hard_sigmoid(U[2].dot(x_e) + W[2].dot(s_prev) + b[2])
			g = T.tanh(U[3].dot(x_e) + W[3].dot(s_prev) + b[3])
			c_t = c_prev * f + g * i
			s_t = T.tanh(c_t) * o

			#output
			#o_t = V.dot(s_t) + c
			#o_t = T.maximum(0, V.dot(s_t) + c)
			#o_t = T.tanh(V.dot(s_t) + c) #
			o_t = T.nnet.softmax(V.dot(s_t) + c)
			return [o_t, s_t, c_t]


		[o, s, c_o], updates = theano.scan(
			forword_prop_step,
			sequences=x,
			truncate_gradient=self.bptt_truncate,
			outputs_info=[None,
						  dict(initial=T.zeros(self.hidden_dim)),
						  dict(initial=T.zeros(self.hidden_dim))]
		)

		o = T.reshape(o, (o.shape[0], self.output_dim))
		prediction = T.argmax(o, axis=1)
		#MSE
		#o_error = T.sum(T.sqrt(T.sum(T.square(o - y), axis=0)))
		#CE
		o_error = -T.mean(T.sum(y * T.log(o), axis=1))
		#o_error = T.sum(T.nnet.categorical_crossentropy(o, y))
		cost = o_error

		# calculate gradient
		#dE = T.grad(cost, E)
		dU = T.grad(cost, U)
		dW = T.grad(cost, W)
		dV = T.grad(cost, V)
		db = T.grad(cost, b)
		dc = T.grad(cost, c)

		self.predict = theano.function([x], o)
		self.hidden_layer = theano.function([x], s)
		self.predict_class = theano.function([x], prediction)
		self.ce_error = theano.function([x, y], cost)
		self.bptt = theano.function([x, y], [dU, dW, db, dV, dc])


		# parameter for SGD
		learning_rate = T.scalar('learning_rate')
		decay = T.scalar('decay')
		# update
		#mE = decay * self.mE + (1 - decay) * dE**2
		mU = decay * self.mU + (1 - decay) * dU**2
		mW = decay * self.mW + (1 - decay) * dW**2
		mb = decay * self.mb + (1 - decay) * db**2
		mV = decay * self.mV + (1 - decay) * dV**2
		mc = decay * self.mc + (1 - decay) * dc**2

		self.sgd_step = theano.function(
			[x, y, learning_rate, theano.Param(decay, default=0.9)],
			[],
			updates=[
					 (U, U - learning_rate * dU / T.sqrt(mU + 1e-6)),
					 (V, V - learning_rate * dV / T.sqrt(mV + 1e-6)),
					 (b, b - learning_rate * db / T.sqrt(mb + 1e-6)),
                     (c, c - learning_rate * dc / T.sqrt(mc + 1e-6)),
					 (self.mU, mU),
					 (self.mW, mW),
                     (self.mV, mV),
                     (self.mb, mb),
                     (self.mc, mc)]
			#allow_input_downcast=True
		)

	def calculate_total_loss(self, X, Y):
		return np.sum([self.ce_error(x, y) for x, y in zip(X, Y)])

	def calculate_loss(self, X, Y):
		num_words = np.sum([len(y) for y in Y])
		return self.calculate_total_loss(X, Y) / float(num_words)

	#def calculate_MSE(self, x, y):



