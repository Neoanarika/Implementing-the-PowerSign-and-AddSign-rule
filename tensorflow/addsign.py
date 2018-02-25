import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf

class AddSign(optimizer.Optimizer):
    """Implementation of AddSign.
    See [Bello et. al., 2017](https://arxiv.org/abs/1709.07417)
    @@__init__
    """

    def __init__(self, learning_rate=1.001,alpha=0.01,beta=0.5, use_locking=False, name="AddSign"):
        super(AddSign, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._alpha = alpha
        self._beta = beta
        
        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._alpha_t = None
        self._beta_t = None
      
    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._alpha_t = ops.convert_to_tensor(self._beta, name="beta_t")
        self._beta_t = ops.convert_to_tensor(self._beta, name="beta_t")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta_t = math_ops.cast(self._beta_t, var.dtype.base_dtype)
        alpha_t = math_ops.cast(self._alpha_t, var.dtype.base_dtype)
    
        eps = 1e-7 #cap for moving average
        
        m = self.get_slot(var, "m")
        m_t = m.assign(tf.maximum(beta_t * m + eps, tf.abs(grad)))
        
        var_update = state_ops.assign_sub(var, lr_t*grad*(1.0+alpha_t*tf.sign(grad)*tf.sign(m_t) ) )
        #Create an op that groups multiple operations
        #When this op finishes, all ops in input have finished
        return control_flow_ops.group(*[var_update, m_t])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")

if __name__ == '__main__':
	# Hyper Parameters
	learning_rate = 0.01
	training_epochs = 2
	batch_size = 100
	tf.set_random_seed(25)

	# Step 1: Initial Setup
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets("mnist", one_hot=True)

	X = tf.placeholder(tf.float32, [None, 784])
	y = tf.placeholder(tf.float32, [None, 10])

	L1 = 200
	L2 = 100
	L3 = 50
	L4 = 40

	W1 = tf.Variable(tf.truncated_normal([784, L1], stddev=0.1))
	B1 = tf.Variable(tf.truncated_normal([L1],stddev=0.1))
	W2 = tf.Variable(tf.truncated_normal([L1, L2], stddev=0.1))
	B2 = tf.Variable(tf.truncated_normal([L2],stddev=0.1))
	W3 = tf.Variable(tf.truncated_normal([L2, L3], stddev=0.1))
	B3 = tf.Variable(tf.truncated_normal([L3],stddev=0.1))
	W4 = tf.Variable(tf.truncated_normal([L3, L4], stddev=0.1))
	B4 = tf.Variable(tf.truncated_normal([L4],stddev=0.1))
	W5 = tf.Variable(tf.truncated_normal([L4, 10], stddev=0.1))
	B5 = tf.Variable(tf.truncated_normal([10],stddev=0.1))

	# Step 2: Setup Model
	# Y1 = tf.nn.sigmoid(tf.matmul(X, W1) + B1)
	# Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + B2)
	# Y3 = tf.nn.sigmoid(tf.matmul(Y2, W3) + B3)
	# Y4 = tf.nn.sigmoid(tf.matmul(Y3, W4) + B4)
	Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)
	Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
	Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
	Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
	Ylogits = tf.matmul(Y4, W5) + B5
	yhat = tf.nn.softmax(Ylogits)

	# Step 3: Loss Functions
	loss = tf.reduce_mean(
	   tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=Ylogits))

	# Step 4: Optimizer
	train = AddSign(learning_rate).minimize(loss)

	# accuracy of the trained model, between 0 (worst) and 1 (best)
	is_correct = tf.equal(tf.argmax(y,1),tf.argmax(yhat,1))
	accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)

	# Step 5: Training Loop
	for epoch in range(training_epochs):
	    num_batches = int(mnist.train.num_examples/batch_size)
	    for i in range(num_batches):
	        batch_X, batch_y = mnist.train.next_batch(batch_size)
	        train_data = {X: batch_X, y: batch_y}
	        sess.run(train, feed_dict=train_data)

	        print(epoch*num_batches+i+1, "Training accuracy =", sess.run(accuracy, feed_dict=train_data),
	              "Loss =", sess.run(loss, feed_dict=train_data))

	# Step 6: Evaluation
	test_data = {X:mnist.test.images,y:mnist.test.labels}
	print("Testing Accuracy = ", sess.run(accuracy, feed_dict = test_data))