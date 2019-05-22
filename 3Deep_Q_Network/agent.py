"""
Agent for the reinforcement learning, which learn from the environment and choose action.
"""


import numpy as np
import tensorflow as tf


def network(name, state, n_action, target_Q=None):
    """A simple 2 layer neural network
    """
    n_state = state.shape[1].value
    with tf.variable_scope(name):
        # c_names(collections_names) are the collections to store variables
        c_names, n_l1, w_initializer, b_initializer = (
            ["{}_params".format(name), tf.GraphKeys.GLOBAL_VARIABLES], 10,
            tf.random_normal_initializer(0., 0.3),
            tf.constant_initializer(0.1) # config of layers
        )

        # first layer.collections is used later when assign to target net
        with tf.variable_scope("l1"):
            w1 = tf.get_variable("w1",
                                 [n_state, n_l1], 
                                 initializer=w_initializer,
                                 collections=c_names)
            b1 = tf.get_variable("b1",
                                 [1, n_l1],
                                 initializer=b_initializer,
                                 collections=c_names)
            l1 = tf.nn.relu(tf.matmul(state, w1) + b1)

        # second layer. collections is used later when assign to target net
        with tf.variable_scope("l2"):
            w2 = tf.get_variable("w2",
                                 [n_l1, n_action],
                                 initializer=w_initializer,
                                 collections=c_names)
            b2 = tf.get_variable("b2",
                                 [1, n_action],
                                 initializer=b_initializer,
                                 collections=c_names)
            predict_Q = tf.matmul(l1, w2) + b2
    if target_Q is not None:
        with tf.variable_scope("loss"):
            loss = tf.losses.mean_squared_error(target_Q, predict_Q)

        return predict_Q, loss
    else:
        return predict_Q



class Agent(object):
    def __init__(self, opt):
        self.n_state = opt.n_state
        self.n_action = len(opt.action_space)
        self.actions = list(range(self.n_action))
        self.gamma = opt.gamma
        self.lr = opt.learning_rate
        self.batch_size = opt.batch_size
        self.epsilon = opt.epsilon
        self.loss_history = []
        self.memory_size = opt.memory_size
        self.memory = np.zeros((self.memory_size, 2 * self.n_state + 2))
        self.replace_target_iter = opt.replace_target_iter
        self.learn_step_counter = 0
        self._build_network()
        tnet_params = tf.get_collection("target_net_params")
        pnet_params = tf.get_collection("predict_net_params")
        self.replace_target_op = [tf.assign(t, p) for t, p in zip(tnet_params, pnet_params)]
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if opt.output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

    def _build_network(self):
        self.target_Q = tf.placeholder(tf.float32, shape=(None,self.n_action))
        self.state = tf.placeholder(tf.float32, shape=(None, self.n_state))
        self.predict_Q, self.loss = network("predict_net", self.state,
                                            self.n_action, target_Q=self.target_Q)
        with tf.variable_scope("train"):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
        
        self.next_state = tf.placeholder(tf.float32, [None, self.n_state], name="next_state")
        self.next_Q = network("target_net", self.next_state, self.n_action)

    def choose_action(self, state):
        if np.random.randn() < self.epsilon:
            state = state[np.newaxis, :]
            actions = self.sess.run(
                self.predict_Q,
                feed_dict={self.state: state}
            )
            action = np.argmax(actions)
        else:
            action = np.random.choice(self.actions)
        action = int(action)
        return action
    
    def store(self, state, action, reward, next_action):
        if not hasattr(self, "memory_counter"):
            self.memory_counter = 0
        observation = np.hstack((state, action, reward, next_action))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = observation
        self.memory_counter += 1

    def learn(self):
        # check to replace target net parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
        
        if self.learn_step_counter % self.replace_target_iter * 100 == 0:
            self.save()
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, self.batch_size)
        batch_memory = self.memory[sample_index, :]
        next_Q, predict_Q = self.sess.run(
            [self.next_Q, self.predict_Q],
            feed_dict={
                self.next_state: batch_memory[:, -self.n_state: ],
                self.state: batch_memory[:, :self.n_state]
            }
        )
        target_Q = predict_Q.copy()

        batch_index = np.arange(self.batch_size)
        action_index = batch_memory[:, self.n_state].astype(int)
        reward = batch_memory[:, self.n_state + 1]
        target_Q[batch_index, action_index] = reward + self.gamma * np.max(next_Q, 1)

        # train network
        _, loss = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.state: batch_memory[:, :self.n_state],
                self.target_Q: target_Q
            }
        )
        self.loss_history.append(loss)
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.loss_history)), self.loss_history)
        plt.ylabel("loss")
        plt.xlabel('training steps')
        plt.show()
    
    def save(self):
        model_name = "./models/model_{}iter.ckpt".format(self.learn_step_counter)
        save_path = self.saver.save(self.sess, model_name)
        print("Model saved in path: {}".format(save_path))
    
    def closs_session(self):
        self.sess.close()