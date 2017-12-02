import math
import numpy as np
import tensorflow as tf

import util


class Network(object):
  def __init__(self, variable_scope, inputs, reward_scaling, config,
               write_summaries):
    self.scope = variable_scope
    self.inputs = inputs
    self.config = config
    self.write_summaries = write_summaries
    self.num_heads = config.num_bootstrap_heads
    self.using_ensemble = config.bootstrap_use_ensemble

    self.build_ram_action_value_heads(inputs, reward_scaling)

    if self.using_ensemble:
      self.build_ensemble()

    self.sample_head()

  def build_ram_action_value_heads(self, inputs, reward_scaling):

    self.heads = [
        RamActionValueHead('head%d' % i, inputs, reward_scaling,
                        self.config) for i in range(self.num_heads)
    ]

    self.action_values = tf.stack(
        [head.action_values for head in self.heads],
        axis=1,
        name='action_values')
    self.activation_summary(self.action_values)

    self.taken_action_value = self.action_value(
        inputs.action, name='taken_action_value')

    value, greedy_action = tf.nn.top_k(self.action_values, k=1)
    print('value: {}'.format(value))
    self.value = tf.squeeze(value, axis=2, name='value')
    self.greedy_action = tf.squeeze(
        greedy_action, axis=2, name='greedy_action')

  def action_value(self, action, name='action_value'):
    with tf.name_scope(name):
      return self.choose_from_actions(self.action_values, action)

  def log_policy(self, action, name='log_policy'):
    with tf.name_scope(name):
      return self.choose_from_actions(self._log_policy, action)

  def choose_from_actions(self, actions, action):
    return tf.reduce_sum(
        actions * tf.one_hot(action, self.config.num_actions), axis=2)

  def sample_head(self):
    self.active_head = self.heads[np.random.randint(self.num_heads)]

  @property
  def choose_action(self):
    if self.num_heads == 1 or not self.using_ensemble:
      return self.active_head.greedy_action
    else:
      return self.ensemble_greedy_action

  @property
  def eval_actions(self):
      return self.active_head.reg_action_values

  @property
  def variables(self):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope.name)

  @property
  def flat_inp(self):
      return self.active_head.flat_inp

  def activation_summary(self, tensor):
    if self.write_summaries:
      tensor_name = tensor.op.name
      tf.summary.histogram(tensor_name + '/activations', tensor)
      tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(tensor))

class RamActionValueHead(object):
  def __init__(self, name, inputs, reward_scaling, config):
    with tf.variable_scope(name):
      flat_inputs = tf.reshape(inputs.ram, [-1, 128])
      action_values = self.action_value_layer(flat_inputs, config)
      action_values = reward_scaling.unnormalize_output(action_values)
      value, greedy_action = tf.nn.top_k(action_values, k=1)

      self.flat_inp = flat_inputs

      self.reg_action_values = tf.identity(action_values)
      self.action_values = tf.multiply(
          inputs.alive, action_values, name='action_values')
      self.value = tf.squeeze(inputs.alive * value, axis=1, name='value')
      self.greedy_action = tf.squeeze(
          greedy_action, axis=1, name='greedy_action')

  def action_value_layer(self, ram_vec, config):
    if config.dueling:
      hidden_value_1 = tf.layers.dense(
          ram_vec, 256, tf.nn.relu, name='hidden_value_1')
      hidden_value_2 = tf.layers.dense(
          hidden_value_1, 256, tf.nn.relu, name='hidden_value_2')
      value = tf.layers.dense(hidden_value_2, 1, name='value')

      hidden_actions_1 = tf.layers.dense(
          ram_vec, 256, tf.nn.relu, name='hidden_actions_1')
      hidden_actions_2 = tf.layers.dense(
          hidden_actions_1, 256, tf.nn.relu, name='hidden_actions_2')
      actions = tf.layers.dense(
          hidden_actions_2, config.num_actions, name='actions')

      return value + actions - tf.reduce_mean(actions, axis=1, keep_dims=True)

    else:
      hidden_1 = tf.layers.dense(ram_vec, 256, tf.nn.relu, name='hidden_1')
      hidden_2 = tf.layers.dense(hidden_1, 256, tf.nn.relu, name='hidden_2')
      return tf.layers.dense(hidden_2, config.num_actions, name='action_value')
