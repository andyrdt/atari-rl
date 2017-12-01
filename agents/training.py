import os
import tensorflow as tf
from threading import Thread
import numpy as np
import os.path

from networks.factory import NetworkFactory
import util


class Trainer(object):
  def __init__(self, config):
    util.log('Creating network and training operations')
    self.config = config

    # Creating networks
    factory = NetworkFactory(config)
    self.global_step, self.train_op = factory.create_train_ops()
    self.reset_op = factory.create_reset_target_network_op()
    self.agents = factory.create_agents()
    self.summary = factory.create_summary()

  def train(self):
    self.training = True

    util.log('Creating session and loading checkpoint')
    session = tf.train.MonitoredTrainingSession(
        checkpoint_dir=self.config.run_dir,
        save_summaries_steps=0,  # Summaries will be saved with train_op only
        config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

    with session:
      if len(self.agents) == 1:
        self.train_agent(session, self.agents[0])
      else:
        self.train_threaded(session)

    util.log('Training complete')

  def train_threaded(self, session):
    threads = []
    for i, agent in enumerate(self.agents):
      thread = Thread(target=self.train_agent, args=(session, agent))
      thread.name = 'Agent-%d' % (i + 1)
      thread.start()
      threads.append(thread)

    for thread in threads:
      thread.join()

  def train_agent(self, session, agent):
    # Populate replay memory
    if self.config.load_replay_memory:
      util.log('Loading replay memory')
      agent.replay_memory.load()
    else:
      util.log('Populating replay memory')
      agent.populate_replay_memory()

    # Initialize step counters
    step, steps_until_train = 0, self.config.train_period

    #test_data_out_dir = '/Users/andy/Desktop/Columbia 4.1/Deep Learning/Project/test_data_gen/data/'
    test_data_out_dir = self.config.test_data_dir
    out_pair_n = 0

    if self.config.eval_test_data:
        self.eval_test_data(agent, session, step)
        return

    util.log('Starting training')

    while self.training and step < self.config.num_steps:
      # Start new episode
      observation, _, done = agent.new_game()

      # Play until losing
      while not done:
        self.reset_target_network(session, step)

        # run evaluation network on observation to get best action:
        action = agent.action(session, step, observation)


        action_values = agent.get_action_values(session, step, observation)

        ram_state = agent.get_ram_state()

        if np.random.binomial(1,0.01) == 1 and self.config.save_test_data:
            np.save(test_data_out_dir+'frames'+str(out_pair_n),observation)
            np.save(test_data_out_dir+'ram'+str(out_pair_n),ram_state)
            out_pair_n = out_pair_n + 1

            # print('Action values: {}'.format(action_values))
            # print('Chosen action: {}'.format(action))
            # print('Observation frames: {}'.format(np.shape(observation)))
            # print('RAM state: {}'.format(ram_state))

        # take best action, and get back resulting observation
        observation, reward, done = agent.take_action(action)

        step += 1
        steps_until_train -= 1
        if done or (steps_until_train == 0):
          step = self.train_batch(session, agent.replay_memory, step)
          steps_until_train = self.config.train_period

      # Log episode
      agent.log_episode(step)

    if self.config.save_replay_memory:
      agent.replay_memory.save()

  def reset_target_network(self, session, step):
    if self.reset_op:
      if step > 0 and step % self.config.target_network_update_period == 0:
        session.run(self.reset_op)

  def train_batch(self, session, replay_memory, step):
    fetches = [self.global_step, self.train_op] + self.summary.operation(step)

    batch = replay_memory.sample_batch(fetches, self.config.batch_size)
    if batch:
      step, priorities, summary = session.run(fetches, batch.feed_dict())
      batch.update_priorities(priorities)
      self.summary.add_summary(summary, step)

    return step

  def stop_training(self):
    util.log('Stopping training')
    self.training = False

  def eval_test_data(self,agent,session, step):
    test_data_out_dir = self.config.test_data_dir
    for i in range(0,1000):
      if os.path.exists(test_data_out_dir + 'frames' + str(i) + '.npy'):
        observation = np.load(test_data_out_dir + 'frames' + str(i) + '.npy')
        action_values = agent.get_action_values(session, step, observation)
        print(action_values)
      else:
        break
