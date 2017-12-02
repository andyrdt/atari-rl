import numpy as np

from atari import Atari
from agents.exploration_bonus import ExplorationBonus


class Agent(object):
  def __init__(self, policy_network, replay_memory, summary, config):
    self.config = config
    self.policy_network = policy_network
    self.replay_memory = replay_memory
    self.summary = summary

    # Create environment
    self.atari = Atari(summary, config)
    self.exploration_bonus = ExplorationBonus(config)

  def new_game(self):
    self.policy_network.sample_head()
    frames_observation, reward, done = self.atari.reset()
    ram_observation = self.get_ram_state()

    self.replay_memory.store_new_episode(frames_observation,ram_observation)
    return frames_observation, reward, done


  def action(self, session, step, frames_observation):
    # Epsilon greedy exploration/exploitation even for bootstrapped DQN
    if np.random.rand() < self.epsilon(step):
      return self.atari.sample_action()
    else:
      [action] = session.run(
          self.policy_network.choose_action,
          {self.policy_network.inputs.observations: [frames_observation]})
      return action

  def action_ram(self, session, step, ram_observation):
    # Epsilon greedy exploration/exploitation even for bootstrapped DQN

    if np.random.rand() < self.epsilon(step):
      return self.atari.sample_action()
    else:
      [action] = session.run(
          self.policy_network.choose_action,
          {self.policy_network.inputs.ram: [np.expand_dims(ram_observation, axis=0)]})
      return action


  def get_action_values(self, session, step, observation):
      return session.run(self.policy_network.eval_actions,{self.policy_network.inputs.observations: [observation]})

  def get_ram_state(self):
      return self.atari.env._get_ram()

  def epsilon(self, step):
    """Epsilon is linearly annealed from an initial exploration value
    to a final exploration value over a number of steps"""

    initial = self.config.initial_exploration
    final = self.config.final_exploration
    final_frame = self.config.final_exploration_frame

    annealing_rate = (initial - final) / final_frame
    annealed_exploration = initial - (step * annealing_rate)
    epsilon = max(annealed_exploration, final)

    self.summary.epsilon(step, epsilon)

    return epsilon

  def take_action(self, action):
    frames_observation, reward, done = self.atari.step(action)
    training_reward = self.process_reward(reward, frames_observation)
    ram_observation = self.atari.env._get_ram()

    # Store action, reward and done with the next observation
    self.replay_memory.store_transition(action, training_reward, done,
                                        frames_observation, ram_observation)

    return frames_observation, reward, done

  def process_reward(self, reward, frames):
    if self.config.exploration_bonus:
      reward += self.exploration_bonus.bonus(frames)

    if self.config.reward_clipping:
      reward = max(-self.config.reward_clipping,
                   min(reward, self.config.reward_clipping))

    return reward

  def populate_replay_memory(self):
    """Play game with random actions to populate the replay memory"""

    count = 0
    done = True

    while count < self.config.replay_start_size or not done:
      if done: self.new_game()
      _, _, done = self.take_action(self.atari.sample_action())
      count += 1

    self.atari.episode = 0

  def log_episode(self, step):
    self.atari.log_episode(step)
