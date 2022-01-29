import tensorflow as tf
import numpy as np
import cv2

import utils
import sketch_env
import model
from HyperParameters import HP

continue_training = True
saved_model = '/models/creator'
Model_Name = 'creator'

if continue_training:
  actor_model, critic_model, target_actor, target_critic = utils.load_models()
else:
  actor_model = model.getActor()
  critic_model = model.getCritic()
  target_actor = model.getActor()
  target_critic = model.getCritic()
  target_actor.set_weights(actor_model.get_weights())
  target_critic.set_weights(critic_model.get_weights())

ou_noise = utils.OUActionNoise()

critic_optimizer = tf.keras.optimizers.Adam(HP.critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(HP.actor_lr)

env = sketch_env.SketchEnv()

#Replay Buffer for experience based training according to the DDPG paper
class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=HP.batch_size):
        """
        Replay Buffer Initializer

        Args:
        buffer_capacity: size of the buffer
        batch_size: batch_size for training

        Returns:
        Buffer class object
        """
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0

        num_states = HP.state_dims[0]
        num_actions = HP.action_dimension
        self.state_buffer = np.zeros((self.buffer_capacity, num_states, num_states))
        self.prev_action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states, num_states))
        self.done_buffer = np.zeros((self.buffer_capacity, 1))

    def record(self, obs_tuple):
        """
        This performs recording of the (s, a, r, s') tuple according to DDPG paper

        Args:
        obs_tuple: the tuple (state, action, reward, next_state)

        Returns:
        Nothing
        """
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.prev_action_buffer[index] = obs_tuple[1]
        self.action_buffer[index] = obs_tuple[2]
        self.reward_buffer[index] = obs_tuple[3]
        self.next_state_buffer[index] = obs_tuple[4]
        self.done_buffer[index] = obs_tuple[5]

        self.buffer_counter += 1

    @tf.function
    def update(
        self, state_batch, prev_action_batch, action_batch, reward_batch,
        next_state_batch, done_batch
    ):
        """
        This is the main training step. Performs a step of training according to the DDPG algorithm.

        Args:
        state_batch: batch of current states
        action_batch: batch of actions taken in the state
        reward_batch: batch of rewards received for above states and actions
        next_state_batch: batch of states we went into after performing the actions

        Returns:
        Nothing
        """
        with tf.GradientTape() as tape:
            target_actions = target_actor([next_state_batch, tf.squeeze(action_batch)], training=True)
            y = reward_batch + HP.gamma * (1.0 - done_batch) * target_critic(
                [next_state_batch, tf.squeeze(action_batch), tf.squeeze(target_actions)], training=True
            )
            critic_value = critic_model([state_batch, tf.squeeze(prev_action_batch), tf.squeeze(action_batch)], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model([state_batch, tf.squeeze(prev_action_batch)], training=True)
            critic_value = critic_model([state_batch, tf.squeeze(prev_action_batch), tf.squeeze(actions)], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )
    def learn(self):
        """
        Samples a batch from the Replay Buffer and performs a training step

        Args:
        None

        Returns:
        Nothing
        """
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)

        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        prev_action_batch = tf.convert_to_tensor(self.prev_action_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
        done_batch = tf.convert_to_tensor(self.done_buffer[batch_indices])
        done_batch = tf.cast(done_batch, dtype=tf.float32)

        self.update(state_batch, prev_action_batch, action_batch, reward_batch, next_state_batch, done_batch)

buffer = Buffer()

def policy(state, prev_action, model, noise_obj=None):
  sampled_action = tf.squeeze(model([state, prev_action]))
  sampled_action = sampled_action.numpy()
  
  if noise_obj is not None:
    noise = noise_obj()
    sampled_action = sampled_action + noise

  legal_action = [np.clip(sampled_action[0]*47, 0, 47), np.clip(sampled_action[1]*47, 0, 47), np.clip(sampled_action[2], 0, 1)]

  return legal_action
  

def train():
  # To store reward history of each episode
  ep_reward_list = []
  final_prob_list = []

  for ep in range(HP.total_episodes):

      prev_state, prev_action = env.reset()
      ou_noise.reset()
      episodic_reward = 0

      for i in range(HP.max_seq_length):

          tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
          tf_prev_action = tf.expand_dims(tf.convert_to_tensor(prev_action), 0)

          # Recieve state and reward from environment.
          action = policy(tf_prev_state, tf_prev_action, actor_model, ou_noise)
          state, prev_action, reward, done, _ = env.step(action)

          buffer.record((prev_state, prev_action, action, reward, state, float(done)))
          episodic_reward += reward

          buffer.learn()
          model.update_target(target_actor.variables, actor_model.variables, HP.tau)
          model.update_target(target_critic.variables, critic_model.variables, HP.tau)

          # End this episode when `done` is True
          if done:
              break

          prev_state = state
          prev_action = action
          
      final_prob_list.append(env.get_final_prob())
      ep_reward_list.append(episodic_reward)
      
      if ep%200 == 199:
          utils.save_models(actor_model, critic_model, target_actor, target_critic)

      print("episode number: {} | final classification probability = {} | number of strokes = {}".format(ep+1, final_prob_list[-1], i+1))
      #env.render()

  utils.save_models(actor_model, critic_model, target_actor, target_critic)
  return final_prob_list
  
def test():
  env = sketch_env.SketchEnv()
  prev_state, prev_action = env.reset()
  for _ in range(HP.max_seq_length):

    prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
    prev_action = tf.expand_dims(tf.convert_to_tensor(prev_action), 0)

    legal_action = policy(prev_state, prev_action, target_actor)
    # Recieve state and reward from environment.
    state, prev_action, _, done, _ = env.step(legal_action)

    # End this episode when `done` is True
    if done:
      break

    prev_state = state
    prev_action = legal_action

  cv2.imwrite('./results/creator/final_train.png', env.image)
  env.render()
  
if __name__ == "__main__":
    history = train()
    utils.plot_reward(history)

