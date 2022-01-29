import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from HyperParameters import HP

wd = os.getcwd()

#Saving and Checkpointing helper functions
def save_models(actor, critic, target_actor, target_critic, Model_Name='creator', 
working_directory = wd, save_dir='./models'):
    """
    This function saves all 4 models

    Args:
    actor: actor model instance
    critic: critic model instance
    target_actor: target actor model instance
    target_critic: target critic model instance
    save_dir: directory to save the models

    Returns:
    nothing
    """
    save_dir = os.path.join(save_dir, Model_Name)
    try:
      os.chdir(save_dir)
    except FileNotFoundError:
      os.mkdir(save_dir)
      os.chdir(save_dir)
    actor.save('actor_model')
    critic.save('critic_model')
    target_actor.save('target_actor_model')
    target_critic.save('target_critic_model')
    print("Saved all models to directory: {}".format(os.getcwd()))
    os.chdir(working_directory)

def load_models(Model_Name='creator', working_directory = wd, 
save_dir='./models'):
    """
    This function loads all 4 models

    Args:
    save_dir: location of the saved models

    Returns:
    a list of the 4 models - actor, critic, target_actor, target_critic
    """
    save_dir = os.path.join(save_dir, Model_Name)
    try:
      os.chdir(save_dir)
    except FileNotFoundError:
      print("Model with specified name does not exist in given directory!")
      os.chdir(working_directory)
      return
    actor = tf.keras.models.load_model('actor_model')
    critic = tf.keras.models.load_model('critic_model')
    target_actor = tf.keras.models.load_model('target_actor_model')
    target_critic = tf.keras.models.load_model('target_critic_model')
    print("Loaded all models from directory: {}".format(os.getcwd()))
    os.chdir(working_directory)
    return [actor, critic, target_actor, target_critic]

def plot_reward(reward_list, working_directory = wd, 
save_dir="./results", name="final_reward"):
  fig = plt.figure()
  plt.plot(reward_list)
  plt.grid(True, which='both', axis='both')
  plt.title(name)
  plt.xlabel("Episode")
  plt.ylabel("Epsiodic Reward")
  plt.show()
  plot_dir = os.path.join(save_dir, 'creator')
  try:
    os.chdir(plot_dir)
  except FileNotFoundError:
    os.mkdir(plot_dir)
  os.chdir(working_directory)
  plot_loc = plot_dir + "/" + name + ".png"
  fig.savefig(plot_loc)
  print("Plot saved to {}".format(plot_loc))
  
#Action Noise definition according to the DDPG paper (Ornstein-Uhlenbeck process)
class OUActionNoise:
    def __init__(self, mean=np.zeros(HP.action_dimension),
                std_deviation=HP.std_dev*np.ones(HP.action_dimension), 
                theta=0.15, dt=1e-2, x_initial=None):
        """
        Noise process initializer
        
        Args:
        mean: mean of the process
        std_deviation: standard deviation of the noise
        dt: time interval between two process samples
        x_initial: initial state
        
        Returns:
        Action Noise object
        """
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        """
        This defines the call behaviour of the noise object. 
        Behaviour taken from https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
        
        Args:
        None
        
        Returns:
        A sample from the noise process
        """
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        
        self.x_prev = x
        return x

    def reset(self):
        """
        Function to reset the noise to initial state
        
        Args:
        None
        
        Returns:
        nothing
        """
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)
