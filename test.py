import numpy as np
import tensorflow as tf
import cv2
import train

import sketch_env
from HyperParameters import HP

def get_ground_truth(target=False):
  if target:
    filename = './results/creator/final_target.png'
    model = tf.keras.models.load_model('./models/creator/target_actor_model/')
  else:
    filename = './results/creator/final.png'
    model = tf.keras.models.load_model('./models/creator/actor_model/')
  env = sketch_env.SketchEnv()
  prev_state, prev_action = env.reset()
  for _ in range(HP.max_seq_length):

    prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
    prev_action = tf.expand_dims(tf.convert_to_tensor(prev_action), 0)

    legal_action = train.policy(prev_state, prev_action, model)
    # Recieve state and reward from environment.
    state, prev_action, _, done, _ = env.step(legal_action)

    # End this episode when `done` is True
    if done:
      break

    prev_state = state
    prev_action = legal_action

  cv2.imwrite(filename, env.image)
  #env.render()
  
if __name__ == "__main__":
    get_ground_truth()
    get_ground_truth(True)
