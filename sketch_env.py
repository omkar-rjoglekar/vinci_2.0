import cv2
import gym
from skimage.draw import line_aa
from HyperParameters import HP
import tensorflow as tf
import numpy as np
import os
#from google.colab.patches import cv2_imshow

"""def sample_goals(sample_size=1, sample_dir=HP.goals_dir):
    goal_set = np.array(os.listdir(sample_dir))
    indices = np.random.choice(len(goal_set), sample_size)
    goals_sampled = goal_set[indices]"""

class SketchEnv(gym.Env):
  metadata = {'render.mode': ['human']}

  def __init__(self, input_img=None):
    super(SketchEnv, self).__init__()
    
    #self.start_pt = input_img
    
    """
    if self.start_pt is None:
      self.strokes = [np.array([0,0,1,0,0])]
    else:
      self.strokes = self.start_pt
    """

    self.input_img = input_img
    self.prev_stroke = [0, 0, 0]

    #self.strokes = self.pad_strokes(self.strokes)
    #print("Environment loaded with the input image:")
    #draw.draw_strokes(np.array(self.strokes), svg_filename="./input_img_.svg", 
                        #save_to_file=True, show=True)
    if self.input_img is None:
      self.image = np.zeros((HP.img_dims[0], HP.img_dims[1]))
    else:
      self.image = cv2.resize(cv2.imread(self.input_img, flags=cv2.IMREAD_GRAYSCALE), 
                              (HP.img_dims[0], HP.img_dims[1]), interpolation=cv2.INTER_LINEAR)

    self.time_step = 0
    self.min_reward = -1
    self.max_reward = 1

    self.classifier = tf.keras.models.load_model('./models/classifier3')
    
    generated_img = np.expand_dims(self.image, axis=(0, -1))
    pred = self.classifier.predict(generated_img)
    self.old_prob = pred[0][0]

  def reset(self):
    self.time_step = 0

    """
    if self.start_pt is None:
      self.strokes = [np.array([0,0,1,0,0])]
    else:
      self.strokes = self.start_pt
    """

    if self.input_img is None:
      self.image = np.zeros((HP.img_dims[0], HP.img_dims[1]))
    else:
      self.image = cv2.resize(cv2.imread(self.input_img, flags=cv2.IMREAD_GRAYSCALE), 
                              (HP.img_dims[0], HP.img_dims[1]), interpolation=cv2.INTER_LINEAR)

    self.prev_stroke = [0, 0, 0]
    
    generated_img = np.expand_dims(self.image, axis=(0, -1))
    pred = self.classifier.predict(generated_img)
    self.old_prob = pred[0][0]
    
    #state = {}
    #state['St'] = self.strokes
    state = cv2.resize(self.image, (HP.state_dims[0], HP.state_dims[1]), interpolation=cv2.INTER_LINEAR)

    return state, self.prev_stroke

  def step(self, action):
    self.time_step += 1
    #stroke = self.sample_gaussian(action)
    stroke = [round(action[0]), round(action[1])]

    #self.strokes.append(stroke)

    #draw.draw_strokes(np.array(self.strokes), svg_filename="./current_img_.svg", 
                      #save_to_file=True)
    #print(action)
    draw = np.random.binomial(1, action[2])

    if draw:
        rr, cc, val = line_aa(self.prev_stroke[0], self.prev_stroke[1],
                          stroke[0], stroke[1])
        self.image[rr, cc] = val*255

    prev_action = self.prev_stroke
    self.prev_stroke = [stroke[0], stroke[1], action[2]]

    state = self.image

    #reward = self._get_reward(self.image)
    generated_img = np.expand_dims(self.image, axis=(0, -1))
    new_prob = self.classifier.predict(generated_img)[0][0]

    done = new_prob > 0.97
    
    reward = self._get_reward(new_prob, done)
    self.old_prob = new_prob
    
    return state, prev_action, reward, done, {}

  def _get_reward(self, newp, done):
    diffp = newp - self.old_prob
    if diffp == 0:
        zero_penalty = -1/HP.max_seq_length
    else:
        zero_penalty = 0
    if done:
        diffp += newp
    
    return diffp + zero_penalty

  def get_final_prob(self):
    return self.classifier.predict(np.expand_dims(self.image, axis=(0, -1)))[0][0]

  def render(self, mode='human', close=False):
    cv2_imshow(cv2.resize(self.image, (HP.state_dims[0], HP.state_dims[1]), interpolation=cv2.INTER_LINEAR))
