'Author: Aimore Resende Riquetti Dutra'
'''email: aimorerrd@hotmail.com'''
# -------------------------------------------------------------------------------------------------- #
# This code can run 4 different models of Reinforcement Learning:
# Q-Learning (QL), DQN, SRL (DSRL), SRL+CS(DSRL_object_near) and some other variations of SRL
# The setting for each run can be set at the end of the code
# It can load and save the models in Excel form
# There are some pre-defined environments, but you can create your own
# Press G to get intermediate Graphs and P to stop
# -------------------------------------------------------------------------------------------------- #

import Class
import pprint
import random
import sys
import numpy as np
import pygame
# from pyglet import clock
import pandas as pd
import time
import json
from time import sleep
import math
import matplotlib.pyplot as plt
import os
import glob

## Comment this part if not using DQN model:
# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Flatten
# from keras.models import model_from_json
# from keras.optimizers import sgd
# from keras.utils import plot_model
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# set_session(tf.Session(config=config))

# region COLOR DEFINITION
white = (255, 255, 255)
black = (0, 0, 0)
grey = (80, 80, 80)
red = (255, 0, 0)
blue = (0, 0, 255)
green = (0, 255, 0)
yellow = (250, 250, 0)
pink = (250, 105, 180)
# endregion

# region PANDAS DEFINITION
pd.set_option('display.max_columns', None)
pd.set_option('display.large_repr', 'info')
desired_width = 180
pd.set_option('display.width', desired_width)
pd.set_option('precision', 4)
# endregion

np.random.seed(123)  # For reproducibility
pygame.init()  # Pygame initialialization
pp = pprint.PrettyPrinter(indent=4)
actions = ['up', 'down', 'right', 'left']
actions_dict = {'up':0, 'down':1, 'right':2, 'left':3}
p_keys = [pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d]
# clock.tick(20)

def pop(self):
    '''Removes a layer instance on top of the layer stack.
    '''
    while self.outputs:
        self.layers.pop()
        if not self.layers:
            self.outputs = []
            self.inbound_nodes = []
            self.outbound_nodes = []
        else:
            self.layers[-1].outbound_nodes = []
            self.outputs = [self.layers[-1].output]
        self.built = False

# region REWARDS
negative_reward = 10  # Negative Reward
positive_reward = 1  # Positive Reward
step_reward = 0  # Reward received by each step
# endregion

# region TEXT FONTS DEFINITION
smallfont = pygame.font.SysFont('comicsansms', 13)
smallfont_act = pygame.font.SysFont('arial', 13)
mediumfont_act = pygame.font.SysFont('arial', 18, bold=True)
pygame.font.init()
# endregion

# region DISPLAY FUNCTIONS
def show_Alg(alg, screen):
    text = smallfont.render("Alg: " + alg, True, black)
    screen.blit(text, [5 + 90 * 0, 0])

def show_Samples(sample, screen):
    text = smallfont.render("Sample: " + str(sample), True, black)
    screen.blit(text, [60+100*1, 0])

def show_Level(level, screen):
    text = smallfont.render("Episode: " + str(level), True, black)
    screen.blit(text, [50+100*2, 0])

def show_Score(score, screen):
    text = smallfont.render("Score: " + str(score), True, black)
    screen.blit(text, [50+100*3, 0])

def show_Steps(steps, screen):
    text = smallfont.render("Steps: " + str(steps), True, black)
    screen.blit(text, [50+100*4, 0])

def show_Percent(percent, screen):
    text = smallfont.render("Percent: " + str(['%.2f' % elem for elem in percent]), True, black)
    screen.blit(text, [5, 30 * 4])

def show_Steps_list(steps_list, screen):
    text = smallfont.render("Steps_list: " + str(steps_list), True, black)
    screen.blit(text, [5, 30 * 1])

def show_Act_List(act_list, screen):
    text = smallfont_act.render("act_list: " + str(act_list), True, black)
    screen.blit(text, [5, 30 * 2])

def show_Action(act, screen):
    text = smallfont_act.render("Chosen Action: " + act, True, black)
    screen.blit(text, [5, 30 * 3])

def show_Env(env, screen):
    text = mediumfont_act.render("Environment:  " + str(env), True, black)
    screen.blit(text, [50, 30 * 5])
# endregion

# region CREATE OBJ_LIST FROM STATE AND RELATIONSHIP LIST BETWEEN AGENT AND OBJECTS
''' CREATE obj_list - FROM env '''
def create_obj_list(env):
    obj_list_fun = []
    tp_list = []
    loc_list = []
    env = env.transpose()
    h_max = env.shape[0]
    # print("h_max", h_max)
    v_max = env.shape[1]
    # print("v_max",v_max)
    for h in range(1, (h_max - 1)):
        for v in range(1, (v_max - 1)):
            if env[h][v] != 0:
                tp_list.append(env[h][v])
                loc_list.append((h, v))
    for i in range(len(loc_list)):
        tp = tp_list[i]
        loc = loc_list[i]
        obj = Class.Obj(tp, loc)
        obj_list_fun.append(obj)
    return obj_list_fun

''' CREATE A RELATIONSHIP LIST BETWEEN AGENT AND OBJECTS - FROM obj_list '''
def relation_obj_list(obj_list, agent_pos):
    rel_list = []
    xA = agent_pos[0]
    yA = agent_pos[1]
    # print("xA", xA)
    # print("yA", yA)
    for obj in obj_list:
        xB = obj.loc[0]
        yB = obj.loc[1]
        x = xA - xB
        y = yA - yB
        loc_dif = (x, y)
        # loc_dif = (x[0], y[0])
        tp = obj.tp
        obj = Class.Obj(tp, loc_dif)
        rel_list.append(obj)
    return rel_list
# endregion

# region DRAW OBJECTS
x_zero_screen = 50
y_zero_screen = 180
size_obj = 37
def draw_objects(agent, positivo_list, negativo_list, wall_list, screen):
    # Class.Grid.draw_grid(screen) # Uncomment to display a Grid
    for i in positivo_list:  # POSITIVO
        screen.blit(i.icon, (i.pos[0] * size_obj + x_zero_screen, y_zero_screen + i.pos[1] * size_obj))
    for i in negativo_list:  # NEGATIVO
        screen.blit(i.icon, (i.pos[0] * size_obj + x_zero_screen, y_zero_screen + i.pos[1] * size_obj))
    screen.blit(agent.icon, (agent.pos[0] * size_obj + x_zero_screen, y_zero_screen + agent.pos[1] * size_obj))  # AGENT
    for i in wall_list:  # WALL
        screen.blit(i.icon, (i.pos[0] * size_obj + x_zero_screen, y_zero_screen + i.pos[1] * size_obj))
# endregion

# region CREATE THE STATE FROM THE ENVIRONMENT
def update_state(h_max, v_max, agent, positivo_list, negativo_list, wall_list):
    state = np.zeros((v_max, h_max)).astype(np.int16)
    for i in positivo_list:
        state[i.pos[1]][i.pos[0]] = 60  # SYMBOL 60 POSITIVE
    for i in negativo_list:
        state[i.pos[1]][i.pos[0]] = 180  # SYMBOL 180 NEGATIVE
    for i in wall_list:
        state[i.pos[1]][i.pos[0]] = 255  # SYMBOL 255
    # state[agent.pos[1]][agent.pos[0]] = 120  # SYMBOL 60
    return state
    # TODO I have to check if this v_max and h_max have to be declared eveytime
# endregion

# region ENVIRONMENT CONFIGURATION
def environment_conf(s_env):
    if s_env == 1:
        v_max = 4
        h_max = 5
        x_agent = 1
        y_agent = 2
        m_nega = np.matrix([[0, 0, 0],
                            [0, 1, 0]])
        m_posi = np.matrix([[0, 1, 0],
                            [0, 0, 0]])

    elif s_env == 2:
        v_max = 4
        h_max = 5
        x_agent = 1
        y_agent = 2
        m_nega = np.matrix([[0, 0, 0],
                            [0, 0, 1]])
        m_posi = np.matrix([[0, 0, 1],
                            [0, 0, 0]])

    elif s_env == 3:
        v_max = 4
        h_max = 5
        x_agent = 1
        y_agent = 2
        m_nega = np.matrix([[1, 0, 0],
                            [0, 0, 0]])
        m_posi = np.matrix([[0, 1, 0],
                            [0, 0, 0]])

    elif s_env == 4:
        v_max = 4
        h_max = 4
        x_agent = 1
        y_agent = 1
        m_nega = np.matrix([[0, 0],
                            [0, 0]])
        m_posi = np.matrix([[0, 0],
                            [0, 1]])

    elif s_env == 5:
        v_max = 5
        h_max = 5
        x_agent = 2
        y_agent = 2
        m_nega = np.zeros(shape=(v_max - 2, h_max - 2))
        m_posi = np.zeros(shape=(v_max - 2, h_max - 2))
        while (True):
            x = random.randrange(0, h_max - 2)
            y = random.randrange(0, v_max - 2)
            if x != x_agent-1 or y != y_agent-1:
                element = (x, y)
                break
        m_posi[element] = 1

    elif s_env == 6:
        v_max = 7
        h_max = 7
        x_agent = 3
        y_agent = 3
        m_nega = np.zeros(shape=(v_max - 2, h_max - 2))
        m_posi = np.zeros(shape=(v_max - 2, h_max - 2))
        while (True):
            x = random.randrange(0, h_max - 2)
            y = random.randrange(0, v_max - 2)
            if x != x_agent - 1 or y != y_agent - 1:
                element = (x, y)
                break
        m_posi[element] = 1

    elif s_env == 7:
        v_max = 9
        h_max = 9
        x_agent = 4
        y_agent = 4
        m_nega = np.zeros(shape=(v_max - 2, h_max - 2))
        m_posi = np.zeros(shape=(v_max - 2, h_max - 2))
        while (True):
            x = random.randrange(0, h_max - 2)
            y = random.randrange(0, v_max - 2)
            if x != x_agent - 1 or y != y_agent - 1:
                element = (x, y)
                break
        m_posi[element] = 1

    elif s_env == 8:
        v_max = 5
        h_max = 5
        x_agent = 2
        y_agent = 2
        m_nega = np.matrix([[0, 0, 0],
                            [0, 0, 0],
                            [1, 0, 1]])
        m_posi = np.matrix([[1, 0, 1],
                            [0, 0, 0],
                            [0, 0, 0]])

    elif s_env == 9:
        v_max = 5
        h_max = 5
        x_agent = 2
        y_agent = 2
        m_nega = np.matrix([[1, 0, 0],
                            [0, 0, 0],
                            [0, 0, 1]])
        m_posi = np.matrix([[0, 0, 1],
                            [0, 0, 0],
                            [1, 0, 0]])

    elif s_env == 10:
        v_max = 9
        h_max = 9
        x_agent = 4
        y_agent = 4
        m_nega = np.matrix([[1, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 1]])
        m_posi = np.matrix([[0, 0, 1, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 1, 0, 0]])

    elif s_env == 11:
        v_max = 9
        h_max = 9
        x_agent = 4
        y_agent = 4
        element_list = []
        for n in range(14):
            while(True):
                x = random.randrange(0,7)
                y = random.randrange(0,7)
                if x != 3 and y != 3 and (x,y) not in element_list:
                    element = (x, y)
                    break
            element_list.append(element)

        m_nega = np.zeros(shape=(v_max-2, h_max-2))
        m_posi = np.zeros(shape=(v_max-2, h_max-2))
        half = len(element_list) / 2
        nega_list = element_list[:int(half)]
        posi_list = element_list[int(half):]
        for ele in nega_list:
            m_nega[ele] = 1
        for ele in posi_list:
            m_posi[ele] = 1

    elif s_env == 12:
        v_max = 3
        h_max = 5
        x_agent = 2
        y_agent = 1
        m_nega = np.matrix([1, 0, 0])
        m_posi = np.matrix([0, 0, 1])

    elif s_env == 13:
        v_max = 3
        h_max = 5
        x_agent = 2
        y_agent = 1
        m_nega = np.matrix([0, 0, 0])
        m_posi = np.matrix([1, 0, 1])

    elif s_env == 14:
        v_max = 3
        h_max = 6
        x_agent = 2
        y_agent = 1
        m_nega = np.matrix([1, 0, 0, 0])
        m_posi = np.matrix([0, 0, 0, 1])

    elif s_env == 15:
        v_max = 3
        h_max = 6
        x_agent = 2
        y_agent = 1
        m_nega = np.matrix([0, 0, 0, 0])
        m_posi = np.matrix([1, 0, 0, 1])

    elif s_env == 16:
        v_max = 3
        h_max = 7
        x_agent = 3
        y_agent = 1
        m_nega = np.matrix([1, 0, 0, 0, 0])
        m_posi = np.matrix([0, 0, 0, 0, 1])

    elif s_env == 17:
        v_max = 3
        h_max = 7
        x_agent = 3
        y_agent = 1
        m_nega = np.matrix([0, 0, 0, 0, 0])
        m_posi = np.matrix([1, 0, 0, 0, 1])

    elif s_env == 18:
        v_max = 3
        h_max = 9
        x_agent = 4
        y_agent = 1
        m_nega = np.matrix([1, 0, 0, 0, 0, 0, 0])
        m_posi = np.matrix([0, 0, 0, 0, 0, 0, 1])

    elif s_env == 19:
        v_max = 3
        h_max = 9
        x_agent = 4
        y_agent = 1
        m_nega = np.matrix([0, 0, 0, 0, 0, 0, 0])
        m_posi = np.matrix([1, 0, 0, 0, 0, 0, 1])

    elif s_env == 20:
        v_max = 5
        h_max = 5
        x_agent = 2
        y_agent = 2
        m_nega = np.matrix([[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]])
        m_posi = np.matrix([[1, 0, 1],
                            [0, 0, 0],
                            [0, 1, 0]])

    elif s_env == 21:
        v_max = 5
        h_max = 5
        x_agent = 2
        y_agent = 2
        m_nega = np.matrix([[0, 1, 0],
                            [0, 0, 0],
                            [1, 0, 1]])
        m_posi = np.matrix([[1, 0, 1],
                            [0, 0, 0],
                            [0, 1, 0]])

    elif s_env == 22:
        v_max = 5
        h_max = 5
        x_agent = 2
        y_agent = 2
        m_nega = np.matrix([[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]])
        m_posi = np.matrix([[1, 0, 1],
                            [0, 0, 0],
                            [1, 0, 1]])

    if s_env == 31:
        v_max = 5
        h_max = 5
        x_agent = 1
        y_agent = 2
        m_nega = np.matrix([[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]])
        m_posi = np.matrix([[0, 1, 0],
                            [0, 0, 0],
                            [0, 0, 0]])

    elif s_env == 32:
        v_max = 5
        h_max = 5
        x_agent = 1
        y_agent = 2
        m_nega = np.matrix([[0, 0, 0],
                            [0, 0, 1],
                            [0, 0, 0]])
        m_posi = np.matrix([[0, 0, 1],
                            [0, 0, 0],
                            [0, 0, 0]])

    elif s_env == 33:
        v_max = 5
        h_max = 5
        x_agent = 1
        y_agent = 2
        m_nega = np.matrix([[1, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]])
        m_posi = np.matrix([[0, 1, 0],
                            [0, 0, 0],
                            [0, 0, 0]])

    else:
        pass

    "INSTANCE THE wall_list"
    wall_list = []
    for y in range(v_max):
        for x in range(h_max):
            if y == v_max - 1 or y == 0 or x == h_max - 1 or x == 0:
                wall = Class.Wall('wall', x, y)
                wall_list.append(wall)
    "INSTANCE THE AGENT"
    agent = Class.Agent('agent', x_agent, y_agent)

    "INSTANCE POSITIVE OBJECTS"
    positivo_list = []
    for x in range(m_posi.shape[0]):
        for y in range(m_posi.shape[1]):
            if m_posi[x, y] == 1:
                positivo = Class.Positivo('positivo', y + 1, x + 1)
                positivo_list.append(positivo)

    "INSTANCE NEGATIVE OBJECTS"
    negativo_list = []
    for x in range(m_nega.shape[0]):
        for y in range(m_nega.shape[1]):
            if m_nega[x, y] == 1:
                negativo = Class.Negativo('negativo', y + 1, x + 1)
                negativo_list.append(negativo)

    return negativo_list, positivo_list, agent, wall_list, h_max, v_max
# endregion

# region SAVE - LOAD - CREATE
def save_model(model, path):
    model.save_weights(path + ".h5", overwrite=True)
    with open(path + ".json", "w") as outfile:
        json.dump(model.to_json(), outfile)

def load_model(s_alg, path):
    optimizer_config = []
    print(path)
    if s_alg == "QL":
        path = path + ".xlsx"
        model = pd.read_excel(path, sheetname="model")

    elif s_alg == "DSRL":
        path = path + ".xlsx"
        model = pd.read_excel(path, sheetname="model", header=[0], index_col=[0,1])

    elif s_alg == "DSRL_dist":
        path = path + ".xlsx"
        model = pd.read_excel(path, sheetname="model", header=[0], index_col=[0,1])

    elif s_alg == "DSRL_dist_type":
        path = path + ".xlsx"
        model = pd.read_excel(path, sheetname="model", header=[0], index_col=[0,1])

    elif s_alg == "DSRL_dist_type_near":
        path = path + ".xlsx"
        model = pd.read_excel(path, sheetname="model", header=[0], index_col=[0,1])

    elif s_alg == "DSRL_dist_type_near_propNeg":
        path = path + ".xlsx"
        model = pd.read_excel(path, sheetname="model", header=[0], index_col=[0,1])

    elif s_alg == "DSRL_object_near":
        path = path + ".xlsx"
        model = pd.read_excel(path, sheetname="model", header=[0], index_col=[0,1])

    elif s_alg == "DSRL_object":
        path = path + ".xlsx"
        model = pd.read_excel(path, sheetname="model", header=[0], index_col=[0, 1])

    elif s_alg == "DQN":
        with open(path + ".json", "r") as jfile:
            model = model_from_json(json.load(jfile))
        model.load_weights(path + ".h5")
        conf = pd.read_excel(path + ".xlsx", sheetname="Run_Conf", header=[0])
        # net_conf = conf.loc[[16:20],:]
        # print("net_conf", net_conf)
        optimizer = conf.loc[19, "A"]
        print("op_conf ", optimizer)
        # pd.Series({'N_actions': net_conf["N_actions"]}),
        # pd.Series({'Max_memory': net_conf["Max_memory"]}),
        # pd.Series({'Hidden_size': net_conf["Hidden_size"]}),
        # pd.Series({'Batch_size': net_conf["Batch_size"]}),
        # pd.Series({'Optimizer': net_conf["Optimizer"]}),
        # pd.Series({'lr': op_conf[0]}),
        # pd.Series({'beta_1': op_conf[1]}),
        # pd.Series({'beta_2': op_conf[2]}),
        # pd.Series({'epsilon': op_conf[3]}),
        # pd.Series({'decay': op_conf[4]}),
        # pd.Series({'rho': op_conf[5]})

        use_optimizer, optimizer_config = define_optimizer(optimizer)
        model.compile(loss='mse', optimizer=use_optimizer)
        model.summary()
        # pass
    return model, optimizer_config

def create_model(s_alg, state_shape, net_conf):
    optimizer_config = []
    if s_alg == "QL":
        model = pd.DataFrame()
        model.index.name = ["States", "Action"]

    elif s_alg == "DSRL" or s_alg == "DSRL_dist" or s_alg == "DSRL_dist_type" or s_alg == "DSRL_dist_type_near" or s_alg == "DSRL_dist_type_near_propNeg" or s_alg == "DSRL_object_near" or s_alg == "DSRL_object":
        m_index = pd.MultiIndex(levels=[[''], [""]],
                                labels=[[], []],
                                names=['state', 'actions'])
        model = pd.DataFrame(index=m_index)

    elif s_alg == "DQN":
        model = Sequential()
        pop(model)
        model = Sequential()
        model.add(Dense(net_conf["Hidden_size"],
                        input_dim=state_shape[0]*state_shape[1],
                        activation="relu",
                        name="DENSE_1"))

        model.add(Dense(net_conf["Hidden_size"],
                        activation='relu',
                        name="DENSE_2"))

        model.add(Dense(net_conf["N_actions"],
                        name="DENSE_3"))

        use_optimizer, optimizer_config = define_optimizer(net_conf["Optimizer"])
        model.compile(loss='mse', optimizer=use_optimizer)
        print(model.summary())
        # plot_model(model, to_file='model.png')
        # d3v.d3viz(model.get_output(), 'test.html')
    return model, optimizer_config

# endregion

# region DQN - CONFIGURATIONS
class ExperienceReplay(object):
    """
    During gameplay all the experiences < s, a, r, s’ > are stored in a replay memory.
    In training, batches of randomly drawn experiences are used to generate the input and target for training.
    """

    def __init__(self, max_memory=100, discount=.9):
        """
        Setup
        max_memory: the maximum number of experiences we want to store
        memory: a list of experiences
        discount: the discount factor for future experience

        In the memory the information whether the game ended at the state is stored seperately in a nested array
        [...
        [experience, game_over]
        [experience, game_over]
        ...]
        """
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, game_over):
        # Save a state to memory
        self.memory.append([states, game_over])
        # We don't want to store infinite memories, so if we have too many, we just delete the oldest one
        if len(self.memory) > self.max_memory:
            del self.memory[0]

        # print(">>> states:", states)

    def get_batch(self, model, batch_size=10):
        # How many experiences do we have?
        len_memory = len(self.memory)

        # Calculate the number of actions that can possibly be taken in the game
        num_actions = model.output_shape[-1]

        # Dimensions of the game field
        env_dim = self.memory[0][0][0].shape[1]

        # We want to return an input and target vector with inputs from an observed state...
        inputs = np.zeros((min(len_memory, batch_size), env_dim))

        # ...and the target r + gamma * max Q(s’,a’)
        # Note that our target is a matrix, with possible fields not only for the action taken but also for the other possible actions.
        # The actions not take the same value as the prediction to not affect them
        targets = np.zeros((inputs.shape[0], num_actions))

        # We draw states to learn from randomly
        for i, idx in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
            """
            Here we load one transition <s, a, r, s’> from memory
            state_t: initial state s
            action_t: action taken a
            reward_t: reward earned r
            state_tp1: the state that followed s’
            """
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            # We also need to know whether the game ended at this state
            game_over = self.memory[idx][1]

            inputs[i:i + 1] = state_t

            # First we fill the target values with the predictions of the model.
            # They will not be affected by training (since the training loss for them is 0)
            targets[i] = model.predict(state_t)[0]
            # print("targets\n", targets)
            # print("action_t", action_t)
            """
            If the game ended, the expected reward Q(s,a) should be the final reward r.
            Otherwise the target value is r + gamma * max Q(s’,a’)
            """
            #  Here Q_sa is max_a'Q(s', a')
            Q_sa = np.max(model.predict(state_tp1)[0])

            # if the game ended, the reward is the final reward
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # r + gamma * max Q(s’,a’)
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets

def define_optimizer(s_optimizer):
    lr = 0
    beta_1 = 0
    beta_2 = 0
    epsilon = 0
    decay = 0
    rho = 0
    if s_optimizer == "adam":
        lr = 0.001  # 0.001
        beta_1 = 0.9  # 0.9
        beta_2 = 0.999  # 0.999
        epsilon = 1e-08  # 1e-08
        decay = 0.0  # 0.0
        optimizer_selected = keras.optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay)
    elif s_optimizer == "rms_opt":
        lr = 0.001  # 0.001
        rho = 0.9  # 0.9
        epsilon = 1e-08  # e-08
        decay = 0.0  # 0.0
        optimizer_selected = keras.optimizers.RMSprop(lr=lr, rho=rho, epsilon=epsilon, decay=decay)
    optimizer_config = [lr, beta_1, beta_2, epsilon, decay, rho]
    return optimizer_selected, optimizer_config
#

def choose_action(s_alg, state, agent_pos, model, s_prob):
    # print("\nPREVIOUS MODEL - CHOOSE ACTION\n", model)
    zero = False
    if s_alg == "QL":
        state[agent_pos[1]][agent_pos[0]] = 120
        s = str(state)
        if s not in model.index:
            indices = [np.array([s, s, s, s]), np.array(['up', 'down', 'right', 'left'])]
            df_zero = pd.DataFrame(np.zeros([4, 1]), index=indices)
            model = model.append(df_zero)
            model = model.fillna(0)
        n_action = np.argmax(model.loc[s][0])  # Choose the max argument
        if max(model.loc[s][0]) == 0: zero = True

    elif s_alg == "DSRL" or s_alg == "DSRL_dist" or s_alg == "DSRL_dist_type" or s_alg == "DSRL_dist_type_near" or s_alg == "DSRL_dist_type_near_propNeg" or s_alg == "DSRL_object_near" or s_alg == "DSRL_object":
        a_v_list = []
        d = {}
        obj_list = create_obj_list(state)
        rel_list = relation_obj_list(obj_list, agent_pos)
        new_state = rel_list

        for obj in new_state: # FOR ALL OBJECTS SEEN
            tp_n_c = str(obj.tp) # GET THE TYPE FROM THE NEW STATE
            s_n_c = str(obj.loc) # GET THE LOCATION FROM THE NEW STATE
            if tp_n_c not in model.columns:
                # print("tp_n_c not in model.columns", tp_n_c)
                model[tp_n_c] = 0
            if s_n_c not in model.index:
                # print("s_n_c not in model.index", s_n_c)
                m_index = pd.MultiIndex(levels=[[s_n_c], actions],
                                        labels=[[0, 0, 0, 0], [0, 1, 2, 3]],
                                        names=['state', 'actions'])
                df_zero = pd.DataFrame(index=m_index)
                model = model.append(df_zero)
                model = model.fillna(0)
            Qts_a = model[tp_n_c].loc[s_n_c]
            # print("Qts_a - ", Qts_a)
            if s_alg == "DSRL_dist_type_near" or s_alg == "DSRL_dist_type_near_propNeg" or s_alg == "DSRL_object_near": # Calculate the distance
                s_n_c_abs = [int(s) for s in s_n_c if s.isdigit()]  # s_n_c_abs = state_new_absolute_distance
                distance = np.sqrt(s_n_c_abs[0]**2 + s_n_c_abs[1]**2)
                # print("distance",distance)
                Qts_a = Qts_a.divide(distance*distance, axis=0)
            a_v = [(value, key) for value, key in Qts_a.items()]
            # print("Qts_a - NEW", Qts_a)
            a_v_list.append(a_v) # Append Q-value

        # Sum the values of all Qs into a single Q
        for element in a_v_list:
            for a in element:
                act = a[0] # Action
                val = a[1] # Value
                d[act] = d.get(act, 0) + val # Sum values for each Q

        # print('a_v_list: (List of the action values for each object in the scene): ')
        # print('{0}'.format(a_v_list))
        # print('\nd: (The sum of all object`s action values )')
        # pp.pprint(d)

        if d != {}: # BE CAREFUL THIS IS A DICT (argmax does not work as usual)
            inverse = [(value, key) for key, value in d.items()] # CALCULATE ALL KEYS
            n_action = max(inverse)[1] # Choose the max argument

            if max(d.values()) == 0: zero = True
        else:
            n_action = "down"

    elif s_alg == "DQN":
        state[agent_pos[1]][agent_pos[0]] = 120
        state = state.reshape((1, -1))
        q = model.predict(state)
        n_act = np.argmax(q[0])
        n_action = actions[n_act]
        if max(q[0]) == 0: zero = True

    x = random.random()  # E greedy exploration
    if x < s_prob:
        n_action = random.choice(actions)
        print_action = 'Random Act (Prob):'
    elif zero == True:
        n_action = random.choice(actions)
        print_action = 'Random Act (Zero):'
    else:
        print_action = 'Chosen Act:'
    # print("\nNEW MODEL - CHOOSE ACTION\n", model)
    return n_action, model, print_action

alfa = 1 # Learning Rate
gamma = 0.9 # Temporal Discount Factor
def learn(s_alg, model, state_t, state_t1, agent_t_pos, agent_t1_pos, reward, action_t, end_game, net_conf, exp_replay):
    # print("\nPREVIOUS MODEL - LEARN\n", model)
    batch_loss = 0
    if s_alg == "QL":
        state_t[agent_t_pos[1]][agent_t_pos[0]] = 120
        state_t1[agent_t1_pos[1]][agent_t1_pos[0]] = 120
        s_t = str(state_t)
        s_t1 = str(state_t1)
        if s_t1 not in model.index:
            indices = [np.array([s_t1, s_t1, s_t1, s_t1]), np.array(['up', 'down', 'right', 'left'])]
            df_zero = pd.DataFrame(np.zeros([4, 1]), index=indices)
            model = model.append(df_zero)
        if s_t not in model.index:
            indices = [np.array([s_t, s_t, s_t, s_t]), np.array(['up', 'down', 'right', 'left'])]
            df_zero = pd.DataFrame(np.zeros([4, 1]), index=indices)
            model = model.append(df_zero)
        model = model.fillna(0)

        if end_game == False:
            max_value = max(model.loc[s_t1][0])  # max(df.loc[new_state][0])
            Q_value = model.loc[s_t, action_t][0]
            updated_model = Q_value + alfa * (reward + (gamma * (max_value)) - Q_value)
        else:
            updated_model = reward
        model.loc[s_t, action_t] = updated_model

    elif s_alg == "DSRL" or s_alg == "DSRL_dist" or s_alg == "DSRL_dist_type" or s_alg == "DSRL_dist_type_near" or s_alg == "DSRL_dist_type_near_propNeg" or s_alg == "DSRL_object_near" or s_alg == "DSRL_object":
        max_value = 0

        obj_list = create_obj_list(state_t)
        rel_list = relation_obj_list(obj_list, agent_t_pos)
        old_state = rel_list

        obj_list = create_obj_list(state_t1)
        rel_list = relation_obj_list(obj_list, agent_t1_pos)
        new_state = rel_list

        for i in range(len(old_state)):
            # Check all items in old state
            obj_prev = old_state[i]
            tp_prev = str(obj_prev.tp)
            s_prev = str(obj_prev.loc)
            # Check all items in new state
            obj_new = new_state[i]
            tp_new = str(obj_new.tp)
            s_new = str(obj_new.loc)

            if tp_new not in model.columns: # If type is new, then add type
                model[tp_new] = 0
            if s_new not in model.index: # If state is new, then add state
                m_index = pd.MultiIndex(levels=[[s_new], actions],
                                        labels=[[0, 0, 0, 0], [0, 1, 2, 3]],
                                        names=['state', 'actions'])
                df_zero = pd.DataFrame(index=m_index)
                model = model.append(df_zero)
                model = model.fillna(0)

            max_value = max(model[tp_new].loc[s_new])
            if s_alg == "DSRL": # THEY STILL HAVE THE PROBLEM OF NOT PROPAGATING THE NEGATIVE SIGNAL
                if end_game == False:
                    Q_v = model[tp_prev].loc[s_prev, action_t]
                    model[tp_prev].loc[s_prev, action_t] = Q_v + alfa * (reward + (gamma * max_value) - Q_v)
                else:
                    model[tp_prev].loc[s_prev, action_t] = reward

            elif s_alg == "DSRL_dist": # THEY STILL HAVE THE PROBLEM OF NOT PROPAGATING THE NEGATIVE SIGNAL
                if reward != 0:
                    s_p_c = [int(s) for s in s_prev if s.isdigit()]
                    if s_p_c[0] < 2 and s_p_c[1] < 2:
                        # EDITIONG DELETE
                        if end_game == False:
                            Q_v = model[tp_prev].loc[s_prev, action_t]
                            model[tp_prev].loc[s_prev, action_t] = Q_v + alfa * (reward + (gamma * max_value) - Q_v)
                        else:
                            model[tp_prev].loc[s_prev, action_t] = reward
                else:
                    if end_game == False:
                        Q_v = model[tp_prev].loc[s_prev, action_t]
                        model[tp_prev].loc[s_prev, action_t] = Q_v + alfa * (reward + (gamma * max_value) - Q_v)
                    else:
                        model[tp_prev].loc[s_prev, action_t] = reward

            elif s_alg == "DSRL_dist_type" or s_alg == "DSRL_dist_type_near": # THEY STILL HAVE THE PROBLEM OF NOT PROPAGATING THE NEGATIVE SIGNAL
                max_value_positive = max(model[tp_new].loc[s_new])
                if reward != 0:
                    s_p_c = [int(s) for s in s_prev if s.isdigit()]  # s_p_c = state_previous_absolute_distance
                    if s_p_c[0] < 2 and s_p_c[1] < 2: # IF IT IS CLOSE BY, THEN UPDATE ONLY THE CLOSE ONE:
                        if reward < 0 and tp_new == "180": # IF REWARD IS NEGATIVE and NEW OBJECT IS NEGATIVE UPDATE ONLY NEGATIVE TYPE:
                            if end_game == False:
                                Q_v = model[tp_prev].loc[s_prev, action_t]
                                model[tp_prev].loc[s_prev, action_t] = Q_v + alfa * (reward + (gamma * max_value_positive) - Q_v)
                            else:
                                model[tp_prev].loc[s_prev, action_t] = reward
                        elif reward > 0 and tp_new == "60":  # IF REWARD IS POSITIVE and NEW OBJECT IS POSITIVE UPDATE ONLY POSITIVE TYPE:
                            if end_game == False:
                                Q_v = model[tp_prev].loc[s_prev, action_t]
                                model[tp_prev].loc[s_prev, action_t] = Q_v + alfa * (reward + (gamma * max_value_positive) - Q_v)
                            else:
                                model[tp_prev].loc[s_prev, action_t] = reward
                # IF reward is zero
                else:
                    if end_game == False:
                        Q_v = model[tp_prev].loc[s_prev, action_t]
                        if tp_prev == "180": # IF THE PREVIOUS OBJECT WAS NEGATIVE
                            model[tp_prev].loc[s_prev, action_t] = Q_v + alfa * (reward + (gamma * max_value_positive) - Q_v)
                        elif tp_prev == "60": # IF THE PREVIOUS OBJECT WAS POSITIVE
                            model[tp_prev].loc[s_prev, action_t] = Q_v + alfa * (reward + (gamma * max_value_positive) - Q_v)
                    else:
                        model[tp_prev].loc[s_prev, action_t] = reward

            elif s_alg == "DSRL_dist_type_near_propNeg": # I try to solve this with max and min, but it did not work very well(THEY STILL HAVE THE PROBLEM OF NOT PROPAGATING THE NEGATIVE SIGNAL)
                max_value_positive = max(model[tp_new].loc[s_new])
                min_value_negative = min(model[tp_new].loc[s_new])
                if reward != 0:
                    s_p_c = [int(s) for s in s_prev if s.isdigit()]  # s_p_c = state_previous_absolute_distance
                    if s_p_c[0] < 2 and s_p_c[1] < 2: # IF IT IS CLOSE BY, THEN UPDATE ONLY THE CLOSE ONE:
                        if reward < 0 and tp_new == "180": # IF REWARD IS NEGATIVE and NEW OBJECT IS NEGATIVE UPDATE ONLY NEGATIVE TYPE:
                            if end_game == False:
                                Q_v = model[tp_prev].loc[s_prev, action_t]
                                model[tp_prev].loc[s_prev, action_t] = Q_v + alfa * (reward + (gamma * min_value_negative) - Q_v)
                            else:
                                model[tp_prev].loc[s_prev, action_t] = reward
                        elif reward > 0 and tp_new == "60":  # IF REWARD IS POSITIVE and NEW OBJECT IS POSITIVE UPDATE ONLY POSITIVE TYPE:
                            if end_game == False:
                                Q_v = model[tp_prev].loc[s_prev, action_t]
                                model[tp_prev].loc[s_prev, action_t] = Q_v + alfa * (reward + (gamma * max_value_positive) - Q_v)
                            else:
                                model[tp_prev].loc[s_prev, action_t] = reward
                # IF reward is zero
                else:
                    if end_game == False:
                        Q_v = model[tp_prev].loc[s_prev, action_t]
                        if tp_prev == "180": # IF THE PREVIOUS OBJECT WAS NEGATIVE
                            model[tp_prev].loc[s_prev, action_t] = Q_v + alfa * (reward + (gamma * min_value_negative) - Q_v)
                        elif tp_prev == "60": # IF THE PREVIOUS OBJECT WAS POSITIVE
                            model[tp_prev].loc[s_prev, action_t] = Q_v + alfa * (reward + (gamma * max_value_positive) - Q_v)
                    else:
                        model[tp_prev].loc[s_prev, action_t] = reward

            elif s_alg == "DSRL_object_near" or s_alg == "DSRL_object":
                max_value_positive = max(model[tp_new].loc[s_new])

                # Find the object that the agent interacted with:
                # This means that the agents has to know that the object which interacted with
                # After finding it, he has to assign the value to that object.
                # This means that I have to find the type and the state of this object that has now x=zero y=zero

                # print("obj_new.loc[0]\n", obj_new.loc[0])
                # print("obj_new.loc[1]\n", obj_new.loc[1])
                # print("action_t\n", action_t)
                # print("s_prev\n", s_prev)

                if obj_new.loc[0] == 0 and obj_new.loc[1] == 0:
                    tp_to_update = tp_new
                    # print("tp_new\n", tp_new)
                    if action_t == "up":
                        s_prev_to_update = str((0,1))
                    elif action_t == "down":
                        s_prev_to_update = str((0,-1))
                    elif action_t == "right":
                        s_prev_to_update = str((-1,0))
                    elif action_t == "left":
                        s_prev_to_update = str((1,0))
                    # print("s_prev_to_update\n", s_prev_to_update)
                    if end_game == False:
                        Q_v = model[tp_to_update].loc[s_prev_to_update, action_t]
                        model[tp_to_update].loc[s_prev_to_update, action_t] = Q_v + alfa * (reward + (gamma * max_value_positive) - Q_v)
                    else:
                        model[tp_to_update].loc[s_prev_to_update, action_t] = reward

                if reward == 0:
                    if end_game == False:
                        Q_v = model[tp_prev].loc[s_prev, action_t]
                        model[tp_prev].loc[s_prev, action_t] = Q_v + alfa * (reward + (gamma * max_value_positive) - Q_v)
                    else:
                        model[tp_prev].loc[s_prev, action_t] = reward

    elif s_alg == "DQN":
        state_t[agent_t_pos[1]][agent_t_pos[0]] = 120
        state_t1[agent_t1_pos[1]][agent_t1_pos[0]] = 120
        state_t = state_t.reshape((1, -1))
        state_t1 = state_t1.reshape((1, -1))
        action_t = actions_dict[action_t]
        exp_replay.remember([state_t, action_t, reward, state_t1], end_game) # [old_state, old_action, reward, new_state]
        inputs, targets = exp_replay.get_batch(model, batch_size=net_conf["Batch_size"])
        batch_loss = model.train_on_batch(inputs, targets)

    # print("\nNEW MODEL - LEARN\n", model)
    return model, batch_loss, exp_replay

''' PROGRAM START '''
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
def run(s_env, s_alg, s_learn, s_load, s_print, s_auto, s_episode, s_cond_to_end, s_server, s_net_comb_param, s_load_path, s_prob, s_sample, s_save):
    net_conf = {"N_actions": n_actions,
                "Max_memory": max_memory_list[s_net_comb_param],
                "Hidden_size": hidden_size_list[s_net_comb_param],
                "Batch_size": batch_size_list[s_net_comb_param],
                "Optimizer": optimizer_list[0]}
    exp_replay = ExperienceReplay(max_memory=net_conf["Max_memory"])
    begin = time.time()
    begin_time = time.strftime('%X %x')
    print("\n\n --- BEGINING --- s_sample: %s \n begin_time: %s \n" % (s_sample, begin_time))

    df_score = pd.DataFrame()
    df_percent_list = pd.DataFrame()
    df_loss_list = pd.DataFrame()
    df_time_sample = pd.DataFrame()
    avg_last_score_list = []

    if s_server == False: screen = pygame.display.set_mode((400 + 37 * 5, 330 + 37 * 5))

    score_list_best = [0]
    for sample in list(range(1, s_sample+1)):
        experiment_configurations = (sample, s_env, s_alg, s_episode, s_learn, s_load, s_print, s_auto, s_cond_to_end, s_server, s_net_comb_param, s_prob)
        print("\n - START - "
              "\n sample: %s"
              "\n s_env: %s"
              "\n s_alg: %s"
              "\n s_episode: %s"
              "\n s_learn: %s"
              "\n s_load: %s"
              "\n s_print: %s"
              "\n s_auto: %s"
              "\n s_cond_to_end: %s"
              "\n s_server: %s"
              "\n s_net_comb_param: %s"
              "\n s_prob: %s" % experiment_configurations)

        start = time.time()
        start_time = time.strftime('%X %x')
        print("\nStart time: ", start_time)
        negativo_list, positivo_list, agent, wall_list, h_max, v_max = environment_conf(s_env)

        env_dim = [h_max, v_max]
        if s_load == True:
            try:
                model, op_conf = load_model(s_alg, __location__ + s_load_path)
            except:
                print("DID NOT FIND THE FILE")
        else:
            model, op_conf = create_model(s_alg, env_dim, net_conf)

        # region INITIALIZE VARIABLES 1
        percent_list = []
        score = 0
        score_list = []
        episodes = 0
        episodes_list = []
        steps = 0
        steps_list = []
        batch_loss = 0
        loss_list = []
        # endregion

        while (episodes < s_episode):  # max_episodes
            negativo_list, positivo_list, agent, wall_list, h_max, v_max = environment_conf(s_env)
            # region INITIALIZE VARIABLES 2
            episodes += 1
            episodes_list.append(episodes)
            max_steps = 10
            steps_list.append(steps)
            steps = 0
            act_list = []
            last_move = False
            action_chosen = ""
            encountered = 0
            pos_collected = 0
            prob = s_prob
            # endregion

            if s_server == False:
                # region DRAW SCREEN
                screen.fill(white)
                show_Alg(s_alg, screen)
                show_Samples(sample, screen)
                show_Level(episodes, screen)
                show_Score(score, screen)
                show_Steps(steps, screen)
                show_Percent(percent_list[-10:], screen)
                show_Steps_list(steps_list[-30:], screen)
                show_Act_List(act_list[-20:], screen)
                show_Action(action_chosen, screen)
                show_Env(s_env, screen)
                draw_objects(agent, positivo_list, negativo_list, wall_list, screen)
                pygame.display.flip()
                # endregion

            while (True):  # max_steps or condition to finish
                sleep(speed)
                ''' EVENT HANDLE '''
                key_pressed = False
                set_action = False
                while (s_server == False):
                    for event in pygame.event.get():
                        # QUIT GAME
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
                        # ADD OR DELETE WALL
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            pass
                            # if (pygame.mouse.get_pressed() == (1, 0, 0)):  # LEFT BUTTON (add wall)
                            #     pos = pygame.mouse.get_pos()
                            #     x = (pos[0] - x_g) / (m + w)
                            #     y = (pos[1] - y_g) / (m + h)
                            #     x = math.trunc(x)
                            #     y = math.trunc(y)
                            #     w_has = False
                            #     for item in wall_list:
                            #         if math.trunc((item[0] - x_g) / (m + w)) == x and math.trunc(
                            #                         (item[1] - y_g) / (m + h)) == y:
                            #             w_has = True
                            #     if w_has == False:
                            #         wall = Class.Wall('wall', x, y)
                            #         print('wall ', wall, 'added')
                            #         wall_list.append(wall)

                            # if (pygame.mouse.get_pressed() == (0, 0, 1)):  # RIGHTBUTTON (delete wall)
                            #     pos = pygame.mouse.get_pos()
                            #     x = (pos[0] - x_g) / (m + w)
                            #     y = (pos[1] - y_g) / (m + h)
                            #     x = math.trunc(x)
                            #     y = math.trunc(y)
                            #     wall = Class.Wall('wall', x, y)
                            #     for i in wall_list:
                            #         if i == wall:
                            #             wall_list.remove(wall)
                            #             print('wall ', wall, 'removed')

                            # EVENT - ANY PRESSED KEY
                        # PRESS A KEY
                        if event.type == pygame.KEYDOWN:
                            # SAVE AND QUIT - KEY P
                            if event.key == pygame.K_p:
                                pygame.quit()
                                sys.exit()
                            # PLOT AGENT`S PERFORMENCE - KEY G
                            if event.key == pygame.K_g:
                                plt.plot(score_list)
                                plt.ylabel('Score')
                                plt.xlabel('Total Steps')
                                plt.title('Performance of the Agent')
                                plt.show()

                                plt.plot(percent_list)
                                plt.ylabel('Percentage of objects +')
                                plt.xlabel('Total Steps')
                                plt.title('Episode over 100 times step each')
                                plt.show()
                                if s_alg == "DQN":
                                    plt.plot(loss_list)
                                    plt.ylabel('loss')
                                    plt.xlabel('Total Steps')
                                    plt.title('batch_loss')
                                    plt.show()
                            # MOVE - SPACE BAR
                            if event.key == pygame.K_SPACE:
                                key_pressed = True
                                break
                            # MOVE - ARROW KEYS
                            if event.key in p_keys:
                                key_pressed = True
                                set_action = True
                                if event.key == pygame.K_w:  # North # add_act('↑') ⇦ ⇨ ⇧ ⇩
                                    key_action = "up"
                                if event.key == pygame.K_s:  # South # add_act('↓') ⬅  ➡  ⬆ ⬇
                                    key_action = "down"
                                if event.key == pygame.K_d:  # West # add_act('→')
                                    key_action = "right"
                                if event.key == pygame.K_a:  # East # add_act('←')
                                    key_action = "left"
                                break
                    # Run game if key is preseed or automatic is selected
                    if key_pressed or s_auto:
                        break
                # BREAK IF IT WAS THE LAST MOVE
                if last_move == True:
                    break
                # RUN_GAME
                steps += 1
                ''' OLD STATE - S 1 - 1'''
                state_t = update_state(h_max, v_max, agent, positivo_list, negativo_list, wall_list)
                agent_t = agent.pos
                ''' CHOOSE ACTION - AGENT ACT - 2'''
                action_chosen, model, print_action = choose_action(s_alg, state_t, agent_t, model, prob)

                if set_action: action_chosen = key_action

                ''' CHANGE THE WORLD - UP_ENV - 3'''
                agent.try_move(action_chosen, wall_list)
                act_list.append(action_chosen)
                if s_print: print(print_action, action_chosen)

                ''' NEW STATE - S2 - 4'''
                state_t1 = update_state(h_max, v_max, agent, positivo_list, negativo_list, wall_list)
                agent_t1 = agent.pos
                if s_print:
                    print('\n>>>>   Level: ' + str(episodes) + ' |  Step: ' + str(
                        steps) + ' |  New_agent_pos: ' + str(agent.pos) + '  <<<<')

                ''' GET REWARD - 5 '''
                # region GET REWARD AND DELETE COLLECTED OBJECT
                prev_score = score
                score += step_reward

                for positivo in positivo_list:
                    if agent.pos == positivo.pos:
                        encountered += 1
                        pos_collected += 1
                        score += positive_reward
                        positivo = Class.Positivo('positivo', agent.pos[0], agent.pos[1])
                        positivo_list.remove(positivo)
                        if s_print == True and s_server == False:
                            print('                                 Hit the Positivo')
                for negativo in negativo_list:
                    if agent.pos == negativo.pos:
                        encountered += 1
                        score -= negative_reward
                        negativo = Class.Negativo('negativo', agent.pos[0], agent.pos[1])
                        negativo_list.remove(negativo)
                        if s_print == True and s_server == False:
                            print('                                 Hit the Negativo')

                new_score = score
                score_list.append(score)
                reward = new_score - prev_score
                # endregion

                ''' LEARN - 6 '''
                # CONDITION TO FINISH THE Episode
                if s_cond_to_end == 'max_steps':
                    if steps == max_steps:
                        last_move = True
                elif s_cond_to_end == 'coll_all' or steps > max_steps:
                    if len(positivo_list) == 0 and len(negativo_list) == 0 or steps > max_steps:
                        last_move = True
                elif s_cond_to_end == 'only_positive' or steps > max_steps:
                    if len(positivo_list) == 0 or steps > max_steps:
                        last_move = True
                elif s_cond_to_end == 'only_negative' or steps > max_steps:
                    if len(negativo_list) == 0 or steps > max_steps:
                        last_move = True

                # LEARN
                if s_learn == True:
                    action_t = action_chosen
                    if last_move == False:
                        ''' LEARN '''
                        model, batch_loss, exp_replay = learn(s_alg, model, state_t, state_t1, agent_t, agent_t1, reward, action_t, False, net_conf, exp_replay)
                    else:
                        ''' LEARN FINAL '''
                        model, batch_loss, exp_replay = learn(s_alg, model, state_t, state_t1, agent_t, agent_t1, reward, action_t, True, net_conf, exp_replay)

                if s_server == False:
                    # region DRAW SCREEN
                    screen.fill(white)
                    show_Alg(s_alg, screen)
                    show_Samples(sample, screen)
                    show_Level(episodes, screen)
                    show_Score(score, screen)
                    show_Steps(steps, screen)
                    show_Percent(percent_list[-10:], screen)
                    show_Steps_list(steps_list[-30:], screen)
                    show_Act_List(act_list[-20:], screen)
                    show_Action(action_chosen, screen)
                    show_Env(s_env, screen)

                    draw_objects(agent, positivo_list, negativo_list, wall_list, screen)
                    pygame.display.flip()
                    # endregion

            try:
                percent = pos_collected / encountered
            except ZeroDivisionError:
                percent = 0
            percent_list.append(percent)
            loss_list.append(batch_loss)
            print("Episode: ", episodes)

        # region TIME 1
        print("Start time: ", start_time)
        end = time.time()
        end_time = time.strftime('%X %x')
        print("End time: ", end_time)
        time_elapsed = end - start
        print("Time elapsed: ", time_elapsed)
        # endregion

        '''GET THE BEST MODEL'''
        if max(score_list) > max(score_list_best):
            best_model = model
            score_list_best = score_list

        # region MAKE LIST OF THE RESULTS
        avg_last_score_list.append(score_list[-1])

        score_list_df = pd.DataFrame({'Score': score_list})
        percent_list_df = pd.DataFrame({'Percent': percent_list})
        loss_list_df = pd.DataFrame({'Batch_loss': loss_list})
        time_sample_df = pd.DataFrame({'Time': [time_elapsed]})

        df_score = pd.concat([df_score, score_list_df], ignore_index=True, axis=1)
        df_percent_list = pd.concat([df_percent_list, percent_list_df], ignore_index=True, axis=1)
        df_loss_list = pd.concat([df_loss_list, loss_list_df], ignore_index=True, axis=1)
        df_time_sample = pd.concat([df_time_sample, time_sample_df], ignore_index=True, axis=1)
        # endregion

    if s_save == True:
        # region PATH TO SAVE
        save_path_core = __location__ + "/Results/"
        if s_learn == True: save_path = save_path_core + "Train/Env_" + str(s_env) + "/Train_Env_" + str(s_env) + "_" + s_alg
        else: save_path = save_path_core + "Test/Env_" + str(s_env) + "/Test_Env_" + str(s_env) + "_" + s_alg
        if s_alg == "DQN": save_path += "_" + str(s_net_comb_param)

        # convert begin_time to string and format it
        time_path = begin_time.replace(" ", "   ")
        time_path = time_path.replace(":", " ")
        time_path = time_path.replace("/", "-")
        # append to the save path
        save_path = save_path + "   " + time_path

        if s_load == True:
            load_path = " loaded_with " + s_load_path.replace("/", "_")
            save_path = save_path + load_path

        # If it doesnt find the path, then create a new path
        if not os.path.exists(os.path.dirname(save_path)):
            try:
                os.makedirs(os.path.dirname(save_path))
            except OSError as exc:  # Guard against race condition
                print("ERROR when saving the File")
        # endregion
        print("save_path: ", save_path)

        # region SAVE ALL
        # IF IT IS NOT DQN NULL NET CONF. VALUES
        if s_alg != "DQN":
            op_conf = [0, 0, 0, 0, 0, 0]
            net_conf = {"N_actions":0, "Max_memory":0, "Hidden_size":0, "Batch_size":0, "Optimizer":"none"}

        avg_last_score = np.average(avg_last_score_list)
        config_list = pd.concat([pd.Series({'Run_Conf': "A"}),
                                 pd.Series({'Env_conf': s_env}),
                                 pd.Series({'Algort': s_alg}),
                                 pd.Series({'Learn': s_learn}),
                                 pd.Series({'Load': s_load}),
                                 pd.Series({'Samples': s_sample}),
                                 pd.Series({'Episode': s_episode}),
                                 pd.Series({'Max_steps': max_steps}),
                                 pd.Series({'s_cond_to_end': s_cond_to_end}),
                                 pd.Series({'Auto': s_auto}),
                                 pd.Series({'Server': s_server}),
                                 pd.Series({'Print': s_print}),
                                 pd.Series({'MODEL CONF': ""}),
                                 pd.Series({'alfa': alfa}),
                                 pd.Series({'gamma': gamma}),
                                 pd.Series({'Prob': Prob}),
                                 pd.Series({'N_actions': net_conf["N_actions"]}),
                                 pd.Series({'Max_memory': net_conf["Max_memory"]}),
                                 pd.Series({'Hidden_size': net_conf["Hidden_size"]}),
                                 pd.Series({'Batch_size': net_conf["Batch_size"]}),
                                 pd.Series({'Optimizer': net_conf["Optimizer"]}),
                                 pd.Series({'lr': op_conf[0]}),
                                 pd.Series({'beta_1': op_conf[1]}),
                                 pd.Series({'beta_2': op_conf[2]}),
                                 pd.Series({'epsilon': op_conf[3]}),
                                 pd.Series({'decay': op_conf[4]}),
                                 pd.Series({'rho': op_conf[5]}),
                                 pd.Series({'': ""}),
                                 pd.Series({'AVG SCORE': avg_last_score})])
        config_list = config_list.to_frame()

        if s_print: print("\nconfig_list:\n", config_list)

        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(save_path + ".xlsx", engine='xlsxwriter')

        # SAVING CONFIG:
        config_list.to_excel(writer, sheet_name='Run_Conf', header=False)
        worksheet = writer.sheets['Run_Conf']
        worksheet.set_column('A:B', 15)
        # SAVING SCORE:
        df_score_mean = df_score.mean(axis=1)
        df_score.insert(0, "Avg " + str(s_sample), df_score_mean)
        df_score.to_excel(writer, sheet_name='Score')
        worksheet = writer.sheets['Score']
        worksheet.write(0, 0, "Score")
        # SAVING PERCENT:
        df_percent_list_mean = df_percent_list.mean(axis=1)
        df_percent_list.insert(0, "Avg " + str(s_sample), df_percent_list_mean)
        df_percent_list.to_excel(writer, sheet_name='Percent')
        worksheet = writer.sheets['Percent']
        worksheet.write(0, 0, "Percent")
        # SAVING LOSS:
        df_loss_list.to_excel(writer, sheet_name='Loss')
        worksheet = writer.sheets['Loss']
        worksheet.write(0, 0, "Loss")
        # SAVING TIME:
        df_time_sample.to_excel(writer, sheet_name='Time')
        worksheet = writer.sheets['Time']
        worksheet.write(0, 0, "Time")
        # region CELL SIZE

        # worksheet = writer.sheets['Score']
        # worksheet.set_column('A:B', 15)
        # worksheet = writer.sheets['Time']
        # worksheet.set_column('A:B', 15)
        # endregion

        # SAVING BEST MODEL (out of # Samples):
        if s_alg == "DSRL" or s_alg == "QL" or s_alg == "DSRL_dist" or s_alg == "DSRL_dist_type" or s_alg == "DSRL_dist_type_near" or s_alg == "DSRL_dist_type_near_propNeg" or s_alg == "DSRL_object_near" or s_alg == "DSRL_object":
            # SAVING MODEL CONFIGURATIONS:
            best_model.to_excel(writer, sheet_name='model')
            # CONDITIONAL COLOR
            worksheet = writer.sheets['model']
            for x in range(2, 700, 4):
                cell = "C" + str(x) + ":D" + str(x + 3)
                worksheet.conditional_format(cell, {'type': '3_color_scale'})
            # CELL SIZE
            worksheet = writer.sheets['model']
            worksheet.set_column('A:A', 50)

            # region ADD PLOTS
        # worksheet = writer.sheets['results']
        # workbook = writer.book
        # chart = workbook.add_chart({'type': 'line'})
        # chart2 = workbook.add_chart({'type': 'line'})
        # chart.add_series({'values': '=results!$B$2:$B$100'})
        # chart2.add_series({'values': '=results!$C$2:$C$10'})
        # worksheet.insert_chart('F3', chart)
        # worksheet.insert_chart('N3', chart2)

        # SAVE DQN MODEL
        if s_learn == True and s_alg == "DQN":
            save_model(best_model, save_path)

        writer.save()
        # endregion

    print("\n - END - "
          "\n sample: %s"
          "\n s_env: %s"
          "\n s_alg: %s"
          "\n s_episode: %s"
          "\n s_learn: %s"
          "\n s_load: %s"
          "\n s_print: %s"
          "\n s_auto: %s"
          "\n s_cond_to_end: %s"
          "\n s_server: %s"
          "\n s_net_comb_param: %s"
          "\n s_prob: %s" % experiment_configurations)

    # region TIME 2
    print("\n\nBegin time: ", begin_time)
    finish = time.time()
    finish_time = time.strftime('%X %x')
    print("Final time: ", finish_time)
    total_time = finish - begin
    print("Total time: ", total_time)
    # endregion

    return


# -------------------------------------------------------------------------------------------------- #
''' SELECT PARAMETERS TO RUN THE SOFTWARE '''
Env = 11
Alg_list = ["QL",
            "DSRL",
            "DSRL_object_near",
            "DQN",
            "DSRL_dist",
            "DSRL_dist_type",
            "DSRL_dist_type_near",
            "DSRL_dist_type_near_propNeg",
            "DSRL_object"]
Alg = Alg_list[8] # Select the algorithm to be used
Learn = True # To update its knowledge
Load = False # To load a learned model
Load_path = "/Results/Train/Env_1/Train_Env_1_DQN_4   00 33 03   01-05-18"

Samples = 2 # Usually 10 samples
Print = False # Print some info in the terminal
Auto = True # Agent moves Automatic or if False it moves by pressing the Spacebar key
Server = False # If running in the server since
Prob = 0 # Probability to make a random move (exploration rate)
Cond_to_end = "only_positive" # Choose from below (there are 4)
Save = False # Save the model
speed = 0 # seconds per frame

# Cond_to_end = "max_steps"
# Cond_to_end = "coll_all"
# Cond_to_end = "only_negative"
Episodes = 1000 # Usually 1000 or 100

# region DQN Model Configurations:
# max_memory_list =  [5, 5,  5,   30,  30, 30,  100, 100, 100]
# hidden_size_list = [5, 30, 270, 5,   30, 270, 5,   30,  270]
# batch_size_list =  [1, 1,  1,   10,  10, 10,  32,  32,  32]
max_memory_list =  [100,    100,    100,    300, 300,   300,    900, 900, 900]
hidden_size_list = [5,      10,     15,     5,   10,    15,     5,   10,  15]
batch_size_list =  [32,     32,     32,     32,  32,    32,     32,  32,  32]
optimizer_list = ["adam", "rms_opt"]
n_actions = 4  # [move_up, move_down, move_left, move_right]
# endregion
Net_comb_param = 4


# ------------------------------------------------------------------------------------------- #
run(Env, Alg, Learn, Load, Print, Auto, Episodes, Cond_to_end, Server, Net_comb_param, Load_path, Prob, Samples, Save)
# ------------------------------------------------------------------------------------------- #

'''                 REPEAT DQN Net_Comb_Param                  '''
# for i in range(9):
#     Net_comb_param = i
#     run(Env, Alg, Learn, Load, Print, Auto, Episodes, Cond_to_end, Server, Net_comb_param, Load_path, Prob, Samples, Save)


'''                 REPEAT Alg for a list of Env                  '''
# env_list = [2,3]
# for Env in env_list:
#     run(Env, Alg, Learn, Load, Print, Auto, Episodes, Cond_to_end, Server, Net_comb_param, Load_path, Prob, Samples, Save)


'''                 Alg_list for Env_list                  '''
# env_list = [2,3]
# alg_list = ["QL", "DSRL", "DSRL_object_near", "DQN"]
# for Env in env_list:
#     for Alg in alg_list:
#         run(Env, Alg, Learn, Load, Print, Auto, Episodes, Cond_to_end, Server, Net_comb_param, Load_path, Prob, Samples, Save)


