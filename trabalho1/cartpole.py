# UNIVERSIDADE FEDERAL DA FRONTEIRA SUL - UFFS, Campus Chapecó.
# Ciência da Computação, 7a Fase, Matutino.
# Trabalho 01 - CartPole usando Q-Learning.
# Estudantes: João Marcos Campagnolo e Renan Casaca.

import numpy as np
import cPickle as pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import gym
import time

env = gym.make('CartPole-v0')
env.reset()
random_episodes = 0
reward_sum = 0

# Aqui ponha o seu codigo para ler os valores dos estados, se houver

while random_episodes < 10:
    observation = env.reset()
    for t in range(200):
        env.render()
        env.step(env.action_space.sample())
        print(observation)
	time.sleep(0.1)
# Aqui ponho o seu codigo para salvar a observacao, estado



# Aqui ponha o seu codigo para escolher a melhor acao dependendo do estado atual

        action = 1 ##env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:

# Aqui ponha o seu codigo para atualizar os valores dos estados apos uma execucao

            print("Episode finished after {} timesteps".format(t+1))
            break


#   Aqui ponha o seu codigo para salvar os valores dos estados calculados
