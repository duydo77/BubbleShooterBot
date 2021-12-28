# Implementation of deep reinforcement learning 
# applied to the game Bubble Shooter

import math, pygame, sys, os, copy, time, random
import numpy as np
import pygame.gfxdraw
from pygame.locals import *
import matplotlib.pyplot as plt
from game import Arrow
import game
from game import *

import tensorflow as tf

# import lasagne
# import theano
# import theano.tensor as T
# from lasagne.layers import cuda_convnet

# Settings of the game
FPS          = 120
WINDOWWIDTH  = 640
WINDOWHEIGHT = 480
TEXTHEIGHT   = 20
BUBBLERADIUS = 20
BUBBLEWIDTH  = BUBBLERADIUS * 2
BUBBLELAYERS = 6
BUBBLEYADJUST = 5
STARTX = WINDOWWIDTH / 2
STARTY = WINDOWHEIGHT - 27
ARRAYWIDTH = 16
ARRAYHEIGHT = 14
DIE = 9
ARRAYWIDTH = 16


# Colours
#            R    G    B
GRAY     = (100, 100, 100)
NAVYBLUE = ( 60,  60, 100)
WHITE    = (255, 255, 255)
RED      = (255,   0,   0)
GREEN    = (  0, 255,   0)
BLUE     = (  0,   0, 255)
YELLOW   = (255, 255,   0)
ORANGE   = (255, 128,   0)
PURPLE   = (255,   0, 255)
CYAN     = (  0, 255, 255)
BLACK    = (  0,   0,   0)
COMBLUE  = (233, 232, 255)

RIGHT = 'right'
LEFT  = 'left'
BLANK = '.'

# background colour
BGCOLOR    = WHITE

# Colours in the game
COLORLIST = [RED, GREEN, YELLOW, ORANGE, CYAN]

# Put display on true for visualisation
DISPLAY = True

# Replay Memory
REPLAYMEMORY = []

# Hyper parameters
BATCHSIZE = 128
AMOUNTOFSHOTS = 500
NUMBEROFACTIONS = 20
GRIDSIZE = 11
discount = 0.99
learning_rate = 0.00008
size_RM = 7500
ITERATIONS = 5#400000


def main():
	global FPSCLOCK, DISPLAYSURF, DISPLAYRECT, MAINFONT
	pygame.init()
	if DISPLAY == True:
		pygame.display.set_caption('Puzzle Bobble')
		DISPLAYSURF, DISPLAYRECT = makeDisplay()
	createMemory()
	network = train()
	# test(network)


# Training of the self-learning agent
def train():

	# initialize game
	direction, launchBubble, newBubble, arrow, bubbleArray, nextBubble, score, alive, shots, getout, loss_game = restartGame()

	# hyperparameters
	epsilon = 0.9

	# counters
	moves = 0
	wins = 0
	gameover = 0
	games = 0
	average_loss = 0
	average_reward = 0

	# with or without display
	display = DISPLAY
	delay = 0

	# building network
	network = build_network()
	target_network = build_network()
	loss_fn = tf.keras.losses.Huber()
	optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

	# get parameters from trained network
	"""
	with np.load('5_colours_20shots.npz') as f:
		param_values = [f['arr_%d' % i] for i in range(len(f.files))]
	lasagne.layers.set_all_param_values(network, param_values)"""

	target_network.set_weights(network.get_weights())

	# get state
	state = gameState(bubbleArray, newBubble.color)
	while moves < ITERATIONS:
		
		if display == True:
			DISPLAYSURF.fill(BGCOLOR)
		
		# act random or greedy
		chance = random.uniform(0, 1)
		chance = 1.0
		launchBubble = True
		if chance < epsilon:
			action = random.randint(0, NUMBEROFACTIONS - 1)
		else:
			predict_state = np.reshape(state, (1, 8, GRIDSIZE * 2, ARRAYWIDTH * 2))
			action = np.argmax(network(predict_state).numpy())
		direction = (action * 8) + 10
		newBubble.angle = direction

		# process game
		bubbleArray, alive, deleteList, nextBubble = processGame(launchBubble, newBubble, bubbleArray, score, arrow, direction, alive, display, delay)
		
		# get reward for the action
		getout, wins, reward, gameover = getReward(alive, getout, wins, deleteList, gameover)

		# getting new bubble for shooting
		newBubble = Bubble(nextBubble.color)
		newBubble.angle = arrow.angle

		# get the newstate
		newState = gameState(bubbleArray, newBubble.color)

		# storage of replay memory
		if getout == True:
			REPLAYMEMORY.append((state, action, reward, newState, 0))
		else:
			REPLAYMEMORY.append((state, action, reward, newState, discount))

		# delete one tuple is replay memory becomes too big
		if len(REPLAYMEMORY) > size_RM:
			REPLAYMEMORY.pop(0)
		
		# training the network
		states, actions, rewards, newstates, discounts = get_batch()
		
		future_reward = target_network(newstates)
		y = reward + discount * tf.reduce_max(future_reward, axis=1)
		masks = tf.one_hot(np.squeeze(actions), NUMBEROFACTIONS)
		
		with tf.GradientTape() as tape:
			q_values = network(states)
			tmp = tf.multiply(q_values, masks)
			q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
			loss = loss_fn(y, q_action)

		# ==========| Backpropagation
		grads = tape.gradient(loss, network.trainable_variables)
		optimizer.apply_gradients(zip(grads, network.trainable_variables))

		average_loss = average_loss + loss
		average_reward = average_reward + reward

		if moves % 1000 == 0 and moves > 0:
			print("Amount of actions taken: ", moves)
			print("Average loss: ", average_loss/1000.0)
			print("Average Reward: ", average_reward/1000.0)
			print("Amount of wins: ", wins)
			average_reward = 0 
			average_loss = 0
			if epsilon > 0.1:
				epsilon = epsilon - 0.01

		# updating the target network
		if moves % 2500 == 0:
			target_network.set_weights(network.get_weights())

		# change the state to newState
		state = newState

		moves = moves + 1
		shots = shots + 1

		if getout == True or shots == AMOUNTOFSHOTS:
			games = games + 1
			direction, launchBubble, newBubble, arrow, bubbleArray, nextBubble, score, alive, shots, getout, loss_game = restartGame()
		state = gameState(bubbleArray, newBubble.color)

	# saving parameters of the network
	network.save_weights("models/network.h5")
	target_network.save_weights("models/target_network.h5")
	return network


# test the performance of the algorithm
def test(network):

	# initalize game
	direction, launchBubble, newBubble, arrow, bubbleArray, nextBubble, score, alive, shots, getout, loss_game = restartGame()

	# Display of the game 
	display = False
	delay = 0


	# counters for evaluation
	moves = 0
	wins = 0
	gameover = 0
	games = 0
	average_reward = 0
	gameover = 0
	average_score = 0
	average_shot = 0
	average_shot_win = 0

	# loading parameters of a trained model
	"""
	with np.load('5_colours_20shots_improve.npz') as f:
		param_values = [f['arr_%d' % i] for i in range(len(f.files))]
	lasagne.layers.set_all_param_values(network, param_values)"""

	# get current state
	state = gameState(bubbleArray, newBubble.color)

	while games < 1000:
		if display == True:
			DISPLAYSURF.fill(BGCOLOR)

		# performing the best action
		predict_state = np.reshape(state, (1, 8, GRIDSIZE * 2, ARRAYWIDTH * 2))
		action = np.argmax(network(predict_state).numpy())
		direction = (action * 8) + 10
		launchBubble = True
		newBubble.angle = direction
		
		# process game
		bubbleArray, alive, deleteList, nextBubble = processGame(launchBubble, newBubble, bubbleArray, score, arrow, direction, alive, display, delay)
		shots = shots + 1
		
		# adding extra ball layers
		"""
		if shots % 32 == 0 and shots > 0:
			bubbleArray = addLayer(bubbleArray)
		"""

		# average amount of balls popped per shot
		if len(deleteList) > 2:
			average_reward = len(deleteList) + average_reward
		getout, wins, reward, gameover = getReward(alive, getout, wins, deleteList, gameover)

		# new bubble for shooting
		newBubble = Bubble(nextBubble.color)
		newBubble.angle = arrow.angle

		moves = moves + 1
		
		if getout == True or shots == AMOUNTOFSHOTS:
			if alive == "win":
				average_shot_win = average_shot_win + shots
			average_score = average_score + score.total
			games = games + 1
			if games % 50 == 0:
				print(games, "/1000")
			average_shot = average_shot + shots
			direction, launchBubble, newBubble, arrow, bubbleArray, nextBubble, score, alive, shots, getout, loss_game = restartGame()


		# update state
		state = gameState(bubbleArray, newBubble.color)
	print("Amount of games played: ", games)
	print("Amount of wins: ", wins)
	print("Amount of loses: ", gameover)
	print("Average reward: ", average_reward/float(moves))
	print("Average score: ", average_score/float(games))
	print("Average amount of shots per game: ", average_shot/float(games))
	if wins > 0:
		print("Average amount of shots per winning game: ", average_shot_win/float(wins))


# creating Replay memory
def createMemory():

	# start game
	direction, launchBubble, newBubble, arrow, bubbleArray, nextBubble, score, alive, shots, getout, loss_game = restartGame()

	# Display of the game
	display = False
	delay = 0

	# counters
	gameover = 0
	games = 0
	wins = 0
	moves = 0
	terminal = 0
	layer = 0

	# get the current game state
	state = gameState(bubbleArray, newBubble.color)

	while moves < 5000:
		if display == True:
			DISPLAYSURF.fill(BGCOLOR)

		# performing random action
		action = random.randint(0, NUMBEROFACTIONS - 1)
		direction = (action * 8) + 10
		launchBubble = True
		newBubble.angle = direction

		# process game
		bubbleArray, alive, deleteList, nextBubble = processGame(launchBubble, newBubble, bubbleArray, score, arrow, direction, alive, display, delay)
		shots = shots + 1

		# getting reward for action
		getout, wins, reward, gameover = getReward(alive, getout, wins, deleteList, gameover)

		# new bubble for shooting
		newBubble = Bubble(nextBubble.color)
		newBubble.angle = arrow.angle

		# getting new state
		newState = gameState(bubbleArray, newBubble.color)

		moves = moves + 1

		# storage of the replay memory
		if getout == True:
			REPLAYMEMORY.append((state, action, reward, newState, 0))
		else:
			REPLAYMEMORY.append((state, action, reward, newState, discount))

		# restart game when game is won or lost
		if getout == True or shots == AMOUNTOFSHOTS:
			games = games + 1
			direction, launchBubble, newBubble, arrow, bubbleArray, nextBubble, score, alive, shots, getout, loss_game = restartGame()
		state = gameState(bubbleArray, newBubble.color)
	print("Size of the Replay Memory: ", len(REPLAYMEMORY))


# for getting a batch for training
def get_batch():
	counter = 0
	actions  = np.zeros((BATCHSIZE, 1))
	discounts = np.zeros((BATCHSIZE, 1))
	rewards = np.zeros((BATCHSIZE, 1))
	states = np.zeros((BATCHSIZE, 8, GRIDSIZE * 2, ARRAYWIDTH *2))
	newstates = np.zeros((BATCHSIZE, 8, GRIDSIZE * 2, ARRAYWIDTH *2))
	while counter < BATCHSIZE:
		random_action = random.randint(0, len(REPLAYMEMORY) - 1)
		(state, action, reward, newstate, discount) = REPLAYMEMORY[random_action]
		actions[counter][0] = action
		rewards[counter][0] = reward
		discounts[counter][0] = discount
		states[counter] = state
		newstates[counter] = newstate
		counter = counter + 1
	return states, actions, rewards, newstates, discounts 


# adding extra lines of balls to the game
def addLayer(bubbleArray):
	gameColorList = updateColorList(bubbleArray)
	if gameColorList[0] == WHITE:
		return bubbleArray
	bubbleArray = np.delete(bubbleArray, -1, 0)
	bubbleArray = np.delete(bubbleArray, -1, 0)
	newRow = []
	for index in range(len(bubbleArray[0])):
		newRow.append(BLANK)
	newRow = np.asarray(newRow)
	newRow = np.reshape(newRow, (1,16))
	bubbleArray = np.concatenate((newRow, bubbleArray))
	bubbleArray = np.concatenate((newRow, bubbleArray))
	for n in range(2):
		for index in range(len(bubbleArray[0])):
			random.shuffle(gameColorList)
			newBubble = Bubble(gameColorList[0], n, index)
			bubbleArray[n][index] = newBubble
	bubbleArray[1][15] = BLANK
	return bubbleArray


# reward function
def getReward(alive, getout, wins, deleteList, gameover):
	if alive == "win":
		wins = wins + 1
		reward = 20.
		getout = True
	elif alive == "lose":
		reward = -15.
		getout = True
		gameover = gameover + 1
	elif len(deleteList) > 2:
		getout = False
		reward = 1. * len(deleteList)
	elif len(deleteList) == 2:
		getout = False
		reward = 1.
	else: 
		reward = -1.
		getout = False
	return getout, wins, reward, gameover


# get target state, action, reward, nextate, discount
def getTarget(network, target_network, batch):
	no_state = np.zeros((8, GRIDSIZE * 2, ARRAYWIDTH * 2))
	starting_states = np.array([observation[1][0] for observation in batch])
	ending_states = np.array([no_state if observation[1][3] is None else observation[1][3] for observation in batch])
	
	current_predictions = network(starting_states)
	predictions_ending_online = network(starting_states)
	predictions_ending_target = target_network(ending_states)

	targets = []
	states = []
	errors = []

	# for (i, (memory_index, (state)))


# to restart the game
def restartGame():
	direction = None
	launchBubble = False
	gameColorList = copy.deepcopy(COLORLIST)
	arrow = Arrow()
	bubbleArray = makeBlankBoard()
	setBubbles(bubbleArray, gameColorList)
	nextBubble = Bubble(gameColorList[0])
	nextBubble.rect.right = WINDOWWIDTH - 5
	nextBubble.rect.bottom = WINDOWHEIGHT - 5
	score = Score()
	alive = "alive"
	newBubble = Bubble(nextBubble.color)
	newBubble.angle = arrow.angle
	shots = 0
	getout = False
	loss_game = 0
	return direction, launchBubble, newBubble, arrow, bubbleArray, nextBubble, score, alive, shots, getout, loss_game


# building the network
def build_network(): 
	'''
	input
	conv2d 32 6 rectify
	conv2d 64 2 rectify
	dense 512   rectify
	dense 512   rectify
	dense out 	rectify

	l_in = lasagne.layers.InputLayer(
		shape=(None, 8, GRIDSIZE * 2, ARRAYWIDTH * 2)
	)
	l_conv1 = cuda_convnet.Conv2DCCLayer(
		l_in,
		num_filters=32,
		filter_size=(6, 6),
		stride=(1, 1),
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.HeUniform(),
		b=lasagne.init.Constant(.1),
	)	
	l_conv2 = cuda_convnet.Conv2DCCLayer(
		l_conv1,
		num_filters=64,
		filter_size=(2, 2),
		stride=(1, 1),
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.HeUniform(),
		b=lasagne.init.Constant(.1),
	)
	l_hidden1 = lasagne.layers.DenseLayer(
		l_conv2,
		num_units=512,
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.HeUniform(),
		b=lasagne.init.Constant(.1)
	)
	l_hidden2 = lasagne.layers.DenseLayer(
		l_hidden1,
		num_units=512,
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.HeUniform(),
		b=lasagne.init.Constant(.1)
	)
	l_out = lasagne.layers.DenseLayer(
		l_hidden2,
		num_units=NUMBEROFACTIONS,
		nonlinearity=None,
		W=lasagne.init.HeUniform(),
		b=lasagne.init.Constant(.1)
	)
	return l_out
	'''

	import tensorflow as tf
	input_layer = tf.keras.layers.Input(shape=(8, GRIDSIZE * 2, ARRAYWIDTH * 2))
	first_layer = input_layer

	conv1 = tf.keras.layers.Conv2D(32, (6, 6), activation='relu', padding='same')(first_layer)
	conv2 = tf.keras.layers.Conv2D(64, (2, 2), activation='relu', padding='same')(conv1)
	flatten = tf.keras.layers.Flatten()(conv2)
	fc1 = tf.keras.layers.Dense(512, activation='relu')(flatten)
	fc2 = tf.keras.layers.Dense(512, activation='relu')(fc1)
	out = tf.keras.layers.Dense(NUMBEROFACTIONS, activation='relu')(fc2)


	# out = m(first_layer)
	
	model = tf.keras.Model(inputs=[input_layer], outputs=[out], name='agent_model')
	
	# model.compile(
	# 	loss='sparse_categorical_crossentropy',
	# 	optimizer=tf.keras.optimizers.RMSprop(learning_rate),
	# 	metrics=['accuracy']
	# )

	return model


# getting the current game state
def gameState(bubbleArray, ballcolor):
	dimension = 0
	state = np.ones((8, GRIDSIZE * 2, ARRAYWIDTH * 2)) * -1
	for colour in COLORLIST:
		counter = 0
		balls = 0
		if ballcolor == colour:
			state[dimension][GRIDSIZE * 2 - 1][ARRAYWIDTH] = 1
			state[dimension][GRIDSIZE * 2 - 1][ARRAYWIDTH + 1] = 1
			state[dimension][GRIDSIZE * 2 - 2][ARRAYWIDTH] = 1
			state[dimension][GRIDSIZE * 2 - 2][ARRAYWIDTH + 1] = 1
			balls = balls + 1
		for row in range(GRIDSIZE):
			for column in range(len(bubbleArray[0])):
				if bubbleArray[row][column] != BLANK and bubbleArray[row][column].color == colour:
					if counter % 2 == 0:
						state[dimension][row * 2][(2 * column)] = 1
						state[dimension][row * 2][(2 * column) + 1] = 1
						state[dimension][row * 2 + 1][(2 * column)] = 1
						state[dimension][row * 2 + 1][(2 * column) + 1] = 1
					elif counter % 2 != 0:
						state[dimension][row * 2][2 * column + 1] = 1
						state[dimension][row * 2][2 * column + 2] = 1
						state[dimension][row * 2 + 1][2 * column + 1] = 1
						state[dimension][row * 2 + 1][2 * column + 2] = 1
					balls = balls + 1
			counter = counter + 1
		for row in range(GRIDSIZE * 2):
			for column in range(len(bubbleArray[0]) * 2):
				if state[dimension][row][column] > 0:
					state[dimension][row][column] = 1/(float(balls) * 4.)
				if state[dimension][row][column] < 0:
					state[dimension][row][column] = -1 * 1/float((GRIDSIZE * 2 * ARRAYWIDTH * 2) - 4. * balls)
		dimension = dimension + 1
	
	balls = 1 # shooting ball
	counter = 0
	state[dimension] = np.ones((GRIDSIZE * 2, ARRAYWIDTH * 2))
	state[dimension][GRIDSIZE * 2 - 1][ARRAYWIDTH] = -1
	state[dimension][GRIDSIZE * 2 - 1][ARRAYWIDTH + 1] = -1
	state[dimension][GRIDSIZE * 2 - 2][ARRAYWIDTH] = -1
	state[dimension][GRIDSIZE * 2 - 2][ARRAYWIDTH + 1] = -1
	for row in range(GRIDSIZE):
		for column in range(len(bubbleArray[0])):
			if bubbleArray[row][column] != BLANK:
				balls = balls + 1
				if counter % 2 == 0:
					state[dimension][row * 2][(2 * column)] = -1
					state[dimension][row * 2][(2 * column) + 1] = -1
					state[dimension][row * 2 + 1][(2 * column)] = -1
					state[dimension][row * 2 + 1][(2 * column) + 1] = -1
				elif counter % 2 != 0:
					state[dimension][row * 2][2 * column + 1] = -1
					state[dimension][row * 2][2 * column + 2] = -1
					state[dimension][row * 2 + 1][2 * column + 1] = -1
					state[dimension][row * 2 + 1][2 * column + 2] = -1
		counter = counter + 1
	for row in range(GRIDSIZE * 2):
		for column in range(len(bubbleArray[0]) * 2):
			if state[dimension][row][column] > 0:
				state[dimension][row][column] = 1/(float(balls) * 4.)
			if state[dimension][row][column] < 0:
				state[dimension][row][column] = -1 * 1/(float((GRIDSIZE * 2 * ARRAYWIDTH * 2) - 4. * balls))
	dimension = dimension + 1
	for n in range(8 - len(COLORLIST) - 1):
		state[dimension] = np.zeros((GRIDSIZE * 2, ARRAYWIDTH * 2))
		dimension = dimension + 1
	return state

				
if __name__ == '__main__':
	main()
	# input = np.random.rand(20,8,GRIDSIZE * 2, ARRAYWIDTH * 2)
	# m = build_network()
	# out = m(input)
	# print(tf.reduce_max(out, axis=-1))

