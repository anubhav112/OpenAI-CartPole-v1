import gym
import tflearn
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import numpy as np
from collections import Counter
import random
import tensorflow as tf
import pickle

num_episodes = 100000

env = gym.make('CartPole-v1')
env.reset()
game_score = 500
def generate_training_data():
	train_data = []
	scores = []
	for _ in range(num_episodes):
		env.reset()
		score = 0
		prev_info = []
		game_info = []
		
		for _ in range(game_score):
			action = random.randrange(0,2)
			observation,reward,done,info = env.step(action)
			score += reward
			if len(prev_info) > 0:
				game_info.append([prev_info,action])
			prev_info = observation
			if done:
				# print("Game Over")
				break

		if score > 80:
			scores.append(score)
			for i in game_info:
				if i[1] == 1:
					output = [0,1]
				elif i[1] == 0:
					output = [1,0]
				train_data.append([i[0],output])
	# print(Counter(scores))
	return train_data

# train_data = generate_training_data()
# print(len(train_data))
# with open('D:\\anubhav\\Codes\\CartPole.pickle','wb') as f:
# 	pickle.dump(train_data,f)
pickle_in = open('D:\\anubhav\\Codes\\CartPole.pickle','rb')
train_data = pickle.load(pickle_in)

n_nodes_h1 = 128
n_nodes_h2 = 64
n_nodes_h3 = 32

n_classes = 2

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

hidden_1 = {'weights':tf.Variable(tf.truncated_normal([4,n_nodes_h1],stddev = 0.1)),
          'biases':tf.Variable(tf.constant(0.1,shape = [n_nodes_h1]))}
hidden_2 = {'weights':tf.Variable(tf.truncated_normal([n_nodes_h1,n_nodes_h2],stddev = 0.1)),
          'biases':tf.Variable(tf.constant(0.1,shape = [n_nodes_h2]))}
hidden_3 = {'weights':tf.Variable(tf.truncated_normal([n_nodes_h2,n_nodes_h3],stddev = 0.1)),
          'biases':tf.Variable(tf.constant(0.1,shape = [n_nodes_h3]))}
output = {'weights':tf.Variable(tf.truncated_normal([n_nodes_h3,n_classes],stddev = 0.1)),
          'biases':tf.Variable(tf.constant(0.1,shape = [n_classes]))}

def nn_model(data):
    layer1 = tf.matmul(data,hidden_1['weights'])+hidden_1['biases']
    layer1 = tf.nn.relu(layer1)
    layer2 = tf.matmul(layer1,hidden_2['weights'])+hidden_2['biases']
    layer2 = tf.nn.relu(layer2)
    layer3 = tf.matmul(layer2,hidden_3['weights'])+hidden_3['biases']
    layer3 = tf.nn.relu(layer3)
    classification = tf.matmul(layer3,output['weights'])+output['biases']

    return classification

y_predict = nn_model(x)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_predict,labels = y))

model = tf.train.AdamOptimizer().minimize(cost)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

num_iter = 10

x_ = np.array([i[0] for i in train_data])
y_ = [i[1] for i in train_data]

for i in range(num_iter):
	iter_loss = 0
	batch_x = x_
	batch_y = y_
	_,c = sess.run([model,cost],feed_dict = {x:batch_x,y:batch_y})
	iter_loss += c
	print(i,iter_loss)


answers = tf.equal(tf.argmax(y_predict,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(answers,tf.float32))

print(sess.run(accuracy,feed_dict = {x:x_,y:y_}))

def run_example():
	sum1 = 0
	for i in range(25):
		env.reset()
		score = 0
		prev_info = []
		train_x = []
		train_y = []
		for _ in range(game_score):
			env.render()
			if len(prev_info) > 0:
				action = np.argmax(sess.run(y_predict,feed_dict = {x:prev_info.reshape(-1,len(prev_info))})[0])
				if action == 1:
					output = [0,1]
				elif action == 0:
					output = [1,0]
				train_x.append(prev_info)
				train_y.append(output)
				# print(action)
			else :
				action = random.randrange(0,2)
			observation,reward,done,info = env.step(action)
			score += reward
			prev_info = observation
			if done:
				train_x = np.array(train_x).reshape(-1,len(train_x[0]))
				if score > 200:
					_,c=sess.run([model,cost],feed_dict = {x:train_x,y:train_y})
				sum1 += score
				print("Game Over! Your score is",score,": Avg till now =",sum1/(i+1))
				break
run_example()

sess.close()











# def nn_model(shape):
# 	network = input_data(shape = shape,name = 'input')
	
# 	network = fully_connected(network,128,activation = 'relu')
# 	network = dropout(network,0.8)

# 	# network = fully_connected(network,512,activation = 'relu')
# 	# network = dropout(network,0.8)

# 	# network = fully_connected(network,256,activation = 'relu')
# 	# network = dropout(network,0.8)

# 	network = fully_connected(network,2,activation = 'softmax')
# 	network = dropout(network,0.8)

# 	network = regression(network,optimizer = 'adam',learning_rate = 1e-2,loss = 'categorical_crossentropy',name = 'label')
# 	model = tflearn.DNN(network)

# 	return model

# def train_model(train_data):
# 	x = np.array([i[0] for i in train_data])
# 	y = [i[1] for i in train_data]
# 	# print(x[:5])
# 	# x = x.reshape(-1,len(train_data[0][0]),1)
# 	# print(y[:10])
# 	model = nn_model([None,len(x[0])])
# 	# print(len(x[0]))
# 	model.fit({'input':x},{'label':y},n_epoch = 5,snapshot_step = 500,show_metric = True)

# 	return model

# train_data = generate_training_data()
# model = train_model(train_data)

# # model.save('CartPole.model')
# def run_example(model):
# 	for _ in range(10):
# 		env.reset()
# 		score = 0
# 		prev_info = []
# 		for _ in range(game_score):
# 			env.render()
# 			if len(prev_info) > 0:
# 				action = np.argmax(model.predict(prev_info.reshape(-1,len(prev_info)))[0])
# 				print(model.predict(prev_info.reshape(-1,len(prev_info))))
# 			else :
# 				action = random.randrange(0,2)
# 			observation,reward,done,info = env.step(action)
# 			score += reward
# 			prev_info = observation
# 			if done:
# 				print("Game Over! Your score is",score)
# 				break
# run_example(model)