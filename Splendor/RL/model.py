# Splendor/RL/model.py

import numpy as np
import os
from collections import deque
from keras.models import load_model


class RLAgent:
    def __init__(self, layer_sizes, model_path=None):
        self.state_size = 247 # Size of state vector
        self.action_size = 61 # Maximum number of actions 
        self.memory = deque(maxlen=500)
        self.gamma = 1  # discount rate
        self.epsilon = 0.3  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9
        self.learning_rate = 0.0002

        self.layer_sizes = layer_sizes
        if model_path:
            self.model = load_model(model_path)
        else:
            print("Building a new model")
            self.model = self._build_model()

    def _build_model(self):
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.optimizers import Adam

        model = Sequential()
        model.add(Dense(self.layer_sizes[0], input_dim=self.state_size, activation='relu'))
        for size in self.layer_sizes[1:]:
            model.add(Dense(size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def get_predictions(self, state, legal_mask):
        state = np.reshape(state, [1, self.state_size])
        if np.random.rand() <= self.epsilon:
            act_values = np.random.rand(self.action_size)  # Exploration
        else:
            act_values = self.model.predict(state, verbose=0)[0]  # All actions

        # Illegal move filter
        act_values = np.where(legal_mask, act_values, -np.inf)
        return act_values

    def remember(self, state, action, reward, next_state, done):
        state = np.reshape(state, [1, self.state_size])
        next_state = np.reshape(next_state, [1, self.state_size])
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        minibatch_size = -(-len(self.memory)//2)
        minibatch = np.random.choice(len(self.memory), minibatch_size, replace=False)
        states = np.zeros((minibatch_size, self.state_size))
        targets = np.zeros((minibatch_size, self.action_size))

        for i, idx in enumerate(minibatch):
            state, action, reward, next_state, done = self.memory[idx]
            target = self.model.predict(state)[0]
            if done:
                target[action] = reward
            else:
                t = self.model.predict(next_state)[0]
                target[action] = reward + self.gamma * np.amax(t)
            states[i] = state
            targets[i] = target

        self.model.fit(states, targets, epochs=1, verbose=0)

    def train_individual(self, state, action, reward, next_state, done):
        target_f = self.model.predict(state, verbose=0)
        target_f[0][action] = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f[0][action] = target
        # print(f'training fit, reward = {reward}')
        self.model.fit(state, target_f, epochs=1, verbose=0)

    def train_batch(self, batch, reward):
        np.random.shuffle(batch)

        states = np.zeros((len(batch), self.state_size))
        targets = np.zeros((len(batch), self.action_size))

        for i, (state, action, _, next_state, done) in enumerate(batch):
            target = self.model.predict(state)[0]
            if done:
                target[action] = reward
            else:
                t = self.model.predict(next_state)[0]
                target[action] = reward + self.gamma * np.amax(t)
            states[i] = state
            targets[i] = target

        self.model.fit(states, targets, epochs=1, verbose=0)


    def save_model(self, model_dir, player_name):
        layer_sizes_str = '_'.join(map(str, self.layer_sizes))
        model_path = os.path.join(model_dir, f"{player_name}_{layer_sizes_str}")
        os.makedirs(model_path, exist_ok=True)
        self.model.save(os.path.join(model_path, 'model.keras'))

    def load_model(self, model_path):
        return load_model(model_path)