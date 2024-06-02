# Splendor/RL/model.py

import numpy as np
from collections import deque
from keras.models import load_model

class RLAgent:
    def __init__(self):
        self.state_size = 158 # Size of state vector
        self.action_size = 71 # Maximum number of actions
        self.batch_size = 32
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.optimizers import Adam

        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def get_predictions(self, state, legal_moves):
        if np.random.rand() <= self.epsilon:
            return [(move, np.random.rand()) for move in legal_moves]  # Exploration

        act_values = self.model.predict(state)[0]  # All actions
        return [(move, act_values[move]) for move in legal_moves]  # Legal moves


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        minibatch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        for i in minibatch:
            state, action, reward, next_state, done = self.memory[i]
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, model_path):
        self.model.save(model_path)

    def load_model(self, model_path):
        self.model = load_model(model_path)