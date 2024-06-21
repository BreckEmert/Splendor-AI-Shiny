# Splendor/RL/model.py

import numpy as np
import os
import tensorflow as tf
from keras.models import load_model


class RLAgent:
    def __init__(self, layer_sizes, model_path=None):
        self.state_size = 247 # Size of state vector
        self.action_size = 61 # Maximum number of actions 

        self.state_memory = np.empty((0, self.state_size), dtype=float)
        self.action_memory = np.empty((0, ), dtype=int)
        self.num_predicts = 0

        self.gamma = 1  # discount rate
        self.epsilon = 0.3  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9
        self.learning_rate = 0.03

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
        model.add(Dense(self.layer_sizes[0], input_dim=self.state_size, activation='selu'))
        for size in self.layer_sizes[1:]:
            model.add(Dense(size, activation='selu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def get_predictions(self, state, legal_mask):
        self.num_predicts += 1 # More accurate than game.half_turns

        state = np.expand_dims(state, axis=0) # Batch dimension
        if np.random.rand() <= self.epsilon:
            act_values = np.random.rand(self.action_size)  # Exploration
        else:
            act_values = self.model.predict(state, verbose=0)[0]  # All actions

        # Illegal move filter
        act_values = np.where(legal_mask, act_values, -np.inf)
        return act_values

    def remember(self, state, action):
        self.state_memory = np.vstack([self.state_memory, state])
        self.action_memory = np.append(self.action_memory, action)

    def train_batch(self, states, actions, padding_mask):
        with tf.GradientTape() as tape:
            predictions = self.model(states, training=True)
            
            # Apply masks to get valid predictions and actions
            masked_predictions = tf.boolean_mask(predictions, padding_mask)
            masked_actions = tf.boolean_mask(actions, padding_mask)

            # Calculate the categorical cross-entropy loss for the correct actions
            loss = tf.keras.losses.categorical_crossentropy(masked_actions, masked_predictions)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def save_model(self, model_path):
        if not model_path:
            model_path = "C:/Users/Public/Documents/Python_Files/Splendor/RL/trained_agents/Player1_1024_512"
        os.makedirs(model_path, exist_ok=True)
        self.model.save(model_path)
        print(f"Saved the model at {model_path}")

    def load_model(self, model_path):
        return load_model(model_path)
    
if __name__ == "__main__":
    import tensorflow as tf
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
