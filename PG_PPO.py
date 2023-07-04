# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 16:02:04 2023

@author: ACER
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt

# Define the Othello game environment
class OthelloGame:
    def __init__(self, board_size=8):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.board[int(board_size/2)-1:int(board_size/2)+1, int(board_size/2)-1:int(board_size/2)+1] = [[2, 1], [1, 2]]
        self.current_player = 1

    def get_valid_moves(self):
        valid_moves = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.is_valid_move(row, col):
                    valid_moves.append((row, col))
        return valid_moves

    def is_valid_move(self, row, col):
        if self.board[row, col] != 0:
            return False

        for dir in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            x, y = row + dir[0], col + dir[1]
            if self.is_on_board(x, y) and self.board[x, y] == self.get_opponent():
                x += dir[0]
                y += dir[1]
                while self.is_on_board(x, y) and self.board[x, y] == self.get_opponent():
                    x += dir[0]
                    y += dir[1]
                if self.is_on_board(x, y) and self.board[x, y] == self.current_player:
                    return True

        return False

    def is_on_board(self, row, col):
        return 0 <= row < self.board_size and 0 <= col < self.board_size

    def get_opponent(self):
        return 3 - self.current_player

    def make_move(self, row, col):
        if not self.is_valid_move(row, col):
            return self.board, -1, True  # Invalid move, return current board, -1 reward, and game over

        self.board[row, col] = self.current_player
        cells_to_flip = []
        for dir in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            x, y = row + dir[0], col + dir[1]
            while self.is_on_board(x, y) and self.board[x, y] == self.get_opponent():
                x += dir[0]
                y += dir[1]
                if self.is_on_board(x, y) and self.board[x, y] == self.current_player:
                    while True:
                        x -= dir[0]
                        y -= dir[1]
                        if x == row and y == col:
                            break
                        cells_to_flip.append((x, y))

        for cell in cells_to_flip:
            self.board[cell] = self.current_player

        self.current_player = self.get_opponent()

        game_over = self.is_game_over()
        winner = self.get_winner() if game_over else None
        reward = self.calculate_reward() if game_over else 0

        return self.board, reward, game_over

    def calculate_reward(self):
        player_1_score = np.sum(self.board == 1)
        player_2_score = np.sum(self.board == 2)
        if player_1_score > player_2_score:
            return 1
        elif player_1_score < player_2_score:
            return -1
        else:
            return 0

    def is_game_over(self):
        return len(self.get_valid_moves()) == 0

    def get_winner(self):
        player_1_score = np.sum(self.board == 1)
        player_2_score = np.sum(self.board == 2)
        if player_1_score > player_2_score:
            return 1
        elif player_1_score < player_2_score:
            return 2
        else:
            return 0

# Define the policy network class
class PolicyNetwork(tf.keras.Model):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = layers.Dense(256, activation='relu')
        self.fc2 = layers.Dense(128, activation='relu')
        self.fc3 = layers.Dense(output_size, activation='softmax')

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


# Agent class
class Agent:
    def __init__(self, input_size, output_size, learning_rate):
        self.policy_network = PolicyNetwork(input_size, output_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def choose_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        logits = self.policy_network(state)
        action = tf.random.categorical(logits, num_samples=1)
        return action.numpy()[0, 0]

    def train_step(self, states, actions, rewards):
        with tf.GradientTape() as tape:
            logits = self.policy_network(states)
            action_probs = tf.nn.softmax(logits)
            neg_log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions)
            loss = tf.reduce_mean(neg_log_probs * rewards)

        gradients = tape.gradient(loss, self.policy_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy_network.trainable_variables))
   
    
# Define the Proximal Policy Optimization (PPO) agent
class PPOAgent:
    def __init__(self, input_size, output_size, learning_rate, clip_ratio, value_coef, entropy_coef, num_epochs):
        self.input_size = input_size
        self.output_size = output_size
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.num_epochs = num_epochs
        self.model = self.build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def build_model(self):
        inputs = layers.Input(shape=(self.input_size,))
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.Dense(64, activation='relu')(x)
        logits = layers.Dense(self.output_size, activation='softmax')(x)
        value = layers.Dense(1)(x)
        return tf.keras.Model(inputs=inputs, outputs=[logits, value])

    def choose_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        logits, _ = self.model(state)
        action = tf.random.categorical(logits, num_samples=1)
        return action.numpy()[0, 0]

    def update(self, states, actions, rewards, dones):
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
    
        old_logits, old_values = self.model(states)
        old_log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=old_logits, labels=actions)
        old_log_probs = tf.stop_gradient(old_log_probs)
    
        with tf.GradientTape() as tape:
            logits, values = self.model(states)
            log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions)
            ratio = tf.exp(old_log_probs - log_probs)
            surrogate1 = ratio * rewards
            surrogate2 = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * rewards
            actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
    
            value_loss = tf.reduce_mean(tf.square(values - rewards))
    
            entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=logits))
            loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy
    
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))



# Training function
def train(alg):
    game = OthelloGame()
    if alg=='PPOA':
        agent = PPOAgent(input_size=64, output_size=64, learning_rate=0.0015, clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01, num_epochs=10)
    elif alg=='PG':
        agent = Agent(input_size=64, output_size=64, learning_rate=0.0015)
    
    num_episodes = 10000
    num_eval_games = 500
    eval_interval = 100
    win_rates = []
    avg_durations = []
    avg_rewards = []

    for episode in range(1, num_episodes + 1):
        state = game.board.flatten()
        states, actions, rewards, dones = [], [], [], []
        done = False

        while not done:
            action = agent.choose_action(state)
            valid_moves = game.get_valid_moves()

            if len(valid_moves) == 0:
                break

            if action >= len(valid_moves):
                action = np.random.randint(0, len(valid_moves))

            next_state, reward, done = game.make_move(*valid_moves[action])

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            state = next_state.flatten()

        if states:
            if alg=='PPOA':
                agent.update(states, actions, rewards, dones)
            elif alg=='PG':
                agent.train_step(tf.convert_to_tensor(states, dtype=tf.float32),
                                 tf.convert_to_tensor(actions, dtype=tf.int32),
                                 tf.convert_to_tensor(rewards, dtype=tf.float32))


        if episode % eval_interval == 0:
            win_rate, avg_duration, avg_reward = evaluate(agent, num_eval_games)
            win_rates.append(win_rate)
            avg_durations.append(avg_duration)
            avg_rewards.append(avg_reward)
            print(f"Episode: {episode}/{num_episodes} | Win rate: {win_rate} | Avg. Duration: {avg_duration} | Avg. Reward: {avg_reward}")
    
    plot_results(win_rates, avg_durations, avg_rewards, num_episodes, num_eval_games, eval_interval)


# Evaluation function
def evaluate(agent, num_games):
    wins = 0
    game_durations = []
    total_rewards = []

    for _ in range(num_games):
        game = OthelloGame()
        done = False
        duration = 0
        total_reward = 0

        while not done:
            state = game.board.flatten()
            action = agent.choose_action(state)
            valid_moves = game.get_valid_moves()

            if len(valid_moves) == 0:
                break

            if action >= len(valid_moves):
                action = np.random.randint(0, len(valid_moves))

            next_state, reward, done = game.make_move(*valid_moves[action])

            state = next_state.flatten()
            duration += 1
            total_reward += reward

        if game.get_winner() == 1:
            wins += 1
        game_durations.append(duration)
        total_rewards.append(total_reward)

    win_rate = wins / num_games
    avg_duration = np.mean(game_durations)
    avg_reward = np.mean(total_rewards)
    return win_rate, avg_duration, avg_reward




# Plotting function
def plot_results(win_rates, avg_durations, avg_rewards, num_episodes, num_eval_games, eval_interval):
    episodes = range(eval_interval, num_episodes + 1, eval_interval)

    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.plot(episodes, win_rates)
    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.title("Agent Performance - Win Rate")

    plt.subplot(132)
    plt.plot(episodes, avg_durations)
    plt.xlabel("Episode")
    plt.ylabel("Average Duration")
    plt.title("Agent Performance - Average Duration")

    plt.subplot(133)
    plt.plot(episodes, avg_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Agent Performance - Average Reward")

    plt.tight_layout()
    plt.show()

# ... (existing code for the evaluation and plotting functions)

# Main function
def main():
    train(alg='PPOA')

if __name__ == "__main__":
    main()

