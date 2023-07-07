# ReinOthello: Reinforcement Learning for Othello GamePG-PPO-OthelloSolver

This project provides an implementation of the Othello game environment and offers two reinforcement learning algorithms, Proximal Policy Optimization (PPO) and Policy Gradient (PG), to train an AI agent to play the game. The code includes a game environment class, policy network class, agent class, and training functions. The Othello game environment allows players to make moves, calculate rewards, and determine game outcomes. The policy network class defines the neural network architecture for the AI agent, while the agent class handles action selection and training. The training functions utilize the chosen algorithm to train the agent by interacting with the game environment and updating the model based on collected experiences. The project also provides evaluation functions to assess the agent's performance and plotting functions to visualize training progress. The main function orchestrates the training process using the PPO algorithm. With this project, users can train an AI agent to play Othello and evaluate its performance using reinforcement learning techniques.

## Installation
To run the code in this repository, please follow these installation steps:

1. Install Python: Ensure that Python is installed on your system. You can download the latest version of Python from the official Python website (https://www.python.org) and follow the installation instructions specific to your operating system.

2. Install Required Libraries: Open a terminal or command prompt and navigate to the project directory. Use the following command to install the required libraries:

   ```bash
   pip install numpy tensorflow matplotlib
   ```

   This command will install the necessary dependencies, including NumPy for numerical computations, TensorFlow for machine learning, and Matplotlib for data visualization.

By following these installation steps, you will have the necessary dependencies installed and be able to run the code in this repository. Make sure you have the required version of Python and the specified libraries to ensure smooth execution.

# Project Description
The code in this repository is an implementation of the Othello game environment and provides two different algorithms for training an AI agent to play the game: Proximal Policy Optimization (PPO) and Policy Gradient (PG).

* The `OthelloGame` class defines the game environment, which includes the game board, the current player, and methods for making moves, checking valid moves, calculating rewards, and determining the game's outcome. The game board is represented as a NumPy array, and the class provides functions to interact with the board, such as `get_valid_moves()`, `is_valid_move()`, `make_move()`, `calculate_reward()`, `is_game_over()`, and `get_winner()`.

* The `PolicyNetwork` class represents the policy network used by the AI agent. It is implemented using the TensorFlow framework and consists of fully connected layers. The class defines the layers of the network and includes a `call()` method to forward pass input data through the network.

* The `Agent` class implements the training logic for the PG algorithm. It takes the input size, output size, and learning rate as parameters. The agent has a policy network instance and an optimizer, and it provides methods for choosing an action based on the current state and training the agent using the `train_step()` function.

* The `PPOAgent` class extends the `Agent` class and includes the additional functionality required for the PPO algorithm. It has parameters such as clip ratio, value coefficient, entropy coefficient, and the number of epochs. The agent builds a model using TensorFlow's Keras API, defines the necessary layers, and includes methods for choosing an action (`choose_action()`) and updating the model based on collected experiences (`update()`).

* The `train()` function is responsible for training the AI agent using either the PPO or PG algorithm. It initializes the game environment, creates an instance of the agent, and runs a specified number of training episodes. Within each episode, the agent interacts with the game environment, collects states, actions, rewards, and dones, and updates the model using the appropriate algorithm.

* The `evaluate()` function evaluates the agent's performance by playing a specified number of games against the game environment. It tracks the win rate, average duration, and average reward to assess the agent's progress.

* The `plot_results()` function visualizes the training progress by plotting the win rate, average duration, and average reward over episodes.

* The `main()` function is the entry point of the program. It calls the `train()` function with the desired algorithm (either PPO or PG) to start the training process.

By following the code structure and running the appropriate functions, you can train an AI agent to play Othello using either the PPO or PG algorithm and evaluate its performance.

## Usage
Run the Code: Once the installations are complete, you can execute the code by running the following command:

   ```bash
   python PG_PPO.py
   ```

<p align="center">
  <img src="https://github.com/farkoo/PG-PPO-OthelloSolver/blob/master/Figure_1.png" alt="Training Progress">
  <br>
  <em>Figure 1: Training Progress - Win Rate, Average Duration, and Average Reward</em>
</p>

## Support

**Contact me @:**

e-mail:

* farzanehkoohestani2000@gmail.com

Telegram id:

* [@farzaneh_koohestani](https://t.me/farzaneh_koohestani)

## License
[MIT](https://github.com/farkoo/PG-PPO-OthelloSolver/blob/master/LICENSE)
&#0169; 
[Farzaneh Koohestani](https://github.com/farkoo)
