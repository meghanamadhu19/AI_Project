AI_PROJECT: Reinforcement Learning Technique to assist Cab Drivers

Reinforcement learning (RL) is a type of machine learning where an agent learns to make decisions based on feedback from its environment. It has been used successfully in a variety of applications including robotics, game-playing, and recommendation systems.

In this project, cab drivers could benefit from RL-based route optimization. RL could be used to learn the best routes based on factors like traffic, weather, and time of day. The system could also help cab drivers make better decisions about when to pick up passengers, which routes to take, and when to take breaks.

One key challenge with implementing RL in a real-world setting is determining how to provide feedback to the agent. In this case, feedback might come in the form of rewards for taking efficient routes or penalties for taking longer routes. Designing a good reward system that incentivizes desirable behavior can be difficult, but is critical to the success of an RL system.

Another challenge is ensuring that the system is able to adapt to changing conditions over time. For example, traffic patterns may change seasonally or due to construction. The system will need to learn and adapt to these changes to continue providing useful recommendations to cab drivers.

To implement the above, we need two files- Env.py and DQN_Agent.ipynb.

Env.py :

Training an agent that interacts with its surroundings is the goal of a reinforcement learning challenge. By taking certain acts, the agent enters several scenarios known as states. Rewarding behaviors can result in positive and negative.
Maximizing its total payout across five episodes is the agent's only goal in this situation. Everything that occurs in the environment between the initial and final or terminal states is referred to as this episode. We assist the agent in learning the best actions through practice. This is the plan or course of action. This is the "environment class"; each method (function) serves a particular function within the class. begin setting the hyperparameters.

# Defining hyperparameters
m = 5  # number of cities, ranges from 0 ..... m-1
t = 24  # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5  # Per hour fuel and other costs
R = 9  # per hour revenue from a passenger

Make the state into a vector so that the NN may use it. Using this method, a given state is transformed into a vector format. The vector has a length of m + t + d.
We use simply State as an input and have 2 DQN architectures. We accept inputs for State and Action.
Because we will receive Q(s, a) for each action and must only run the NN once for each state, Architecture 1 (which accepts only State as input) performs better than Architecture 2. Do the thing for which Q(s, a) is the most.

__init__(): The CabDriver class is defined with an __init__ method that initializes the state, action space, and state space. It also selects a random initial state.
state_encod_arch1(): The state_encod_arch1 method encodes the given state into a vector representation.
requests(): The requests method determines the number of requests based on the location and returns the possible actions for the given state.
new_time_day(): The new_time_day method calculates the new time and day after a driver's journey based on the current state and the time taken.
next_state_func(): The next_state_func method calculates the next state based on the current state and the chosen action. It also calculates the various time durations.
reset(): The reset method returns the action space, state space, and initial state.
reward_func(): The reward_func method calculates the reward based on the waiting, pickup, and drop times.
step(): The step method performs a step in the environment by calculating the rewards, next state, and total time spent based on the current state and chosen action.
This code defines the core functionality of the CabDriver environment, allowing you to interact with the environment, perform actions, and receive rewards.


DQN_Agent.ipynb:

The function save_obj saves a Python object as a pickle file with the given name. The function takes two arguments: obj is the object to be saved, and name is the name to be used for the file. With open(name + '.pkl', 'wb') as f: opens the file in binary write mode and assigns it to the file object f. The .pkl file extension is added to the name to indicate that the file is a pickle file. pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL) writes the object obj to the file f in pickle format, using the highest protocol available (pickle.HIGHEST_PROTOCOL).

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)



__init__(): Initializes the DQNAgent with the size of the state and action space. It also defines the hyperparameters for the DQN like discount_factor, learning_rate, epsilon and batch_size. It creates a replay memory using deque and a target model to update weights.

build_model(): It defines the neural network used to approximate Q-values. In this implementation, the neural network consists of three fully connected layers with relu activation function.

get_action(): This method selects an action based on the epsilon-greedy policy. If a random number is less than the epsilon value, it selects a random action, else it selects the action with the highest Q-value.

append_sample(): This method saves a tuple of (state, action, reward, next_state, done) to the replay memory.

train_model(): It samples a mini-batch of size batch_size from the replay memory, and trains the neural network using the sampled data.

save_tracking_states(): This method saves the Q-value of the action being tracked at a particular state to states_tracked.

save_test_states(): This method saves the Q-value of the action being tracked during testing to states_test.

get_model_weights(): This method returns the weights of the neural network model.

save(): This method saves the neural network model to a file with the given name.


DQN Block code:

The code trains a DQN agent to learn how to navigate a CabDriver environment, in which the agent has to decide which cab to drive and for how long to earn maximum reward within a given episode time.

The environment is initialized in each episode.
At each time step, the agent selects an action from the set of possible actions based on an epsilon-greedy policy, and evaluates the reward and next state.
The agent's memory stores the experience tuple (state, action, reward, next_state, terminal_state).
The agent's neural network is trained by randomly sampling from its memory and updating the Q-values based on the Bellman equation.
The agent's epsilon value is decayed after each episode.
The total rewards obtained in each episode are stored and tracked.
The Q-values of the state-action pairs we are tracking are saved after every 5 episodes.
The model weights are saved as a .h5 file after every 5 episodes and as a pickle file after training is complete.

agent.memory: agent.memory is a list of experiences that the agent has collected during training. Each experience is a tuple of (state, action, reward, next_state, done), where state and next_state are the state vectors before and after taking an action, action is the index of the action taken, the reward is the reward obtained by taking that action in that state, and done is a boolean flag indicating whether the episode has ended after taking that action. The agent uses this memory to train its Q-network in an off-policy manner, by randomly sampling a batch of experiences from the memory and updating the Q-network based on the Bellman equation.


agent.states_tracked is a dictionary that keeps track of the Q-values of specific state-action pairs that we are interested in tracking throughout the training process. The keys of the dictionary are tuples representing the state-action pairs, and the values are lists that store the Q-values of that pair at every 5th episode during training. The state-action pairs to be tracked are initialized at the start of training and are defined in the DQNAgent class.


state_tracked_sample = [agent.states_tracked[i] for i in range(len(agent.states_tracked)) if agent.states_tracked[i] < 1000]

This code creates a list called state_tracked_sample that contains all the elements from the agent.states_tracked list whose value is less than 1000.


plt.figure(0, figsize=(16,7))
plt.title('Q_value for state [1,2,3]  action (0,2)')
xaxis = np.asarray(range(0, len(state_tracked_sample)))
## We are using log scale because the initial q_values are way to high compared to the steady state value
plt.semilogy(xaxis,np.asarray(state_tracked_sample))
plt.show()


The purpose of this code is to filter out the state-action pairs that have not been visited frequently enough during training. The agent.states_tracked list keeps track of the number of times each state-action pair has been visited. By filtering out state-action pairs with low visit count, the code aims to focus on the more promising state-action pairs that are more likely to lead to better rewards in future episodes.


Graphs and their Behaviour:

->The first graph plots the Q-value of a specific state-action pair ([1,2,3], (0,2)) over the episodes. It first filters out the Q-values of that state-action pair from the states_tracked list and stores them in state_tracked_sample. Then it plots the logarithm of the Q-values against the episode number on the x-axis. The y-axis is also logarithmic, which means that the Q-values will be displayed on a logarithmic scale. This is done because the initial Q-values may be too high compared to the steady-state value, making it difficult to observe the changes in the Q-values over time. The resulting plot will show how the Q-values for the state-action pair change over the course of the training episodes.


->The second graph plots the rewards obtained per episode over the course of the training process. It uses the episodes and rewards_per_episode lists that were being appended to in the training loop. The x-axis of the plot represents the episode number, while the y-axis represents the rewards obtained in each episode. This can help us see how the agent's performance improves over time as it learns to navigate the environment more effectively.

->In the third graph, it looks like the agent is gradually learning and improving as the rewards per episode are increasing over time. The graph shows that the agent's performance has increased after every few episodes, and it has learned to make better decisions. However, there is still some variance in the reward, which may indicate that the agent has not yet converged to the optimal policy. Overall, it seems like the agent is learning, and it would be interesting to see how it performs in a more extended training period or with different hyperparameters.

->The fourth plot shows the total rewards obtained by the agent for each episode during training. We can see that the rewards are increasing initially but start to plateau around episode 300-400. This suggests that the agent has converged to a near-optimal policy and further training may not lead to significant improvements in performance. However, it's possible that the agent could benefit from longer training or tuning of hyperparameters.

->Based on the fifth graph, it appears that the average rewards per episode is increasing and converging to a stable value after around 800 episodes. This suggests that the DQN agent is learning and improving over time.

->The final plot shows how the agent's exploration rate changes over time and how it learns to make better decisions through experience.

