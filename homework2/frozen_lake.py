import gym
import numpy as np
from matplotlib import pyplot as plt
import tqdm

smooth = 20

env = gym.make('FrozenLake-v1', is_slippery=False)  # 构建环境


def train(epsilon, min_eps, learning, discount, episodes, policy):
    eps_decay = (epsilon - min_eps) / episodes
    num_states = env.observation_space.n

    # Initialize Q table
    # Q = np.random.uniform(low=0, high=0.1,
    #                       size=(num_states, env.action_space.n))
    Q = np.zeros((num_states, env.action_space.n))

    reward_list = []
    ave_reward_list = []

    action1, action2 = None, None

    for i in range(1, episodes + 1):
        done = False
        tot_reward, reward = 0, 0
        state1 = env.reset()

        # print('-------------------------')
        while not done:
            if i >= (episodes - 10):
                env.render()

            if policy == 'q-learning':
                # Determine next action - epsilon greedy strategy
                action1 = choose_action_eps_greedy(Q, epsilon, state1)
            elif policy == 'sarsa':
                action1 = action2 if action2 else choose_action_eps_greedy(Q, epsilon, state1)

            # Get next state and reward
            state2, reward, done, info = env.step(action1)

            # print(state1, action1, state2, reward)
            if policy == 'q-learning':
                delta = learning * (reward +
                                    discount * np.max(Q[state2]) -
                                    Q[state1, action1])
            elif policy == 'sarsa':
                # Determine next action - epsilon greedy strategy
                action2 = choose_action_eps_greedy(Q, epsilon, state2)

                delta = learning * (reward +
                                    discount * Q[state2, action2] -
                                    Q[state1, action1])

            Q[state1, action1] += delta

            tot_reward += reward
            state1 = state2

        epsilon -= eps_decay

        reward_list.append(tot_reward)

        if i % smooth == 0:
            ave_reward = np.mean(reward_list)
            ave_reward_list.append(ave_reward)
            reward_list = []

            print('Episode {} Average Reward: {}'.format(i, ave_reward))

    return ave_reward_list


def choose_action_eps_greedy(Q, epsilon, state):
    if np.random.random() < 1 - epsilon:
        max_q = np.max(Q[state])
        actions = np.where(Q[state] == max_q)[0]
        action = np.random.choice(actions)
    else:
        action = np.random.randint(0, env.action_space.n)
    return action


epsilon = 0.1
min_eps = 0.1
learning = 0.1
discount = 0.9
episodes = 500
policy = 'q-learning'

for policy in ['sarsa']:
    for learning in [0.1]:
        for discount in [0.9]:
            label = f'{policy}-lr={learning}-gamma={discount}'
            reward_list = train(epsilon, min_eps, learning, discount, episodes, policy=policy)
            plt.plot(smooth * np.arange(len(reward_list)), reward_list, label=label)

env.close()

# Plot Rewards
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Reward vs Episodes')
plt.show()
