import dill
import numpy as np
import agent
import environment
import matplotlib.pyplot as plt


episodes = 3000         # number of training episodes
episode_length = 50     # maximum episode length
x = 10                  # horizontal size of the box
y = 10                  # vertical size of the box
goal = [0, 3]           # objective point
discount = 0.9          # exponential discount factor
softmax = False         # set to true to use Softmax policy
sarsa = False           # set to true to use the Sarsa algorithm

# alpha and epsilon profile
#alpha = np.ones(episodes) * 0.25
alpha = -np.sort(-np.round(np.random.random(episodes),4))

epsilon = np.linspace(0.7, 0.001, episodes)

# initialize the agent
learner = agent.Agent((x * y), 5, discount, max_reward=1, softmax=softmax, sarsa=sarsa)
moves_vec = [];

# In the center there is a hole 4x4
hole = [3, 4, 5, 6];
fixed_position = [[9,9], [6,8], [3,9], [7,4], [5,1],[2,8]];
# perform the training
counter = 0;
revards_all = [];
for index in range(0, episodes):
    # start from a random state
    while(True):
        initial = [np.random.randint(0, x), np.random.randint(0, y)]
        
        if(not(initial[0] in hole and initial[1] in hole)):
            break
    # To have a common ground, the final 6 positions are fixed
    if(index > (episodes-1)-6):
        initial = fixed_position[counter]
        counter += 1
        
    # initialize environment
    state = initial
    env = environment.Environment(x, y, state, goal)
    reward = 0
    # run episode
    moves_tmp = [];
    for step in range(0, episode_length):
        # find state index
        state_index = state[0] * y + state[1]
        # choose an action
        action = learner.select_action(state_index, epsilon[index])
        # the agent moves in the environment
        result = env.move(action)
        
        moves_tmp.append(result[0]);
        
        # Q-learning update
        next_index = result[0][0] * y + result[0][1]
        learner.update(state_index, action, result[1], next_index, alpha[index], epsilon[index])
        # update state and reward
        reward += result[1]
        state = result[0]
    reward /= episode_length
    moves_vec.append(moves_tmp);
    
    revards_all.append(reward);
   
    
    # periodically save the agent
    if ((index + 1) % 10 == 0):
        with open('agent.obj', 'wb') as agent_file:
            dill.dump(agent, agent_file)
        print('Episode ', index + 1, ': the agent has obtained an average reward of ', reward, ' starting from position ', initial) 
#%% Print of the results
fixed_position = np.flip(fixed_position, axis =0);

        
for jj in range(6):
    treasure_map = moves_vec[len(moves_vec)-(jj+1)];
    checkerboard = np.zeros((x,y), dtype=int);
    guard = True;
    for tmp_position in treasure_map:
        # if(guard):
        #     checkerboard[tmp_position[0],tmp_position[1]] += 1;
        #     guard = False;
        #     continue;    
        checkerboard[tmp_position[0],tmp_position[1]] += 1;
    
    start_tmp = fixed_position[jj];
    checkerboard[start_tmp[0],start_tmp[1]] = 2;

    checkerboard[0,3] = 3;
    for i in range(10):
        print(checkerboard[i])
    print("#####")
    
#%% Reward plot
# tests_num = np.asarray(list(range(0, episodes-1)));
# tests_num_sampled = tests_num[0::9];
# revards_all_sampled = revards_all[0::9];
# plt.plot(tests_num_sampled,revards_all_sampled)
# plt.ylabel('Reward')
# plt.xlabel('Episodes')
# plt.show()
# print("Mean reward: " + str(np.mean(revards_all)))