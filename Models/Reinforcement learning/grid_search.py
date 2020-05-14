import numpy as np
import agent
import environment


episodes = 3000         # number of training episodes
episode_length = 50     # maximum episode length
x = 10                  # horizontal size of the box
y = 10                  # vertical size of the box
goal = [0, 3]           # objective point
discount = 0.9          # exponential discount factor

# alpha and epsilon profile
#alpha = np.ones(episodes) * 0.25
alpha_1 = -np.sort(-np.round(np.random.random(episodes),4))
alpha_2 = np.ones(episodes) * 0.25
epsilon_1 = np.linspace(0.9, 0.001, episodes)
epsilon_2 = np.linspace(0.7, 0.001, episodes)

alpha_all = [alpha_1, alpha_2]
epsilon_all = [epsilon_1, epsilon_2]
# initialize the agent

# In the center there is a hole 4x4
hole = [3, 4, 5, 6];
fixed_position = [[9,9], [6,8], [3,9], [7,4], [5,1],[2,8]];
# perform the training

# Q-learning
counter = 0;
revards_all = [];
counter_net = 0;
softmax = False         # set to true to use Softmax policy
sarsa = False           # set to true to use the Sarsa algorithm
learner = agent.Agent((x * y), 5, discount, max_reward=1, softmax=softmax, sarsa=sarsa)
moves_vec = [];

mean_rew_1 = []
print("\n## Q-learning ## \n") 
for alpha in alpha_all:
    for epsilon in epsilon_all:
        counter_net +=1
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
        tmp_rev = round(np.mean(revards_all),4)              
        print("For net #" + str(counter_net) + ", the average reward is: " + str(tmp_rev)) 
        mean_rew_1.append(tmp_rev)
    
    
# SARSA + e-greedy
print("\n## SARSA + e-greedy ## \n")  
counter = 0;
revards_all = [];
counter_net = 0;   
    
softmax = False         # set to true to use Softmax policy
sarsa = True           # set to true to use the Sarsa algorithm
learner = agent.Agent((x * y), 5, discount, max_reward=1, softmax=softmax, sarsa=sarsa)
moves_vec = [];

mean_rew_2=[]
for alpha in alpha_all:
    for epsilon in epsilon_all:
        counter_net +=1
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
        tmp_rev = round(np.mean(revards_all),4)              
        print("For net #" + str(counter_net) + ", the average reward is: " + str(tmp_rev)) 
        mean_rew_2.append(tmp_rev)
        
# SARSA + softmax
print("\n## SARSA + softmax ## \n")        
counter = 0;
revards_all = [];
counter_net = 0;        
softmax = True         # set to true to use Softmax policy
sarsa = True           # set to true to use the Sarsa algorithm
learner = agent.Agent((x * y), 5, discount, max_reward=1, softmax=softmax, sarsa=sarsa)
moves_vec = [];

mean_rew_3 = []
for alpha in alpha_all:
    for epsilon in epsilon_all:
        counter_net +=1
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
        tmp_rev = round(np.mean(revards_all),4)              
        print("For net #" + str(counter_net) + ", the average reward is: " + str(tmp_rev)) 
        mean_rew_3.append(tmp_rev)
#%%

winner_vec = []
winner_vec.append(np.argmax(mean_rew_1) + 1)
winner_vec.append(np.argmax(mean_rew_2) + 1)
winner_vec.append(np.argmax(mean_rew_3) + 1)

# Select the net with most wins
print("\n## Winners") 
print(winner_vec)