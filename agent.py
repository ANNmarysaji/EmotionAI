import torch
import random
import numpy as np
from collections import deque
from game import MazeGameAI
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
ENERGY=0
TL= 15
class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(8, 256, 4)
        self.trainer = QTrainer(self.model, lr = LR, gamma = self.gamma)

    def get_state(self, game):
        x, y = game.player_pos[0], game.player_pos[1]

        danger_up = 1 if game.maze[y-1][x] == '#' else 0
        danger_down = 1 if game.maze[y+1][x] == '#' else 0
        danger_left = 1 if game.maze[y][x-1] == '#' else 0
        danger_right = 1 if game.maze[y][x+1] == '#' else 0

        if game.goal_pos[1] > game.player_pos[1]:
            goal_up = 1
            goal_down = 0
        else:
            goal_up = 0
            goal_down = 1

        if game.goal_pos[0] > game.player_pos[0]:
            goal_right = 1
            goal_left = 0
        else:
            goal_right = 0
            goal_left = 1
        
        state = [danger_up, danger_down, danger_left, danger_right,
        goal_up, goal_down, goal_left, goal_right]

        return np.array(state, dtype = int)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 1500 - self.n_games
        final_move = [0, 0, 0, 0]
        if random.randint(0, 3000) < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move
    

def train():
    
    plot_scores = []
    plot_mean_scores = []
    plot_mean_collratio=[]
    total_score = 0
    record = 0
    ENERGY=0
    COLLISION =0
    TL=15
    #list with 2 values total no of moves, total no of collisions 
    TOTAL = [0]*3
    agent = Agent()
    game = MazeGameAI()
    while(True):
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        TOTAL[0]+=1
        reward, done, score,COLLISION,TL,ENERGY,TOTAL = game.play_step(final_move,COLLISION,TL,ENERGY,TOTAL)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            print("Game :", agent.n_games+1, "Score :", score, "Total moves :", TOTAL[0], "Total collisions: ",TOTAL[1],"Time left:",TL)
            collision_ratio=TOTAL[1]/TOTAL[0]
            COLLISION =0
            ENERGY=0
            TOTAL[0]=TOTAL[1]=0
            agent.n_games += 1
            agent.train_long_memory()

            if score> record:
                record = score
                agent.model.save()
            
            mean_collision_ratio=collision_ratio/agent.n_games
            plot_mean_collratio.append(mean_collision_ratio)
            plot_scores.append(score)
            total_score  += score
            success_ratio=TOTAL[2]/agent.n_games
            mean_score = total_score/ agent.n_games
            plot_mean_scores.append(mean_score)
            print("mean score:" , mean_score,"mean collision ratio:",mean_collision_ratio,"Success ratio :",success_ratio)
            plot(plot_scores, plot_mean_scores,agent.n_games)
        

if __name__ == '__main__':
    train()