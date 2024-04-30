import pygame
import random
import numpy as np
from enum import Enum
from scipy.spatial.distance import euclidean

pygame.init()

# Constants
WIDTH = 640
HEIGHT = 480
BLOCK_SIZE = 20
SPEED = 12
TIMELIMIT = 15
WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 200)
class MazeGameAI:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT + 30))
        pygame.display.set_caption('Maze Game')
        self.clock = pygame.time.Clock()
        self.score = 0

        self.reset()
    
    def reset(self):

        # Generate a random maze layout
        self.generate_random_maze()
        
        # Generate random positions for the player and goal within the maze boundaries
        self.player_pos = self.find_empty_space()
        self.goal_pos = self.find_empty_space()
        self.total_dist = euclidean(self.player_pos, self.goal_pos)

        self.start_time = pygame.time.get_ticks()
        self.ui_update()
        #pygame.time.delay(2000)


    def generate_random_maze(self):
        self.maze = []
        for y in range(HEIGHT // BLOCK_SIZE):
            row = []
            for x in range(WIDTH // BLOCK_SIZE):
                # Randomly choose whether to place a wall or an empty space
                if random.random() < 0.2:  # Adjust the probability as desired
                    row.append("#")  # Wall
                else:
                    row.append(" ")  # Empty space
            self.maze.append(row)

        # Ensure that the outer edges are walls
        for y in range(HEIGHT // BLOCK_SIZE):
            self.maze[y][0] = "#"
            self.maze[y][-1] = "#"
        for x in range(WIDTH // BLOCK_SIZE):
            self.maze[0][x] = "#"
            self.maze[-1][x] = "#"
	
    def find_empty_space(self):
        while True:
            x = random.randint(1, (WIDTH // BLOCK_SIZE) - 2)
            y = random.randint(1, (HEIGHT // BLOCK_SIZE) - 2)
            if (self.maze[y][x] == " "  and (self.maze[y-1][x] != "#" or self.maze[y+1][x] != "#" or  self.maze[y][x-1] != "#" or self.maze[y][x+1] != "#"))  :
                return (x, y)
        
               

    def draw_maze(self):
        self.screen.fill(WHITE)
        for y in range(len(self.maze)):
            for x in range(len(self.maze[y])):
                if self.maze[y][x] == "#":
                    pygame.draw.rect(self.screen, BLACK, (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

    def draw_player(self):
        pygame.draw.rect(self.screen, BLUE, (self.player_pos[0] * BLOCK_SIZE, self.player_pos[1] * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

    def draw_goal(self):
        pygame.draw.rect(self.screen, GREEN, (self.goal_pos[0] * BLOCK_SIZE, self.goal_pos[1] * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

    def ui_update(self):
        self.draw_maze()
        self.draw_goal()
        self.draw_player()
        pygame.display.flip()

    def move_player(self, dx, dy):
        new_x = self.player_pos[0] + dx
        new_y = self.player_pos[1] + dy
        if 0 <= new_x < WIDTH // BLOCK_SIZE and 0 <= new_y < HEIGHT // BLOCK_SIZE:
            if self.maze[new_y][new_x] != "#":
                self.player_pos = (new_x, new_y)
                return True
            else:
                return False
            
    def play_step(self, action,COLLISION,TL,ENERGY,TOTAL):
        #print(euclidean((0,0),((HEIGHT // BLOCK_SIZE)-1,(WIDTH // BLOCK_SIZE)-1)))= 38 avg diagonal dist of maze
        res = True
        x1,y1=self.player_pos[0],self.player_pos[1]  #getting agent's prev coordinate positions value i.e before making current move 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
        if np.array_equal(action, [0, 0, 1 ,0]):
            res = self.move_player(0, -1)
        elif np.array_equal(action, [0, 0, 0 ,1]):
            res = self.move_player(0, 1)
        elif np.array_equal(action, [0, 1, 0 ,0]):
            res = self.move_player(-1, 0)
        elif np.array_equal(action, [1, 0, 0 ,0]):
            res = self.move_player(1, 0)

        

        self.ui_update()
        gameOver = False
        reward = 50 * (1 - euclidean(self.player_pos, self.goal_pos)/self.total_dist)
        if reward <0:
            reward=-10
        self.score = 0 if reward<0 else round(reward, 2)

        #calculating Energy variable and drive stamina
        x2,y2=self.player_pos[0],self.player_pos[1]  #CURRENT PLAYER POSITION COORDINATES
        delta= euclidean(self.goal_pos,(x2,y2))-euclidean(self.goal_pos,(x1,y1))   
        if delta > 0 :   
            ENERGY += -1
            stamina=0- ENERGY
            #reward+= -(stamina)
            
        #else:
        #ENERGY += 0.5

        if not res:
            #updating collision homeostatic variable and calculating drive:pain 
            #WEIGHT FACTOR ALSO ADDED FOR DRIVE PAIN 
            COLLISION += -1
            TOTAL[1]+=1
            pain = 0 - COLLISION
            reward = -5 #+ -(TOTAL[1]/TOTAL[0])*(pain)
        elif self.player_pos == self.goal_pos:
            print("Congratulations! You reached the goal!")
            TOTAL[2]+=1
           
            reward = 2000
            # Change background to green
            for y in range(len(self.maze)):
                for x in range(len(self.maze[y])):
                    if self.maze[y][x] == " ":
                        pygame.draw.rect(self.screen, GREEN, (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            pygame.display.flip()
            pygame.time.delay(1000)  # Wait for 1 second
            self.running = False
            gameOver = True
        
        current_time = pygame.time.get_ticks()
        elapsed_time = (current_time - self.start_time) // 1000
        remaining_time = TIMELIMIT - elapsed_time
        #Updating time left and computing urgency drive 
        TL=remaining_time
        urgency= 15-TL
        #weight calculation for drive urgency 
        if self.total_dist != 0:
            x= (euclidean((x2,y2),self.goal_pos))/self.total_dist
        else:
            x=0
        if x>1 :
            x=1
        w1 = 2*(1-(1/(1+x)))
        #updating Reward to include urgency drive
        #reward=reward+ -w1*(urgency) 
        # Display the remaining time
        font = pygame.font.Font(None, 36)
        timer = font.render(f"Time: {remaining_time}", True, BLACK)
        timer_rect = timer.get_rect()
        timer_rect.midbottom = (WIDTH // 2, HEIGHT + 26)
        self.screen.blit(timer, timer_rect)
        pygame.display.flip()
        self.clock.tick(SPEED)
        if self.total_dist == 0:
            self.score=10
        else:
            self.score=10 * (1 - euclidean(self.player_pos, self.goal_pos)/self.total_dist)
        self.score = 0 if self.score<0 else round(self.score, 2)
        if (remaining_time <= 0):
            print("Time's up! Game over!")
            pygame.time.delay(1000)
            gameOver = True
        
        if gameOver:
            self.screen.fill(WHITE)
            font = pygame.font.Font(None, 36)
            score_text = font.render(f"Score: {self.score}", True, BLACK)
            score_rect = score_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
            self.screen.blit(score_text, score_rect)
            pygame.display.flip()
            pygame.time.delay(2000)
        
        return reward, gameOver, self.score,COLLISION,TL,ENERGY,TOTAL

if __name__ == "__main__":
    game = MazeGameAI()
    game.play()
    pygame.quit()
