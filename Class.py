import pygame, math

white = (255,255,255)
black = (0,0,0)
grey = (80,80,80)
red = (255,0,0)
dark_red = (155,0,0)
blue = (0,0,255)
green = (0,255,0)
yellow = (250,250,0)
pink = (250,105,180)

'''Size of the game'''
size = 'Big'
if size == 'Big':
    m = 5 # Big
    x_g,y_g = 30, 30 # Big
    w,h = 30, 30 # Big
else:
    m = 1 # Small
    x_g,y_g = 5, 5 # Small
    w,h = 5, 5 # Small

class Grid(object):
    grid_w = w
    w = 6
    grid_h = h
    h = 5

    def draw_grid(screen):
        pygame.draw.rect(screen, black, [x_g, y_g, Grid.grid_w , Grid.grid_h])
        for row in range(Grid.h):
            for column in range(Grid.w):
                color = white
                pygame.draw.rect(screen,
                                 color,
                                 [x_g + (m + Grid.grid_w) * column + m,
                                  y_g + (m + Grid.grid_h) * row + m,
                                  Grid.grid_w, Grid.grid_h])

class Wall(pygame.Rect):
    color = grey
    def __init__(self,img,x,y):
        self.icon = pygame.image.load(str(img) + size + '.png')
        self.pos = [x, y]
        self.x = x
        self.y = y

class Obj:
    def __init__(self, tp, loc):
        self.tp = tp
        self.loc = loc

class Agent(pygame.Rect):
    color = yellow
    def __init__(self,img,x,y):
        self.icon = pygame.image.load(str(img) + size + '.png')
        self.pos = [x, y]
        self.x = x
        self.y = y

    def try_move(self, dir, wall_list):
        x_step = 0
        y_step = 0
        past_pos = self.pos
        if dir == 'up':
            y_step = - 1
        if dir == 'down':
            y_step = + 1
        if dir == 'left':
            x_step = - 1
        if dir == 'right':
            x_step = + 1
        fut_x = self.pos[0] + x_step
        fut_y = self.pos[1] + y_step
        fut_pos = [fut_x, fut_y]

        for wall in wall_list:
            if fut_pos == wall.pos:
                self.pos = past_pos
                break
            else:
                self.pos = fut_pos

class Start(pygame.Rect):
    color = blue
    def __init__(self,img,x,y):
         self.icon = pygame.image.load(str(img) + size +  '.png')
         self.pos = [x, y]

class Negativo(pygame.Rect):
    color = pink
    def __init__(self,img,x,y):
         self.icon = pygame.image.load(str(img) + size + '.png')
         self.pos = [x, y]
         self.x = x
         self.y = y

class Positivo(pygame.Rect):
    color = green
    def __init__(self,img,x,y):
         self.icon = pygame.image.load(str(img) + size + '.png')
         self.pos = [x, y]
         self.x = x
         self.y = y