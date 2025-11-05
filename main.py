import pygame
import numpy as np
import heapq
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math

CELL_SIZE = 50
WALL_COLOR = (0, 0, 255); PATH_COLOR = (255, 255, 255)
START_COLOR = (0, 255, 0); END_COLOR = (255, 0, 0)
ASTAR_PATH_COLOR = (255, 165, 0); AGENT_COLOR = (255, 0, 255)
SCREEN_WIDTH = 0; SCREEN_HEIGHT = 0

class Maze:
    def __init__(self, filepath):
        with open(filepath, 'r') as f: lines = f.read().splitlines()
        self.grid = np.array([list(line) for line in lines])
        self.height, self.width = self.grid.shape
        start_pos = np.where(self.grid == 'I'); end_pos = np.where(self.grid == 'F')
        self.start_pos = (start_pos[0][0], start_pos[1][0])
        self.end_pos = (end_pos[0][0], end_pos[1][0])

    def is_wall(self, x, y):
        col, row = int(x / CELL_SIZE), int(y / CELL_SIZE)
        if 0 <= row < self.height and 0 <= col < self.width:
            return self.grid[row, col] == '#'
        return True

    def draw(self, screen, a_star_path=None):
        for row in range(self.height):
            for col in range(self.width):
                char = self.grid[row, col]
                rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                color = PATH_COLOR
                if char == '#': color = WALL_COLOR
                elif char == 'I': color = START_COLOR
                elif char == 'F': color = END_COLOR
                pygame.draw.rect(screen, color, rect)
        if a_star_path:
            points = [(c * CELL_SIZE + CELL_SIZE // 2, r * CELL_SIZE + CELL_SIZE // 2) for r, c in a_star_path]
            if len(points) > 1:
                pygame.draw.lines(screen, ASTAR_PATH_COLOR, False, points, 3)

class FuzzyController:
    def __init__(self):
        dist_frontal = ctrl.Antecedent(np.arange(0, CELL_SIZE * 2, 1), 'dist_frontal')
        dist_izquierda = ctrl.Antecedent(np.arange(0, CELL_SIZE * 2, 1), 'dist_izquierda')
        dist_derecha = ctrl.Antecedent(np.arange(0, CELL_SIZE * 2, 1), 'dist_derecha')
 
        angulo_giro = ctrl.Consequent(np.arange(-45, 46, 1), 'angulo_giro')

        dist_frontal['cerca'] = fuzz.trimf(dist_frontal.universe, [0, 0, CELL_SIZE * 0.75])
        dist_frontal['lejos'] = fuzz.trimf(dist_frontal.universe, [CELL_SIZE * 0.5, CELL_SIZE * 2, CELL_SIZE * 2])
        
        for dist_var in [dist_izquierda, dist_derecha]:
            dist_var['cerca'] = fuzz.trimf(dist_var.universe, [0, 0, CELL_SIZE * 0.6])
            dist_var['lejos'] = fuzz.trimf(dist_var.universe, [CELL_SIZE * 0.4, CELL_SIZE * 2, CELL_SIZE * 2])

        angulo_giro['brusco_izquierda'] = fuzz.trimf(angulo_giro.universe, [-45, -45, -20])
        angulo_giro['suave_izquierda'] = fuzz.trimf(angulo_giro.universe, [-30, -15, 0])
        angulo_giro['recto'] = fuzz.trimf(angulo_giro.universe, [-10, 0, 10])
        angulo_giro['suave_derecha'] = fuzz.trimf(angulo_giro.universe, [0, 15, 30])
        angulo_giro['brusco_derecha'] = fuzz.trimf(angulo_giro.universe, [20, 45, 45])

        self.rules = [
            ctrl.Rule(dist_frontal['cerca'] & dist_derecha['lejos'], angulo_giro['brusco_derecha']),
            ctrl.Rule(dist_frontal['cerca'] & dist_izquierda['lejos'], angulo_giro['brusco_izquierda']),
            
            ctrl.Rule(dist_izquierda['cerca'], angulo_giro['suave_derecha']),
            ctrl.Rule(dist_derecha['cerca'], angulo_giro['suave_izquierda']),
            ctrl.Rule(dist_frontal['lejos'] & dist_izquierda['lejos'] & dist_derecha['lejos'], angulo_giro['recto'])
        ]

        self.control_system = ctrl.ControlSystem(self.rules)
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)

    def compute(self, frontal, izquierda, derecha):
        self.simulation.input['dist_frontal'] = frontal
        self.simulation.input['dist_izquierda'] = izquierda
        self.simulation.input['dist_derecha'] = derecha
        self.simulation.compute()
        return self.simulation.output['angulo_giro']
class Agent:
    def __init__(self, start_pos_grid, initial_target_grid):
        self.x = start_pos_grid[1] * CELL_SIZE + CELL_SIZE // 2
        self.y = start_pos_grid[0] * CELL_SIZE + CELL_SIZE // 2
        self.speed = 2
        self.sensor_length = CELL_SIZE * 2
        self.fuzzy_controller = FuzzyController()

        target_x = initial_target_grid[1] * CELL_SIZE + CELL_SIZE // 2
        target_y = initial_target_grid[0] * CELL_SIZE + CELL_SIZE // 2
        self.angle = math.degrees(math.atan2(target_y - self.y, target_x - self.x))

    def sense_and_decide(self, maze, a_star_path, path_index):
        correction_angle = 0
        if path_index < len(a_star_path):
            target_row, target_col = a_star_path[path_index]
            target_x, target_y = target_col * CELL_SIZE + CELL_SIZE // 2, target_row * CELL_SIZE + CELL_SIZE // 2
            angle_to_target = math.degrees(math.atan2(target_y - self.y, target_x - self.x))
            angle_diff = (angle_to_target - self.angle + 180) % 360 - 180
            correction_angle = np.clip(angle_diff, -20, 20)

        dist_f = self.cast_ray(self.angle, maze)
        dist_l = self.cast_ray(self.angle - 45, maze)
        dist_r = self.cast_ray(self.angle + 45, maze)
        
        avoidance_angle = self.fuzzy_controller.compute(dist_f, dist_l, dist_r)
        
        self.angle += avoidance_angle + correction_angle * 0.1

    def cast_ray(self, angle, maze):
        rad = math.radians(angle)
        for dist in range(int(self.sensor_length)):
            x = self.x + dist * math.cos(rad)
            y = self.y + dist * math.sin(rad)
            if maze.is_wall(x, y): return dist
        return self.sensor_length
        
    def update(self, maze, a_star_path, path_index):
        self.sense_and_decide(maze, a_star_path, path_index)
        rad = math.radians(self.angle)
        new_x, new_y = self.x + self.speed * math.cos(rad), self.y + self.speed * math.sin(rad)
        
        if not maze.is_wall(new_x, new_y):
            self.x, self.y = new_x, new_y

    def draw(self, screen):
        pygame.draw.circle(screen, AGENT_COLOR, (int(self.x), int(self.y)), CELL_SIZE // 4)
        end_x = self.x + 15 * math.cos(math.radians(self.angle))
        end_y = self.y + 15 * math.sin(math.radians(self.angle))
        pygame.draw.line(screen, (0,0,0), (int(self.x), int(self.y)), (int(end_x), int(end_y)), 2)

def a_star_search(maze):
    start, goal = maze.start_pos, maze.end_pos
    open_set = []; heapq.heappush(open_set, (0, start))
    came_from = {start: None}
    g_score = { (r, c): float('inf') for r in range(maze.height) for c in range(maze.width) }; g_score[start] = 0
    
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = [];
            while current: path.append(current); current = came_from[current]
            return path[::-1]
            
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = (current[0] + dr, current[1] + dc)
            if 0 <= neighbor[0] < maze.height and 0 <= neighbor[1] < maze.width and maze.grid[neighbor] != '#':
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
    return None

def heuristic(a, b): return abs(a[0] - b[0]) + abs(a[1] - b[1])

def main():
    global SCREEN_WIDTH, SCREEN_HEIGHT
    pygame.init()
    maze = Maze('laberintos/laberinto1.txt')
    SCREEN_WIDTH, SCREEN_HEIGHT = maze.width * CELL_SIZE, maze.height * CELL_SIZE
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Proyecto IA - Navegación Difusa")
    
    a_star_path = a_star_search(maze)
    if not a_star_path or len(a_star_path) < 2:
        print("No se encontró un camino o el camino es muy corto."); return
    
    agent = Agent(maze.start_pos, a_star_path[1])
    clock = pygame.time.Clock()
    path_index = 1
    
    simulation_finished = False
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
        if not simulation_finished:
            if path_index < len(a_star_path):
                target_r, target_c = a_star_path[path_index]
                dist_to_target = math.hypot(agent.x - (target_c * CELL_SIZE + CELL_SIZE//2), agent.y - (target_r * CELL_SIZE + CELL_SIZE//2))
                if dist_to_target < CELL_SIZE * 0.75:
                    path_index += 1
            else:
                simulation_finished = True

            agent.update(maze, a_star_path, path_index)
        
        screen.fill((0, 0, 0))
        maze.draw(screen, a_star_path)
        agent.draw(screen)
        
        pygame.display.flip()
        clock.tick(60)
        
    pygame.quit()

if __name__ == '__main__':
    main()