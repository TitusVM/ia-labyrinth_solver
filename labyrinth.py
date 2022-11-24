"""Main class for solving labyrinths with genetic algorithms.

Tested Python 3.9+
"""

# TP4 IA Labyrinth solver
# Author : Titus Abele
# Date : 27.11.2022

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from deap import base, creator, tools
import math
import operator
from enum import Enum
from collections import namedtuple
import random
import time


#############################################################
#                         DISPLAY                           #
#############################################################

def display_labyrinth(grid: np.ndarray, start_cell: tuple, end_cell: tuple, solution=None):
    """Display the labyrinth matrix and possibly the solution with matplotlib.
    Free cell will be in light gray.
    Wall cells will be in dark gray.
    Start and end cells will be in dark blue.
    Path cells (start, end excluded) will be in light blue.
    :param grid np.ndarray: labyrinth matrix
    :param start_cell: tuple of i, j indices for the start cell
    :param end_cell: tuple of i, j indices for the end cell
    :param solution: list of successive tuple i, j indices who forms the path
    """
    grid = np.array(grid, copy=True)
    FREE_CELL = 19
    WALL_CELL = 16
    START = 0
    END = 0
    PATH = 2
    grid[grid == 0] = FREE_CELL
    grid[grid == 1] = WALL_CELL
    grid[start_cell] = START
    grid[end_cell] = END
    if solution:
        solution = solution[1:-1]
        for cell in solution:
                grid[cell] = PATH
    else:
        print("No solution has been found")
    plt.matshow(grid, cmap="tab20c")


#############################################################
#                         INIT                              #
#############################################################
def init_toolbox():
    """Initialise toolbox for easier comprehension
    """
    toolbox = base.Toolbox()
    return toolbox

#############################################################
#                         FITNESS                           #
#############################################################
def fitness(start_cell: tuple[int, int], individual: list[int], end_cell: tuple[int, int], grid: np.ndarray) -> tuple[int, int]:
    """Fitness function

    Args:
        start_cell (tuple[int, int]): The cell where the head of the path is currently located
        individual (list[int]): The particular chromosome that needs to be evaluated
        end_cell (tuple[int, int]): The target cell, where the head of the path wants to be
        grid (np.ndarray): The maze

    Returns:
        tuple[int, int]: A fitness value in a tuple because of some typing issues
    """
    path = parse_chromosome_V2(start_cell, end_cell, individual, grid)
    
    big_multiplier = 100
    small_multiplier = big_multiplier / 2
    fitness = None

    if end_cell in path:
        if indecisive_path(path): # some cells are visited more than once which isn't optimal
            fitness = path.index(end_cell) * small_multiplier
        else:
            fitness = path.index(end_cell)
    else:
        fitness = (math.dist(path[-1], end_cell)) * big_multiplier + len(path)
    return (fitness,)

def indecisive_path(path: list[tuple[int, int]]) -> bool:
    """A simple function that checks if the path crosses its own path at times

    Args:
        path (list[tuple[int, int]]): The path to be tested

    Returns:
        bool: Is the path crossing its own path
    """
    return Counter(path).most_common(1)[0][1] > 1

#############################################################
#                          PARSE                            #
#############################################################
    
def parse_chromosome_V2(start_cell: tuple[int, int], end_cell: tuple[int, int], chromosome: list[int], grid: np.ndarray) -> list[tuple[int, int]]:
    """Parsing a chromosome to receive a fully functionnal path

    Args:
        start_cell (tuple[int, int]): The cell where the head of the path is currently located
        end_cell (tuple[int, int]): The target cell, where the head of the path wants to be
        chromosome (list[int]): The chromosome containing the genes that tell directions and that need to be parsed into a path
        grid (np.ndarray): The grid, needed for wall and outside of grid identification

    Returns:
        list[tuple[int, int]]: The comprehensive path on the grid (list of coordonates to follow)
    """
    
    """
    Following code are the directions the head of the path can take to accomplish the instructions given in the chromosome
    """
    DIR_U 	= 0 			# Up 
    DIR_D 	= 1				# Down
    DIR_L 	= 2				# Left
    DIR_R 	= 3				# Right

    OPERATORS = {
        DIR_U: 	(  0, -1),
        DIR_D: 	(  0,  1),
        DIR_L: 	( -1,  0),
        DIR_R: 	(  1,  0)
    }
    
    path = [start_cell]
    
    prev_cell = (None, None)
    curr_cell = start_cell
    next_cell = start_cell
    
    for gene in chromosome:
        
        if in_grid(curr_cell, grid) and dead_end(prev_cell, curr_cell, end_cell, grid) and not curr_cell == start_cell:
            grid[next_cell[0]][next_cell[1]] = 1 # mark cell as being a wall to prevent further investigation
            curr_cell = prev_cell
            prev_cell = curr_cell
        
        next_cell = (curr_cell[0] + OPERATORS.get(gene)[0], curr_cell[1] + OPERATORS.get(gene)[1])
        if valid_cell(next_cell, prev_cell, grid):
            if next_cell == end_cell:
                path.append(next_cell)
                return path
            path.append(next_cell)
            prev_cell = curr_cell
            curr_cell = next_cell

    return path

def valid_cell(cell: tuple[int, int], prev_cell: tuple[int, int], grid: np.ndarray) -> bool:
    """Validates a cell on three conditions:
            * The cell needs to be located inside the grid;
            * The cell needs to not be a wall;
            * The cell needs to not be previously travelled on, this prevents indecisive paths.

    Args:
        cell (tuple[int, int]): The cell to be validated
        prev_cell (tuple[int, int]): The previous cell of the path
        grid (np.ndarray): The grid, needed for wall and outside of grid identification

    Returns:
        bool: Is the cell valid given the conditions or not
    """
    return  in_grid(cell, grid) \
        and not wall(cell, grid) \
        and not cell == prev_cell

def wall(cell: tuple[int, int], grid: np.ndarray) -> bool:
    """Checks if cell is a wall

    Args:
        cell (tuple[int, int]): Cell that needs to be checked
        grid (np.ndarray): Grid in which the cell needs to be checked

    Returns:
        bool: Is the cell a wall or not
    """
    return grid[cell[0]][cell[1]] == 1

def in_grid(cell: tuple[int, int], grid: np.ndarray) -> bool:
    """Checks if cell is inside of grid

    Args:
        cell (tuple[int, int]): Cell that needs to be checked
        grid (np.ndarray): Grid limits which could be over stepped

    Returns:
        bool: Is the cell in the grid or not
    """
    return cell[0] >= 0 \
         and cell[1] >= 0 \
         and cell[0] < grid.shape[0] \
         and cell[1] < grid.shape[1]

def dead_end(prev_cell: tuple[int, int], curr_cell: tuple[int, int], end_cell: tuple[int, int], grid: np.ndarray) -> bool:
    """Checks whether the given current cell is a deadend or not.

    Args:
        prev_cell (tuple[int, int]): Cell previously visited
        curr_cell (tuple[int, int]): Currently visited cell
        end_cell (tuple[int, int]): Final Cell
        grid (np.ndarray): Grid

    Returns:
        bool: Is the current cell a dead end or not
    """
    if curr_cell == end_cell:
        return False # If the cell is the target, it doesn't matter if the end_cell is a dead_end or not
    else:
        DIR_U = 0
        DIR_D = 1
        DIR_L = 2
        DIR_R = 3

        OPERATORS = {
            DIR_U: (  0, -1),
            DIR_D: (  0,  1),
            DIR_L: ( -1,  0),
            DIR_R: (  1,  0)
        }
            
        invalid_moves = []
        for operation in OPERATORS.values():
            neighbour = (curr_cell[0] + operation[0], curr_cell[1] + operation[1])
            if not valid_cell(neighbour, prev_cell, grid):
                invalid_moves.append(neighbour)
                
        return len(invalid_moves) == 4

#############################################################
#                         SOLVE                             #
#############################################################
def solve_labyrinth(grid: np.ndarray, start_cell: tuple, end_cell: tuple, max_time_s: float) -> list[tuple[int, int]]:
    """Attempt to solve the labyrinth by returning the best path found
    :param grid np.array: numpy 2d array
    :start_cell tuple: tuple of i, j indices for the start cell
    :end_cell tuple: tuple of i, j indices for the end cell
    :max_time_s float: maximum time for running the algorithm
    :return list: list of successive tuple i, j indices who forms the path
    """
    
    #############################################################
    #                           PREP                            #
    #############################################################
    # Start and end cells need to be set to not wall for this implementation to work
    grid[start_cell[0]][start_cell[1]] = 0
    grid[end_cell[0]][end_cell[1]] = 0
    
    # Get Grid copy for adding walls
    grid = np.array(grid, copy=True)
    
    #############################################################
    #                           TOOLS                           #
    #############################################################

    toolbox     = init_toolbox()     
    
    CHROMOSOME_LENGTH   = (int)((grid.shape[0] * grid.shape[1]) / 2)
    POPULATION_SIZE     = 50
    TOURN_SIZE          = 3
    CXPB    = 0.5
    MUTPB   = 0.8
    INDPB   = 0.1
    
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox.register("crossover", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=INDPB)
    toolbox.register("select", tools.selTournament, tournsize=TOURN_SIZE)
    toolbox.register("fitness", fitness, start_cell=start_cell, end_cell=end_cell, grid=grid)
    
    toolbox.register("gene", random.randint, 0, 3)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.gene, CHROMOSOME_LENGTH)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    pop = toolbox.population(POPULATION_SIZE)
    
    fitnesses = [toolbox.fitness(individual=individual) for individual in pop]

    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    #############################################################
    #                       IMPLEMENTATION                      #
    #############################################################

    #Timer
    start = time.perf_counter()
    
    while time.perf_counter() - start <= max_time_s - 1: # Remove one second from max_time to ensure respect of max_time

        # Select next gen individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone individuals
        offspring = [toolbox.clone(individual) for individual in offspring]
        
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.crossover(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = [toolbox.fitness(individual=individual) for individual in invalid_ind]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop = offspring
        best = tools.selBest(pop, k=1)[0]    
        
    
    return parse_chromosome_V2(start_cell, end_cell, best, grid) 
