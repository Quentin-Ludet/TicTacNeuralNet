from network import model
import numpy as np
import math
import random
from numba import jit, generated_jit

@jit
def board_to_input(player_num, board):
    inputs = []
    for space in board:
        space_input = [0, 0]
        if space == player_num:
            space_input[0] = 1
        elif space != 0:
            space_input[1] = 1
        inputs.extend(space_input)
    return np.array(inputs)

@jit
def check_victory(board):
    for i in range(0, 7, 3):
        if board[i] == board[i+1] and board[i] == board[i+2] and board[i] != 0:
            return board[i]
    for i in range(3):
        if board[i] == board[i+3] and board[i] == board[i+6] and board[i] != 0:
            return board[i]
    if board[0] == board[4] and board[0] == board[8] and board[0] != 0:
        return board[0]
    if board[2] == board[4] and board[2] == board[6] and board[2] != 0:
        return board[2]
    if 0 not in board:
        return 0
    return None
        
@jit
def play_game(player1, player2):
    board = np.zeros(9)
    winner = None

    p1_turn = True
    skipped = False
    turn = 1
    while winner == None:
        if p1_turn:
            output = player1.get_output(board_to_input(1, board))
            choice = np.argmax(output)
            output[choice] = -1
            while board[choice] != 0:
                choice = np.argmax(output)
                output[choice] = 0
            board[choice] = 1
        else:
            output = player2.get_output(board_to_input(2, board))
            choice = np.argmax(output)
            output[choice] = -1
            while board[choice] != 0:
                choice = np.argmax(output)
                output[choice] = 0
            board[choice] = 2
        winner = check_victory(board)
        if winner == 1:
            player1.score += 20 - 2 * turn
            player2.score -= 10 - turn
        elif winner == 2:
            player2.score += 20 - 2 * turn
            player1.score -= 10 - turn
        p1_turn = not p1_turn
        turn += 1
    return winner

model.num_inputs = 18
model.num_intermediate_layers = 3
model.layer_height = 18
model.num_outputs = 9
population = 500

models = []
@generated_jit
def main():
    for i in range(population):
        models.append(model(random=True, name=str(i), mutation_chance=50))
    for generation in range(10000):
        random.shuffle(models)
        # for i in range(math.floor(population/2)):
        #     winner = play_game(models[i], models[i+1])
        #     if winner == 1:
        #         models.pop(i+1)
        #     else:
        #         models.pop(i)
        for mod in models:
            mod.score = 0
        for i in range(population):
            for n in range(1, 6):
                play_game(models[i], models[i-n])
                #play_game(models[i-n], models[i])
        models.sort(key=lambda x: x.score, reverse=True)
        del models[math.floor(population/2):]
        for i in range(math.floor(population/2)):
            models.append(model.create_child(models[random.randrange(0, math.floor(population/2))],
                                            models[random.randrange(0, math.floor(population/2))]))
        print(generation, '-', models[0].score, models[math.floor(population/4)].score)
    models[0].save()
    return None

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('saving')
        models[0].save()