import random
from model_reader import read_model
from network import model
from game import check_victory, board_to_input
import numpy as np

def play_game(model):
    board = np.zeros(9)
    winner = None

    p1_turn = True
    while winner == None:
        if p1_turn:
            output = model.get_output(board_to_input(1, board))
            choice = np.argmax(output)
            output[choice] = -1
            print(str(board[:3]) + '\n' + str(board[3:6]) + '\n' + str(board[6:9]) + '\n:', choice)
            while board[choice] != 0:
                choice = np.argmax(output)
                output[choice] = 0
            board[choice] = 1
        else:
            print(str(board[:3]) + '\n' + str(board[3:6]) + '\n' + str(board[6:9]) + '\n:')
            choice = int(input(str(board[:3]) + '\n' + str(board[3:6]) + '\n' + str(board[6:9]) + '\n:'))
            while board[choice] != 0:
                choice = int(input(str(board[:3]) + '\n' + str(board[3:6]) + '\n' + str(board[6:9]) + '\n:'))
            board[choice] = 2
        p1_turn = not p1_turn
        if winner == None:
            winner = check_victory(board)
    print(str(board[:3]) + '\n' + str(board[3:6]) + '\n' + str(board[6:9]))
    print(winner)
    return winner

if __name__ == '__main__':
    play_game(read_model())
