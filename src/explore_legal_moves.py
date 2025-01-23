# short function to test # legal moves per turn in chess
import chess
import random

def explore_legal_moves():
    board = chess.Board()
    counter = 0
    color = []
    num_legal_moves = []
    move = []

    while counter < 100:
        # print the number of legal moves
        num_legal_moves.append(len(list(board.legal_moves)))
        color.append(board.turn)

        # shuffle the moves, then pick the first one and apply it
        moves = list(board.legal_moves)
        random.shuffle(moves)
        if len(moves) > 0:
            board.push(moves[0])
            move.append(moves[0])
            counter += 1
        else:
            print("Game over.")
            print(board.result())
            break

    for item in zip(color, num_legal_moves, move):
        print(item)

if __name__ == "__main__":
    explore_legal_moves()