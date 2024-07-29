import tkinter as tk
from tkinter import messagebox
import math

class TicTacToe:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic Tac Toe")
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'
        self.buttons = [[None for _ in range(3)] for _ in range(3)]
        self.create_widgets()

    def create_widgets(self):
        for i in range(3):
            for j in range(3):
                self.buttons[i][j] = tk.Button(self.root, text=' ', font='Arial 20 bold', width=5, height=2,
                                               command=lambda i=i, j=j: self.user_move(i, j))
                self.buttons[i][j].grid(row=i, column=j)

    def user_move(self, row, col):
        if self.board[row][col] == ' ':
            self.board[row][col] = 'X'
            self.buttons[row][col].config(text='X')
            if self.check_winner('X'):
                self.show_winner('X')
            elif self.is_board_full():
                self.show_tie()
            else:
                self.root.after(500, self.computer_move)

    def computer_move(self):
        row, col = self.get_computer_move()
        self.board[row][col] = 'O'
        self.buttons[row][col].config(text='O')
        if self.check_winner('O'):
            self.show_winner('O')
        elif self.is_board_full():
            self.show_tie()

    def get_computer_move(self):
        best_score = -math.inf
        best_move = None
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == ' ':
                    self.board[i][j] = 'O'
                    score = self.minimax(self.board, 0, False)
                    self.board[i][j] = ' '
                    if score > best_score:
                        best_score = score
                        best_move = (i, j)
        return best_move

    def minimax(self, board, depth, is_maximizing):
        if self.check_winner('O'):
            return 1
        elif self.check_winner('X'):
            return -1
        elif self.is_board_full():
            return 0

        if is_maximizing:
            best_score = -math.inf
            for i in range(3):
                for j in range(3):
                    if board[i][j] == ' ':
                        board[i][j] = 'O'
                        score = self.minimax(board, depth + 1, False)
                        board[i][j] = ' '
                        best_score = max(score, best_score)
            return best_score
        else:
            best_score = math.inf
            for i in range(3):
                for j in range(3):
                    if board[i][j] == ' ':
                        board[i][j] = 'X'
                        score = self.minimax(board, depth + 1, True)
                        board[i][j] = ' '
                        best_score = min(score, best_score)
            return best_score

    def check_winner(self, player):
        for i in range(3):
            if all(self.board[i][j] == player for j in range(3)):
                return True
            if all(self.board[j][i] == player for j in range(3)):
                return True
        if all(self.board[i][i] == player for i in range(3)):
            return True
        if all(self.board[i][2-i] == player for i in range(3)):
            return True
        return False

    def is_board_full(self):
        return all(all(cell != ' ' for cell in row) for row in self.board)

    def show_winner(self, player):
        messagebox.showinfo("Game Over", f"{player} wins!")
        self.root.quit()

    def show_tie(self):
        messagebox.showinfo("Game Over", "It's a tie!")
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    game = TicTacToe(root)
    root.mainloop()
