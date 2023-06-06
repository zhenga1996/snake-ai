import torch
import random
from collections import deque
from snake import SnakeGameAI
from util import get_state
from model import Linear_QNet, QTrainer
from plotter import plot
import pickle

MAX_MEM = 100_000
BATCH_SIZE = 1_000
LEARNING_RATE = 0.001


class Agent:
    def __init__(self):
        self.num_games = 0
        self.epsilon = 0  # parameter for randomness
        self.gamma = 0.85  # discount rate
        self.mem = deque(maxlen=MAX_MEM)  # auto popleft when max
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, learning_rate=LEARNING_RATE, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, game_over):
        self.mem.append((state, action, reward, next_state, game_over))

    def train_long_mem(self):
        if len(self.mem) > BATCH_SIZE:
            mini_sample = random.sample(self.mem, BATCH_SIZE)  # return list of tuples
        else:
            mini_sample = self.mem

        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def train_short_mem(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        # random moves => tradeoff exploration / exploitation => gradual improvement
        self.epsilon = 80 - self.num_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
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
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    try:
        plot_scores, plot_mean_scores, total_score, record, agent = pickle.load(open("data.pkl", "rb"))
    except (OSError, IOError) as e:
        pass

    while True:
        state_old = get_state(game)

        final_move = agent.get_action(state_old)

        reward, game_over, score = game.play_step(final_move)
        state_new = get_state(game)

        agent.train_short_mem(state_old, final_move, reward, state_new, game_over)

        agent.remember(state_old, final_move, reward, state_new, game_over)

        if game_over:
            game.reset()
            agent.num_games += 1
            agent.train_long_mem()

            if score > record:
                record = score
                agent.model.save()

            print('Game:', agent.num_games)
            print('Score:', score)
            print('Record:', record, '\n')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.num_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            # Save data
            pickle.dump([
                plot_scores,
                plot_mean_scores,
                total_score,
                record,
                agent,
            ], open("data.pkl", "wb"))


if __name__ == '__main__':
    train()
