import numpy as np
from snake import Direction, Point


def get_state(game):
    head = game.snake[0]
    point_left = Point(head.x - 20, head.y)
    point_right = Point(head.x + 20, head.y)
    point_up = Point(head.x, head.y - 20)
    point_down = Point(head.x, head.y + 20)

    dir_left = game.direction == Direction.LEFT
    dir_right = game.direction == Direction.RIGHT
    dir_up = game.direction == Direction.UP
    dir_down = game.direction == Direction.DOWN

    state = [
        # Danger straight
        (dir_right and game.is_collision(point_right)) or
        (dir_left and game.is_collision(point_left)) or
        (dir_up and game.is_collision(point_up)) or
        (dir_down and game.is_collision(point_down)),

        # Danger right
        (dir_up and game.is_collision(point_right)) or
        (dir_down and game.is_collision(point_left)) or
        (dir_left and game.is_collision(point_up)) or
        (dir_right and game.is_collision(point_down)),

        # Danger left
        (dir_down and game.is_collision(point_right)) or
        (dir_up and game.is_collision(point_left)) or
        (dir_right and game.is_collision(point_up)) or
        (dir_left and game.is_collision(point_down)),

        # Move direction
        dir_left, dir_right, dir_up, dir_down,

        # Food location
        game.food.x < game.head.x,  # food left
        game.food.x > game.head.x,  # food right
        game.food.y < game.head.y,  # food up
        game.food.y > game.head.y,  # food down
    ]

    return np.array(state, dtype=int)