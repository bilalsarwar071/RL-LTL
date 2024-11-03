import collections
import os
import random

import gymnasium as gym  # Updated gym import to gymnasium
import numpy as np
import pygame
from gymnasium.spaces import Box, Discrete

main_dir = os.path.split(os.path.abspath(__file__))[0]
assets_dir = os.path.join(main_dir, 'assets')


def _load_image(name):
    fullname = os.path.join(assets_dir, name)
    try:
        image = pygame.image.load(fullname)
    except pygame.error as e:  # Catch the pygame error
        print(f"Cannot load image: {fullname}")
        raise SystemExit(e)  # Use the caught error for exit
    image = image.convert_alpha()
    return image


def _calculate_topleft_position(position, sprite_size):
    return sprite_size * position[1], sprite_size * position[0]


class _Collectible(pygame.sprite.Sprite):
    _COLLECTIBLE_IMAGES = {
        ('square', 'purple'): 'purple_square.png',
        ('circle', 'purple'): 'purple_circle.png',
        ('square', 'beige'): 'beige_square.png',
        ('circle', 'beige'): 'beige_circle.png',
        ('square', 'blue'): 'blue_square.png',
        ('circle', 'blue'): 'blue_circle.png'
    }

    def __init__(self, sprite_size, shape, colour):
        self.name = shape + '_' + colour
        self._sprite_size = sprite_size
        self.shape = shape
        self.colour = colour
        pygame.sprite.Sprite.__init__(self)
        image = _load_image(self._COLLECTIBLE_IMAGES[(self.shape, self.colour)])
        self.image = pygame.transform.scale(image, (sprite_size, sprite_size))
        self.rect = self.image.get_rect()
        self.position = None

    def reset(self, position):
        self.position = position
        self.rect.topleft = _calculate_topleft_position(position, self._sprite_size)


class _Player(pygame.sprite.Sprite):
    def __init__(self, sprite_size):
        self.name = 'player'
        self._sprite_size = sprite_size
        pygame.sprite.Sprite.__init__(self)
        image = _load_image('character.png')
        self.image = pygame.transform.scale(image, (sprite_size, sprite_size))
        self.rect = self.image.get_rect()
        self.position = None

    def reset(self, position):
        self.position = position
        self.rect.topleft = _calculate_topleft_position(position, self._sprite_size)

    def step(self, move):
        self.position = (self.position[0] + move[0], self.position[1] + move[1])
        self.rect.topleft = _calculate_topleft_position(self.position, self._sprite_size)


class CollectEnv(gym.Env):
    """
    This environment consists of an agent attempting to collect a number of objects.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 5
    }

    _BOARDS = {
        'original': ['##########',
                     '#        #',
                     '#        #',
                     '#    #   #',
                     '#   ##   #',
                     '#  ##    #',
                     '#   #    #',
                     '#        #',
                     '#        #',
                     '##########'],
    }

    _AVAILABLE_COLLECTIBLES = [
        ('square', 'purple'),
        ('circle', 'purple'),
        ('square', 'beige'),
        ('circle', 'beige'),
        ('square', 'blue'),
        ('circle', 'blue')
    ]

    _WALL_IMAGE = 'wall.png'
    _GROUND_IMAGE = 'ground.png'

    _ACTIONS = {
        0: (-1, 0),  # North
        1: (0, 1),  # East
        2: (1, 0),  # South
        3: (0, -1)  # West
    }

    _SCREEN_SIZE = (400, 400)
    _SPRITE_SIZE = 40

    def __init__(self, board='original', available_collectibles=None, start_positions=None,
                 goal_condition=lambda _: True, render_mode=None):
        """
        Create a new instance of the Collect environment.
        """
        self.render_mode = render_mode  # Added render_mode
        self.viewer = None
        self.start_positions = start_positions
        self.goal_condition = goal_condition
        self.action_space = Discrete(4)

        self.available_collectibles = available_collectibles \
            if available_collectibles is not None else self._AVAILABLE_COLLECTIBLES

        self.board = np.array([list(row) for row in self._BOARDS[board]])

        self.observation_space = Box(0, 255, [self._SCREEN_SIZE[0], self._SCREEN_SIZE[1], 3], dtype=np.uint8)
        pygame.init()
        pygame.display.init()
        pygame.display.set_mode((1, 1))

        self._bestdepth = pygame.display.mode_ok(self._SCREEN_SIZE, 0, 32)
        self._surface = pygame.Surface(self._SCREEN_SIZE, 0, self._bestdepth)
        self._background = pygame.Surface(self._SCREEN_SIZE)
        self._clock = pygame.time.Clock()

        self.free_spaces = list(map(tuple, np.argwhere(self.board != '#')))
        self._build_board()

        self.initial_positions = None
        self.collectibles = pygame.sprite.Group()
        self.collected = pygame.sprite.Group()
        self.render_group = pygame.sprite.RenderPlain()
        self.player = _Player(self._SPRITE_SIZE)
        self.render_group.add(self.player)

        for shape, colour in self.available_collectibles:
            self.collectibles.add(_Collectible(self._SPRITE_SIZE, shape, colour))

    def _build_board(self):
        for col in range(self.board.shape[1]):
            for row in range(self.board.shape[0]):
                position = _calculate_topleft_position((row, col), self._SPRITE_SIZE)
                image = self._WALL_IMAGE if self.board[row, col] == '#' else self._GROUND_IMAGE
                image = _load_image(image)
                image = pygame.transform.scale(image, (self._SPRITE_SIZE, self._SPRITE_SIZE))
                self._background.blit(image, position)

    def _draw_screen(self, surface):
        surface.blit(self._background, (0, 0))
        self.render_group.draw(surface)
        surface_array = pygame.surfarray.array3d(surface)
        observation = np.copy(surface_array).swapaxes(0, 1)
        del surface_array
        return observation

    def reset(self, **kwargs):
        collected = self.collected.sprites()
        self.collectibles.add(collected)
        self.collected.empty()
        self.render_group.empty()
        self.render_group.add(self.player)
        self.render_group.add(self.collectibles.sprites())

        render_group = sorted(self.render_group, key=lambda x: x.name)
        if self.start_positions is None:
            positions = random.sample(self.free_spaces, k=len(render_group))
        else:
            start_positions = collections.OrderedDict(sorted(self.start_positions.items()))
            positions = start_positions.values()

        self.initial_positions = collections.OrderedDict()

        to_remove = list()
        for position, sprite in zip(positions, render_group):
            if position is None:
                to_remove.append(sprite)
            else:
                if sprite.name != 'player':
                    self.initial_positions[sprite] = position
                sprite.reset(position)

        self.collected.add(to_remove)
        self.render_group.remove(to_remove)
        return self._draw_screen(self._surface), {}  # Reset now returns observation and info

    def step(self, action):
        direction = self._ACTIONS[action]
        prev_pos = self.player.position
        next_pos = (direction[0] + prev_pos[0], direction[1] + prev_pos[1])
        if self.board[next_pos] != '#':
            self.player.step(direction)

        collected = pygame.sprite.spritecollide(self.player, self.collectibles, True)
        self.collected.add(collected)
        self.render_group.remove(collected)
        done, reward = False, -0.1
        if len(collected) > 0:
            if self.goal_condition(collected[0]):
                done, reward = True, 1.0

        return self._draw_screen(self._surface), reward, done, False, {'collected': self.collected}  # Added 'truncated' flag in step

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                pygame.quit()
                self.viewer = None
            return

        if mode == 'rgb_array':  # Check for 'rgb_array' mode for recording
            return self._draw_screen(self._surface)
        elif mode == 'human':
            if self.viewer is None:
                self.viewer = pygame.display.set_mode(self._SCREEN_SIZE, 0, self._bestdepth)

            self._clock.tick(10 if mode != 'human' else 2)
            self._draw_screen(self.viewer)
            pygame.display.flip()

if __name__ == "__main__":
    env = CollectEnv()  # Set render_mode for testing
    obs, _ = env.reset()  # Update reset for new return values
    env.render()
    for _ in range(10000):
        obs, reward, done, truncated, _ = env.step(env.action_space.sample())  # Updated step method
        env.render()
        if done:
            env.reset()

