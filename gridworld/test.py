 #!/usr/bin/env python3
from curses import wrapper
from gridworld.environment import GridworldEnvironment

ACTIONS = {'w': 0, 's': 1, 'a': 2, 'd': 3}

def main(stdscr):
    env = GridworldEnvironment(5, 5)

    while True:
        obs, goal = env.reset()
        done = False
        while not done:
            stdscr.clear()
            stdscr.addstr(str(obs))
            stdscr.refresh()

            c = stdscr.getch()
            a = ACTIONS.get(chr(c))
            if a is not None:
                obs, _, _, done = env.step(a)

        stdscr.addstr("\nYay, you won! Press q to restart.")
        stdscr.refresh()
        while chr(stdscr.getch()) != 'q':
            pass

wrapper(main)
