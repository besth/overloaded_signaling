class Render1DGrid:
    def __init__(self, env):
        self.env = env

        self.goal_poss = self.env.goal_poss[:]
        self.grid = ["." for _ in range(self.env.env_length)]

        self.terminal = self.env.env_length - 3

        self.reset()

    def reset(self):
        self.grid = ["." for _ in range(self.env.env_length)]
        for gp in self.goal_poss:
            self.grid[gp] = 'G'

        self.grid[self.terminal] = 'T'

    def __call__(self, states: tuple):
        # reset at each call
        self.reset()

        # for i in range(len(states)):
        #     if self.grid[states[i]] == '.':
        #         self.grid[states[i]] = str(i)
        #     elif self.grid[states[i]] == 'G':
        #         self.grid[states[i]] = str(i)
        #         self.goal_poss.remove(states[i])
        #     else:
        #         self.grid[states[i]] = 'X'

        # print(states)
        for curr_poss in states[:2]:
            if self.grid[curr_poss] == '.':
                self.grid[curr_poss] = str(0)
            elif self.grid[curr_poss] == 'G':
                self.grid[curr_poss] = str(0)
                self.goal_poss.remove(curr_poss)
            else:
                self.grid[curr_poss] = 'X'

        # print('\n'.join(map(lambda x: ' '.join(x), self.grid)))
        print(*map(lambda x: ' '.join(x), self.grid))
