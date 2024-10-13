from ai import ai
from core.GymEnvironment import EliminationEnv
from utils.utils import write_to_judger


class Controller:
    def __init__(self):
        self.seat = int(input().split(" ")[1])
        self.env = EliminationEnv().reset(int(input().split()[0]))

    def run(self, ai: function):
        while 1:
            if self.seat == 0:
                op = ai(self.env)
                write_to_judger(f"{op[0]} {op[1]} {op[2]} {op[3]}")
                self.env.step([int(i) for i in input().split()])
            else:
                self.env.step([int(i) for i in input().split()])
                op = ai(self.env)
                write_to_judger(f"{op[0]} {op[1]} {op[2]} {op[3]}")


if __name__ == "__main__":
    controller = Controller()
    controller.run(ai)
