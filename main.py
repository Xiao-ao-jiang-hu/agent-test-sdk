import sys

from ai import ai
from core.GymEnvironment import EliminationEnv
from utils.utils import write_to_judger


class Controller:
    def __init__(self):
        init_info = input().split(" ")
        self.seat = int(init_info[1])
        self.env = EliminationEnv()
        self.env.reset(int(init_info[0]))

    def run(self, ai):
        while 1:
            if self.seat == 0:
                op = self.env.num_to_coord(ai(self.env))
                print(f"send operation {op}", file=sys.stderr)
                write_to_judger(f"{op[0]} {op[1]} {op[2]} {op[3]}")
                self.env.step(self.env.coord_to_num(op))

                enemy_op = input().split()
                print(f"read operation {enemy_op}", file=sys.stderr)
                enemy_op = [int(i) for i in enemy_op]
                self.env.step(self.env.coord_to_num(enemy_op))
            else:
                enemy_op = input().split()
                print(f"read operation {enemy_op}", file=sys.stderr)
                enemy_op = [int(i) for i in enemy_op]
                self.env.step(self.env.coord_to_num(enemy_op))

                op = self.env.num_to_coord(ai(self.env))
                print(f"send operation {op}", file=sys.stderr)
                write_to_judger(f"{op[0]} {op[1]} {op[2]} {op[3]}")
                self.env.step(self.env.coord_to_num(op))


if __name__ == "__main__":
    print("init done", file=sys.stderr)
    controller = Controller()
    controller.run(ai)
