import random

from core.GymEnvironment import EliminationEnv


def ai(env: EliminationEnv):
    return [random.randint(0, 19) for i in range(4)]
