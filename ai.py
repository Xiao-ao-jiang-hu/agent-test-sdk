import random
import sys
import time
from copy import deepcopy

from core.GymEnvironment import EliminationEnv


def ai(env: EliminationEnv):
    """_summary_

    Args:
        env (EliminationEnv): 每局游戏维护的唯一局面，请不要直接操作该局面

    Returns:
        int: 操作对应的序号，可以使用env的coord_to_num方法将坐标转换为操作序号
    """
    max_reward = 0
    max_action = 0
    start = time.time()
    actions = list(range(160000))
    random.shuffle(actions)
    for i in actions:
        env_copy = deepcopy(env)
        board, reward, end = env_copy.step(i)
        if reward > max_reward:
            max_reward = reward
            max_action = i

        if time.time() - start > 0.9:
            break
    print(f"searched {i+1} operations", file=sys.stderr)
    return max_action
