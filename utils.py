import random

def random_seed(min_value, max_value, step=10):
    min_val = max(-1125899906842624, min_value)  # 限制最小值
    max_val = min(1125899906842624, max_value)   # 限制最大值
    step = step / 10
    range2 = (max_val - min_val) / step
    seed = random.randint(0, int(range2)) * step + min_val
    return int(seed)