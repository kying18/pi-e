import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policy.bc_policy import BcPolicy
from policy.action_chunking_policy import ActionChunkingPolicy
from policy.act_policy import ActModel


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    print("Parameter counts:\n")

    bc = BcPolicy(use_checkpoint=False)
    print(f"BC Policy:              {count_params(bc.model):>12,}")

    ac = ActionChunkingPolicy(use_checkpoint=False)
    print(f"Action Chunking Policy: {count_params(ac.model):>12,}")

    act = ActModel()
    print(f"ACT Policy:             {count_params(act):>12,}")
