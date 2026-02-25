from eval.eval import Eval
from env.moving_object import MovingObjectEnv
from policy.policy import Policy
from expert.expert_policy import ExpertPolicy
from policy.bc_policy import BcPolicy
from policy.action_chunking_policy import ActionChunkingPolicy
from policy.act_policy import ActPolicy
from policy.vit_policy import ViTPolicy
from policy.multi_img_bc_policy import MultiImgBcPolicy
from policy.flow_matching_policy import FlowMatchingPolicy

policies = {
  "expert": ExpertPolicy(),
  "bc_dagger": BcPolicy(use_checkpoint=True, checkpoint_name="bc_policy_dagger"),
  "bc": BcPolicy(use_checkpoint=True, checkpoint_name="bc_policy"), 
  "action_chunking": ActionChunkingPolicy(use_checkpoint=True, checkpoint_name="action_chunking_policy"),
  "action_chunking_rh4": ActionChunkingPolicy(use_checkpoint=True, checkpoint_name="action_chunking_policy", actions_per_inference=4),
  "act": ActPolicy(use_checkpoint=True, checkpoint_name="act_policy_small"),
  "act_rh4": ActPolicy(use_checkpoint=True, checkpoint_name="act_policy_small", actions_per_inference=4),
  "vit": ViTPolicy(use_checkpoint=True, checkpoint_name="vit_policy_patch16"),
  "vit_rh4": ViTPolicy(use_checkpoint=True, checkpoint_name="vit_policy_patch16", actions_per_inference=4),
  "flow_matching": FlowMatchingPolicy(use_checkpoint=True, checkpoint_name="flow_matching_policy", chunk_size=8),
  "flow_matching_rh4": FlowMatchingPolicy(use_checkpoint=True, checkpoint_name="flow_matching_policy", chunk_size=8, actions_per_inference=4),
  "random": None,
}

def run_eval_global():
    for policy_name in policies.keys():
        run_eval_single(policy_name)

def run_eval_single(policy_name):
    env = MovingObjectEnv()
    policy = policies[policy_name]
    eval = Eval(env, policy, num_runs=100, max_steps=100, seed=42)
    results = eval.eval()
    results.calculate_all_metrics()
    print(f"{results}")
    print("-" * 100)

if __name__ == "__main__":
    # run_eval_global()

    run_eval_single("flow_matching")
    run_eval_single("flow_matching_rh4")