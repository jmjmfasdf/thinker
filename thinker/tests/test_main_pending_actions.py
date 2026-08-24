from argparse import Namespace
from types import MethodType

import torch

from thinker.main import Env


class BarrierCore:
    def __init__(self):
        self.call = 0

    def step(self, action, model_net):
        self.call += 1
        if self.call == 1:
            info = self._info(
                real_transition=[False, False],
                stored=[True, False],
                accepted=[4, -1],
                executed=[-1, -1],
            )
            reward = torch.zeros(2)
        else:
            info = self._info(
                real_transition=[True, True],
                stored=[False, True],
                accepted=[-1, 3],
                executed=[4, 3],
            )
            reward = torch.tensor([1.0, 2.0])
        return {}, reward, torch.zeros(2, dtype=torch.bool), torch.zeros(
            2, dtype=torch.bool
        ), info

    @staticmethod
    def _info(real_transition, stored, accepted, executed):
        return {
            "real_transition": torch.tensor(real_transition),
            "stored_action_mask": torch.tensor(stored),
            "accepted_primary_action": torch.tensor(accepted),
            "accepted_control": torch.full((2,), -1),
            "executed_primary_action": torch.tensor(executed),
            "think_reward": torch.zeros(2),
            "stage_end": torch.zeros(2, dtype=torch.bool),
            "real_done": torch.zeros(2, dtype=torch.bool),
        }


def test_model_buffer_uses_probability_from_real_action_storage_call():
    env = Env.__new__(Env)
    env.flags = Namespace(wrapper_type=0, min_replay_ratio=0)
    env.dynamic_search = True
    env.env_n = 2
    env.device = torch.device("cpu")
    env.pri_action_shape = (2,)
    env.action_prob_shape = (2, 5)
    env.require_prob = True
    env.train_model = True
    env.parallel = False
    env.rank = 1
    env.counter = 0
    env.model_net = object()
    env.env = BarrierCore()
    env.status = {"finish": False}
    env._logger = type("Logger", (), {"info": lambda *args, **kwargs: None})()
    env.pending_primary_action = torch.zeros(2, dtype=torch.long)
    env.pending_action_prob = torch.zeros(2, 5)
    env.pending_action_valid = torch.zeros(2, dtype=torch.bool)
    env.last_accepted_primary_action = torch.zeros(2, dtype=torch.long)
    env.last_accepted_search_control = torch.zeros(2, dtype=torch.long)
    env.think_episode_return = torch.zeros(2)
    env._train_model = MethodType(lambda self: self.status, env)

    captured = []

    def capture(
            self, state, reward, done, truncated_done, info,
            primary_action, action_prob, real_step_mask=None):
        captured.append((primary_action.clone(), action_prob.clone()))

    env._write_send_model_buffer = MethodType(capture, env)

    first_prob = torch.arange(10, dtype=torch.float32).view(2, 5)
    second_prob = first_prob + 100
    env.step(
        torch.tensor([4, 0]),
        search_control=torch.zeros(2, dtype=torch.long),
        action_prob=first_prob,
    )
    assert captured == []

    env.step(
        torch.tensor([0, 3]),
        search_control=torch.zeros(2, dtype=torch.long),
        action_prob=second_prob,
    )

    assert len(captured) == 1
    actions, probabilities = captured[0]
    assert actions.tolist() == [4, 3]
    torch.testing.assert_close(probabilities[0], first_prob[0])
    torch.testing.assert_close(probabilities[1], second_prob[1])
