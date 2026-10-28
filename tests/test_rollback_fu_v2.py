import unittest

import numpy as np
import torch

from src.fl.fedavg_core import ClientWorker, fedavg, model, state
from src.fu import fu_v2
from src.fu.rollback import fedavg_trajectory, rollback_retrain


def tiny_workers(seed=0, clients=3, n=60, dim=4):
    rng = np.random.default_rng(seed)
    workers = []
    for i in range(clients):
        x = rng.normal(size=(n, dim)).astype(np.float32)
        y = (x[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(np.int64)
        y[:2] = [0, 1]
        workers.append(ClientWorker(i, x, y, np.arange(n) + i * n))
    return workers


class RollbackTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = {'lr': .01, 'local_epochs': 1, 'batch_size': 16}
        self.workers = tiny_workers()
        self.initial = model(4, [6, 5], 42)

    def assert_same(self, a, b):
        for k, v in state(a).items():
            self.assertTrue(torch.equal(v, state(b)[k]), k)

    def test_trajectory_reproduces_fedavg_exactly(self):
        direct, _ = fedavg(self.initial, self.workers, self.cfg, 4, 42)
        traj, logs = fedavg_trajectory(self.initial, self.workers, self.cfg, 4, 42)
        self.assertEqual(len(traj), 5)
        self.assertEqual([e['round'] for e in logs], [0, 1, 2, 3])
        self.assert_same(direct, traj[-1])
        self.assert_same(self.initial, traj[0])

    def test_rollback_endpoints(self):
        traj, _ = fedavg_trajectory(self.initial, self.workers, self.cfg, 4, 42)
        exact, logs = rollback_retrain(traj, self.workers, self.cfg, 0, 42)
        self.assert_same(exact, traj[-1])          # same data + same seeds = exact retraining
        self.assertEqual(len(logs), 4)
        same, logs = rollback_retrain(traj, self.workers, self.cfg, 4, 42)
        self.assert_same(same, traj[-1])
        self.assertEqual(logs, [])
        with self.assertRaises(ValueError):
            rollback_retrain(traj, self.workers, self.cfg, 5, 42)

    def test_rollback_uses_requested_data_and_costs_r_rounds(self):
        traj, _ = fedavg_trajectory(self.initial, self.workers, self.cfg, 4, 42)
        other = tiny_workers(seed=1)
        m, logs = rollback_retrain(traj, other, self.cfg, 2, 42)
        self.assertEqual([e['round'] for e in logs], [2, 3])
        with self.assertRaises(AssertionError):
            self.assert_same(m, traj[-1])


class FuV2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        g = torch.Generator().manual_seed(0)
        self.x = torch.randn(200, 4, generator=g)
        self.y = (self.x[:, 0] > 0).long()
        self.m = model(4, [6, 5], 42)

    def test_projection_bound_holds(self):
        out, log = fu_v2.ga_projected(self.m, self.x[:50], self.y[:50], self.x[50:], self.y[50:],
                                      radius=.05, loss_cap=10., steps=20, lr=.05, batch=16)
        d = (fu_v2.flat(out) - fu_v2.flat(self.m)).norm().item()
        self.assertLessEqual(d, .05 * (1 + 1e-4))
        self.assertTrue(all(e['dist'] <= .05 + 1e-9 for e in log))

    def test_cap_zero_without_retain_term_is_exact_noop(self):
        out, _ = fu_v2.ga_projected(self.m, self.x[:50], self.y[:50], self.x[50:], self.y[50:],
                                    radius=10., loss_cap=0., steps=5, lr=.1, lam_r=0., batch=16)
        self.assertTrue(torch.equal(fu_v2.flat(out), fu_v2.flat(self.m)))

    def test_ascent_increases_forget_loss(self):
        before = fu_v2.mean_ce(self.m, self.x[:50], self.y[:50])
        out, _ = fu_v2.ga_projected(self.m, self.x[:50], self.y[:50], self.x[50:], self.y[50:],
                                    radius=5., loss_cap=50., steps=30, lr=.01, lam_r=0., batch=16)
        self.assertGreater(fu_v2.mean_ce(out, self.x[:50], self.y[:50]), before)

    def test_kd_identical_student_teacher_without_hard_loss_is_noop(self):
        # SGD, not Adam: Adam divides by sqrt(v)+eps, so a ~1e-9 float-rounding gradient still produces
        # a step of order lr. With SGD the step stays ~1e-10, which is what "no-op" should mean.
        out = fu_v2.kd_distill(self.m, self.m, self.x, self.y, tau=3., lam_soft=1., lam_hard=0.,
                               steps=5, lr=.1, batch=16, optimizer="sgd")
        self.assertLess((fu_v2.flat(out) - fu_v2.flat(self.m)).norm().item(), 1e-6)

    def test_class_weighted_cap(self):
        yf = torch.tensor([1, 1, 1, 0])
        cap = fu_v2.class_weighted_cap(self.m, self.x, self.y, yf)
        ce = torch.nn.functional.cross_entropy(self.m(self.x), self.y, reduction='none')
        manual = .75 * ce[self.y == 1].mean() + .25 * ce[self.y == 0].mean()
        self.assertAlmostEqual(cap, manual.item(), places=5)


if __name__ == '__main__':
    unittest.main()
