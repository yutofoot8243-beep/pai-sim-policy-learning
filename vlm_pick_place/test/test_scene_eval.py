import math
import os
import unittest

from vlm_pick_place.evaluation import classify_attempt, cubes_sorted, format_summary, summarize
from vlm_pick_place.scene import (ANGLE_DEG, CUBE_R, MIN_CUBE_GAP, MIN_CUBE_TRAY_GAP, MIN_TRAY_GAP,
                                  TRAY_R, Placement, Scene, in_tray, sample_scene)
from vlm_pick_place.sim_world import parse_pose_info

FIXTURE = os.path.join(os.path.dirname(__file__), 'fixtures', 'pose_info.txt')


def fixed_scene():
    """赤・青のブロックと3枚のトレイ. トレイは向き 0 で並べる."""
    return Scene(seed=0,
                 cubes=[Placement('red_cube', 'red', 0.20, 0.0, 0.0),
                        Placement('blue_cube', 'blue', 0.15, 0.10, 0.0)],
                 trays=[Placement('red_tray', 'red', 0.27, 0.10, 0.0),
                        Placement('green_tray', 'green', 0.27, 0.0, 0.0),
                        Placement('blue_tray', 'blue', 0.27, -0.10, 0.0)])


def pose(x, y, z=0.025):
    return (x, y, z, 0.0)


class TestSampleScene(unittest.TestCase):
    def test_same_seed_same_scene(self):
        self.assertEqual(sample_scene(7).to_dict(), sample_scene(7).to_dict())
        self.assertNotEqual(sample_scene(7).to_dict(), sample_scene(8).to_dict())

    def test_constraints_hold(self):
        for seed in range(200):
            s = sample_scene(seed)
            self.assertTrue(1 <= len(s.cubes) <= 3)
            self.assertEqual(sorted(t.color for t in s.trays), ['blue', 'green', 'red'])
            self.assertEqual(len({c.color for c in s.cubes}), len(s.cubes))
            for objs, (r_lo, r_hi) in ((s.cubes, CUBE_R), (s.trays, TRAY_R)):
                for o in objs:
                    r, deg = math.hypot(o.x, o.y), math.degrees(math.atan2(o.y, o.x))
                    self.assertTrue(r_lo - 1e-9 <= r <= r_hi + 1e-9)
                    self.assertTrue(ANGLE_DEG[0] - 1e-9 <= deg <= ANGLE_DEG[1] + 1e-9)
            for group, gap in ((s.cubes, MIN_CUBE_GAP), (s.trays, MIN_TRAY_GAP)):
                for i, a in enumerate(group):
                    for b in group[i + 1:]:
                        self.assertGreaterEqual(math.hypot(a.x - b.x, a.y - b.y), gap)
            for c in s.cubes:
                for t in s.trays:
                    self.assertGreaterEqual(math.hypot(c.x - t.x, c.y - t.y), MIN_CUBE_TRAY_GAP)

    def test_n_cubes(self):
        self.assertEqual(len(sample_scene(3, n_cubes=3).cubes), 3)
        with self.assertRaises(ValueError):
            sample_scene(3, n_cubes=4)


class TestInTray(unittest.TestCase):
    def test_inside_and_outside(self):
        tray = Placement('red_tray', 'red', 0.27, 0.0, math.radians(45))
        self.assertTrue(in_tray((0.27, 0.0, 0.03), tray))
        # 45° 回ったトレイ: ワールドの x 方向に 4.5 cm は角の内側、(3, 3) cm は辺の外側
        self.assertTrue(in_tray((0.27 + 0.045, 0.0, 0.03), tray))
        self.assertFalse(in_tray((0.27 + 0.03, 0.03, 0.03), tray))

    def test_height(self):
        tray = Placement('red_tray', 'red', 0.27, 0.0, 0.0)
        self.assertFalse(in_tray((0.27, 0.0, 0.025), tray))   # 机の上（トレイの横に落ちた）
        self.assertFalse(in_tray((0.27, 0.0, 0.10), tray))    # まだ掴んで持ち上げている


class TestClassify(unittest.TestCase):
    def setUp(self):
        self.scene = fixed_scene()
        self.before = {'red_cube': pose(0.20, 0.0), 'blue_cube': pose(0.15, 0.10)}

    def classify(self, after, pick=(0.20, 0.0), place=(0.27, 0.10), exec_ok=True):
        return classify_attempt(self.scene, self.before, {**self.before, **after},
                                pick, place, exec_ok)

    def test_success(self):
        r = self.classify({'red_cube': pose(0.27, 0.10, 0.03)})
        self.assertEqual(r['outcome'], 'success')
        self.assertEqual((r['target'], r['tray'], r['collateral']), ('red_cube', 'red_tray', {}))

    def test_perception(self):
        self.assertEqual(self.classify({}, pick=(0.24, -0.05))['outcome'], 'perception')
        self.assertEqual(self.classify({}, place=(0.20, 0.30))['outcome'], 'perception')

    def test_reasoning(self):
        self.assertEqual(self.classify({}, place=(0.27, -0.10))['outcome'], 'reasoning')

    def test_planning_grasp_place(self):
        self.assertEqual(self.classify({}, exec_ok=False)['outcome'], 'planning')
        self.assertEqual(self.classify({'red_cube': pose(0.205, 0.0)})['outcome'], 'grasp')
        self.assertEqual(self.classify({'red_cube': pose(0.26, 0.05)})['outcome'], 'place')

    def test_collateral(self):
        r = self.classify({'red_cube': pose(0.27, 0.10, 0.03), 'blue_cube': pose(0.15, 0.13)})
        self.assertEqual(r['outcome'], 'success')
        self.assertEqual(r['collateral'], {'blue_cube': 0.03})

    def test_cubes_sorted(self):
        after = {'red_cube': pose(0.27, 0.10, 0.03), 'blue_cube': pose(0.15, 0.10)}
        self.assertEqual(cubes_sorted(self.scene, after), {'red_cube': True, 'blue_cube': False})


class TestSummary(unittest.TestCase):
    def test_summary(self):
        scene = fixed_scene().to_dict()
        ok = {'scene': scene, 'success': True, 'sorted': {'red_cube': True, 'blue_cube': True},
              'end_reason': 'agent_done', 'sim_duration': 60.0,
              'attempts': [{'outcome': 'success', 'collateral': []}] * 2}
        ng = {'scene': scene, 'success': False, 'sorted': {'red_cube': True, 'blue_cube': False},
              'end_reason': 'consecutive_failures', 'sim_duration': 90.0,
              'attempts': [{'outcome': 'success', 'collateral': []}]
              + [{'outcome': 'grasp', 'collateral': ['red_cube']}] * 3}
        s = summarize([ok, ng])
        self.assertEqual(s['success_rate'], 0.5)
        self.assertEqual(s['cube_success_rate'], 0.75)
        self.assertEqual(s['outcomes']['grasp'], 3)
        self.assertEqual(s['collateral_attempts'], 3)
        self.assertEqual(s['mean_sim_duration_success'], 60.0)
        self.assertIn('| タスク成功率 | 50.0% |', format_summary(s))


class TestParsePoseInfo(unittest.TestCase):
    def test_fixture(self):
        with open(FIXTURE, encoding='utf-8') as f:
            poses = parse_pose_info(f.read())
        x, y, z, yaw = poses['blue_tray']
        self.assertAlmostEqual(x, 0.0699)
        self.assertAlmostEqual(math.degrees(yaw), 75.0, places=1)
        self.assertEqual(poses['red_tray'][0], 0.0)          # 0 の項目は省略されている
        self.assertAlmostEqual(poses['red_cube'][2], 1.04, places=3)
        self.assertEqual(poses['ground_plane'], (0.0, 0.0, 0.0, 0.0))
        self.assertIn('crane_plus', poses)


if __name__ == '__main__':
    unittest.main()
