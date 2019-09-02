from unittest import TestCase, main
import pointsynthesis.datasetutil as du
import tensorflow as tf
import random
import numpy as np


def random_point_cloud(n, f=3):
    return np.random.random((n, f))


def random_batch_point_cloud(b, n, f=3):
    return np.random.uniform((b, n, f))


class DatasetTransformTestCase(TestCase):
    def test_clip_feature_transform(self):
        for c in [3, 4, 6, 10]:
            func = du.transform_clip_feature(c=c)

            points = random_point_cloud(1000, 10)
            label = np.random.randint(0, 40, 1000)

            points_, label_ = func(points, label)
            self.assertTrue(np.array_equal(label, label_))
            self.assertTrue(np.array_equal(points_, points[:, :c]))

    def test_scaling_transform(self):
        for i in range(5):
            range1 = random.random() * 2.0 - 1
            range2 = random.random() * 2.0 - 1
            r = (min(range1, range2), max(range1, range2))

            func = du.transform_scaling(r)
            points = random_point_cloud(1000, 10)
            label = np.random.randint(0, 40, 1000)

            points_, label_ = func(points, label)

            self.assertTrue(np.array_equal(label, label_))
            self.assertTrue(points_.shape == points.shape)

            points = points[:, :3]
            points_ = points_[:, :3]

            ratio = points_ / points
            mean = np.mean(ratio)

            re_ratio = np.abs(ratio - mean)
            self.assertTrue(
                np.all(np.less_equal(re_ratio, 1e-6)),
                "mean={}, ratio={}, re_ratio={}".format(mean, ratio, re_ratio)
            )

            self.assertTrue(r[0] <= mean <= r[1])

        for i in range(5):
            r = random.random() * 2.0 - 1
            r = (r, r)

            func = du.transform_scaling(r)
            points = random_point_cloud(1000, 10)
            label = np.random.randint(0, 40, 1000)

            points_, label_ = func(points, label)

            self.assertTrue(np.array_equal(label, label_))
            self.assertTrue(points_.shape == points.shape)

            points = points[:, :3]
            points_ = points_[:, :3]

            ratio = points_ / points
            mean = np.mean(ratio)

            self.assertTrue(abs(mean - r[0]) < 1e-7)


if __name__ == "__main__":
    main()