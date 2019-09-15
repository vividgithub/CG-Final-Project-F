from unittest import TestCase, main
import pointsynthesis.datasetutil as du
import tensorflow as tf
import transforms3d
import random
import numpy as np
import math
import pdb


def random_point_cloud(n, f=3):
    return np.random.random((n, f)).astype(np.float32)


def random_batch_point_cloud(b, n, f=3):
    return np.random.random((b, n, f)).astype(np.float32)


def random_seg_labels(n, c=40):
    return np.random.randint(0, c, n, dtype=np.int64)


def random_cls_labels(c=40):
    return np.random.randint(0, c, 1, dtype=np.int64)


class PointSynthesisTestCase(TestCase):
    def assertArrayEqual(self, a, b, msg=None):
        try:
            return self.assertTrue(np.array_equal(a, b), msg=msg)
        except AssertionError:
            raise

    def assertArrayNearlyEqual(self, a, b, msg=None):
        try:
            return self.assertTrue(np.all(np.less(np.abs(a - b), 1e-5)), msg=msg)
        except AssertionError:
            raise


class DatasetTransformTestCase(PointSynthesisTestCase):

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
            self.assertArrayEqual(points[:, 3:], points_[:, 3:])

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

            self.assertTrue(abs(mean - r[0]) < 1e-6, "mean={}".format(mean))

    def test_rotation_transform(self):
        c = math.cos
        s = math.sin

        for i in range(10):
            points = random_point_cloud(1000, 10)
            label = random_cls_labels()

            rx = random.random() * 10.0 - 5.0
            ry = random.random() * 10.0 - 5.0
            rz = random.random() * 10.0 - 5.0
            r = [(rx, rx), (ry, ry), (rz, rz)]

            matx = np.array([
                [1, 0, 0],
                [0, c(rx), -s(rx)],
                [0, s(rx), c(rx)]
            ])
            maty = np.array([
                [c(ry), 0, s(ry)],
                [0, 1, 0],
                [-s(ry), 0, c(ry)]
            ])
            matz = np.array([
                [c(rz), -s(rz), 0],
                [s(rz), c(rz), 0],
                [0, 0, 1]
            ])
            matyz = np.matmul(matz, maty)
            mat = np.matmul(matyz, matx)

            func = du.transform_rotation("euler", r)

            points_, label_ = func(points, label)

            self.assertArrayEqual(label, label_, "label={}, label_={}".format(label, label_))
            self.assertArrayEqual(points[:, 3:], points_[:, 3:], "points[:, 3:]={}, points_[:, 3:]={}".format(points[:, 3:], points_[:, 3:]))

            points = points[:, :3]
            points_ = points_[:, :3]

            mat_points = np.matmul(mat, points[..., np.newaxis])[..., 0]
            self.assertArrayNearlyEqual(mat_points, points_)

    def test_sampling_transform(self):
        confs = [
            (2048, 1024, 245, (921, 1126)),
            (1024, 768, 384, (652, 883)),
            (512, 458, 458, (412, 503))
        ]

        for n, sample_num, stddev, range in confs:
            points = random_batch_point_cloud(10, n, 10)
            label = random_cls_labels()

            func = du.transform_sampling(sample_num, "random-gauss", range=range, stddev=stddev)
            points_, label_ = func(points, label)

            self.assertArrayEqual(label, label_)

            self.assertTrue(points.shape[0] == points_.shape[0])
            self.assertTrue(points.shape[2] == points_.shape[2])
            print("points={}, points_{}".format(points.shape, points_.shape))

            n_ = points.shape[1]

            self.assertTrue(range[0] <= n_ <= n + range[1])
            self.assertTrue(n_ <= n)

            for p, p_ in zip(points, points_):
                hashes = set([np.sum(x) for x in p])
                hashes_ = set([np.sum(x_) for x_ in p_])
                print("hashes.size={}, hashes={}".format(len(hashes), hashes))
                print("hashes_.size={}, hashes_={}".format(len(hashes_), hashes_))
                self.assertTrue(hashes.issuperset(hashes_))


if __name__ == "__main__":
    main()