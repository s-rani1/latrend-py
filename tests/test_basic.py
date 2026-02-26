import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import latrend as lt  # noqa: E402

try:
    import sklearn  # noqa: F401
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

SKIP_MSG = "scikit-learn not installed"


class TestDataGeneration(unittest.TestCase):
    def test_generate_long_data_shape(self):
        df = lt.generateLongData(nIndividuals=10, nTime=5, nClusters=3, seed=1)
        self.assertEqual(set(["Id", "Time", "Y", "Cluster"]), set(df.columns))
        self.assertEqual(len(df), 10 * 5)

    def test_latrendData(self):
        df = lt.latrendData()
        self.assertIn("Id", df.columns)
        self.assertIn("Time", df.columns)
        self.assertIn("Y", df.columns)
        self.assertIn("Class", df.columns)
        self.assertEqual(df["Id"].nunique(), 200)
        self.assertEqual(len(df), 200 * 10)
        self.assertEqual(set(df["Class"].unique()), {1, 2, 3})

    def test_latrendData_reproducible(self):
        df1 = lt.latrendData(seed=42)
        df2 = lt.latrendData(seed=42)
        self.assertTrue(df1.equals(df2))


class TestFormatConversions(unittest.TestCase):
    def test_tsmatrix_tsframe_roundtrip(self):
        df = lt.generateLongData(nIndividuals=10, nTime=5, nClusters=2, seed=1)[["Id", "Time", "Y"]]
        mat = lt.tsmatrix(df)
        self.assertEqual(mat.shape, (10, 5))
        df2 = lt.tsframe(mat)
        self.assertTrue(set(["Id", "Time", "Y"]).issubset(df2.columns))


@unittest.skipUnless(HAS_SKLEARN, SKIP_MSG)
class TestClustering(unittest.TestCase):
    def test_lmkm_clusters(self):
        df = lt.generateLongData(nIndividuals=30, nTime=6, nClusters=3, seed=1)
        method = lt.lcMethodLMKM(formula="Y ~ Time", nClusters=3, seed=1)
        model = lt.latrendCluster(method, df)
        self.assertEqual(model.nClusters(), 3)
        self.assertEqual(len(model.clusters), 30)

    def test_features_clusters(self):
        df = lt.generateLongData(nIndividuals=30, nTime=6, nClusters=3, seed=1)
        method = lt.lcMethodFeatures(nClusters=3, seed=1)
        model = lt.latrendCluster(method, df)
        self.assertEqual(model.nClusters(), 3)
        self.assertEqual(len(model.clusters), 30)

    def test_random_clusters(self):
        df = lt.generateLongData(nIndividuals=20, nTime=5, nClusters=2, seed=1)
        method = lt.lcMethodRandom(nClusters=2, seed=1)
        model = lt.latrendCluster(method, df)
        self.assertEqual(model.nClusters(), 2)

    def test_rep_cluster(self):
        df = lt.generateLongData(nIndividuals=30, nTime=6, nClusters=3, seed=1)
        method = lt.lcMethodRandom(nClusters=3)
        models = lt.latrendRepCluster(method, df, nRep=3)
        self.assertEqual(len(models), 3)

    def test_batch_cluster_sweep_k(self):
        df = lt.generateLongData(nIndividuals=30, nTime=6, nClusters=3, seed=1)
        method = lt.lcMethodLMKM(formula="Y ~ Time", nClusters=2, seed=1)
        models = lt.latrendBatchCluster(method, df, nClusters=[2, 3, 4])
        self.assertEqual(len(models), 3)


@unittest.skipUnless(HAS_SKLEARN, SKIP_MSG)
class TestModel(unittest.TestCase):
    def setUp(self):
        df = lt.generateLongData(nIndividuals=30, nTime=6, nClusters=3, seed=1)
        method = lt.lcMethodLMKM(formula="Y ~ Time", nClusters=3, seed=1)
        self.model = lt.latrendCluster(method, df)

    def test_class_counts(self):
        counts = self.model.classCounts()
        self.assertEqual(counts.sum(), 30)

    def test_class_proportions(self):
        props = self.model.classProportions()
        self.assertAlmostEqual(props.sum(), 1.0)

    def test_class_entropy(self):
        ent = self.model.classEntropy()
        self.assertAlmostEqual(ent, 0.0, places=5)

    def test_postprob(self):
        self.assertIsNotNone(self.model.postprob)
        self.assertEqual(self.model.postprob.shape[0], 30)
        self.assertEqual(self.model.postprob.shape[1], 3)


@unittest.skipUnless(HAS_SKLEARN, SKIP_MSG)
class TestPlotsMpl(unittest.TestCase):
    """Test plotting with matplotlib backend (always available)."""

    def setUp(self):
        df = lt.generateLongData(nIndividuals=20, nTime=6, nClusters=3, seed=1)
        method = lt.lcMethodLMKM(formula="Y ~ Time", nClusters=3, seed=1)
        self.model = lt.latrendCluster(method, df)
        self.df = df

    def test_plotTrajectories_df(self):
        ax = lt.plotTrajectories(self.df, backend="matplotlib")
        self.assertIsNotNone(ax)

    def test_plotTrajectories_model(self):
        ax = lt.plotTrajectories(self.model, backend="matplotlib")
        self.assertIsNotNone(ax)

    def test_plotClusterTrajectories(self):
        ax = lt.plotClusterTrajectories(self.model, backend="matplotlib")
        self.assertIsNotNone(ax)

    def test_plotClusterTrajectories_ci(self):
        ax = lt.plotClusterTrajectories(self.model, ci=True, backend="matplotlib")
        self.assertIsNotNone(ax)

    def test_plotClusterTrajectories_with_traj_overlay(self):
        ax = lt.plotClusterTrajectories(
            self.model, trajectories=True, backend="matplotlib"
        )
        self.assertIsNotNone(ax)

    def test_plotClassProportions(self):
        ax = lt.plotClassProportions(self.model, backend="matplotlib")
        self.assertIsNotNone(ax)

    def test_plotClassProbabilities(self):
        ax = lt.plotClassProbabilities(self.model, backend="matplotlib")
        self.assertIsNotNone(ax)

    def test_plotMetric(self):
        df = lt.generateLongData(nIndividuals=20, nTime=6, nClusters=3, seed=1)
        method = lt.lcMethodRandom(nClusters=2, seed=1)
        models = lt.latrendBatchCluster(method, df, nClusters=[2, 3])
        ax = lt.plotMetric(models, backend="matplotlib")
        self.assertIsNotNone(ax)

    def test_plotMetric_multi(self):
        df = lt.generateLongData(nIndividuals=20, nTime=6, nClusters=3, seed=1)
        method = lt.lcMethodRandom(nClusters=2, seed=1)
        models = lt.latrendBatchCluster(method, df, nClusters=[2, 3])
        ax = lt.plotMetric(models, metric=["BIC", "WMAE"], backend="matplotlib")
        self.assertIsNotNone(ax)


class TestPlotsNoModel(unittest.TestCase):
    """Plot tests that work without sklearn (using raw DataFrames)."""

    def test_plotTrajectories_raw_df(self):
        df = lt.latrendData()
        ax = lt.plotTrajectories(df, backend="matplotlib")
        self.assertIsNotNone(ax)

    def test_plotClusterTrajectories_raw_df(self):
        df = lt.latrendData()
        ax = lt.plotClusterTrajectories(df, cluster="Class", backend="matplotlib")
        self.assertIsNotNone(ax)

    def test_plotClusterTrajectories_raw_df_ci(self):
        df = lt.latrendData()
        ax = lt.plotClusterTrajectories(df, cluster="Class", ci=True, backend="matplotlib")
        self.assertIsNotNone(ax)

    def test_plotClusterTrajectories_raw_df_traj_overlay(self):
        df = lt.latrendData()
        ax = lt.plotClusterTrajectories(
            df, cluster="Class", trajectories=True, backend="matplotlib"
        )
        self.assertIsNotNone(ax)


class TestRMethodConstructor(unittest.TestCase):
    def test_dynamic_r_method_constructor(self):
        method = lt.lcMethodLcmmGMM(formula="Y ~ Time", nClusters=2)
        self.assertTrue(hasattr(method, "r_method"))
        self.assertEqual(method.r_method, "lcMethodLcmmGMM")


@unittest.skipUnless(HAS_SKLEARN, SKIP_MSG)
class TestReport(unittest.TestCase):
    def test_model_report(self):
        df = lt.generateLongData(nIndividuals=20, nTime=6, nClusters=3, seed=1)
        method = lt.lcMethodLMKM(formula="Y ~ Time", nClusters=3, seed=1)
        model = lt.latrendCluster(method, df)
        with tempfile.TemporaryDirectory() as d:
            report_path = lt.lcModelReport(model, d)
            self.assertTrue(report_path.exists())
            self.assertTrue((pathlib.Path(d) / "cluster_trajectories.png").exists())
            self.assertTrue((pathlib.Path(d) / "class_proportions.png").exists())
            self.assertTrue((pathlib.Path(d) / "class_probabilities.png").exists())


@unittest.skipUnless(HAS_SKLEARN, SKIP_MSG)
class TestClusterLabelers(unittest.TestCase):
    def setUp(self):
        df = lt.generateLongData(nIndividuals=20, nTime=6, nClusters=3, seed=1)
        method = lt.lcMethodLMKM(formula="Y ~ Time", nClusters=3, seed=1)
        self.model = lt.latrendCluster(method, df)

    def test_make_clusterPropLabels(self):
        labels = lt.make_clusterPropLabels(self.model)
        self.assertEqual(len(labels), 3)
        for v in labels.values():
            self.assertIn("%", v)

    def test_make_clusterSizeLabels(self):
        labels = lt.make_clusterSizeLabels(self.model)
        self.assertEqual(len(labels), 3)
        for v in labels.values():
            self.assertIn("n=", v)


class TestPalette(unittest.TestCase):
    def test_palette_exists(self):
        self.assertEqual(len(lt.LATREND_PALETTE), 9)
        for c in lt.LATREND_PALETTE:
            self.assertTrue(c.startswith("#"))


if __name__ == "__main__":
    unittest.main()
