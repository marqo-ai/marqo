import unittest

from marqo.vespa.models.application_metrics import ApplicationMetrics, Node, Service, Status, MetricSet


def _make_metrics(node_metric_sets):
    """
    Build an ApplicationMetrics with a single content node whose service reports the given metric sets.
    node_metric_sets: list of (dimensions_dict, values_dict)
    """
    metrics = [MetricSet(dimensions=dims, values=values) for dims, values in node_metric_sets]
    return ApplicationMetrics(nodes=[
        Node(
            hostname='content-node-0',
            role='content',
            services=[
                Service(
                    name='vespa.searchnode',
                    timestamp=0,
                    status=Status(code='up', description=''),
                    metrics=metrics
                )
            ]
        )
    ])


class TestApplicationMetricsDiskUsage(unittest.TestCase):

    def test_documentDb_diskUsage_bytes_filters_by_documenttype(self):
        metrics = _make_metrics([
            ({'documenttype': 'index_a'}, {'content.proton.documentdb.documentstore.disk_usage.last': 1000}),
            ({'documenttype': 'index_b'}, {'content.proton.documentdb.documentstore.disk_usage.last': 2000}),
        ])

        self.assertEqual(1000, metrics.documentDb_diskUsage_bytes('index_a'))
        self.assertEqual(2000, metrics.documentDb_diskUsage_bytes('index_b'))

    def test_documentDb_diskUsage_bytes_sums_across_nodes(self):
        metric_sets = [
            ({'documenttype': 'index_a'}, {'content.proton.documentdb.documentstore.disk_usage.last': 1000})
        ]
        node_a = _make_metrics(metric_sets).nodes[0]
        node_b = _make_metrics(metric_sets).nodes[0]

        metrics = ApplicationMetrics(nodes=[node_a, node_b])

        self.assertEqual(2000, metrics.documentDb_diskUsage_bytes('index_a'))

    def test_documentDb_diskUsage_bytes_missing_metric_returns_none(self):
        metrics = _make_metrics([
            ({'documenttype': 'index_a'}, {'some.other.metric': 1000}),
        ])

        self.assertIsNone(metrics.documentDb_diskUsage_bytes('index_a'))

    def test_documentDb_diskUsage_bytes_no_matching_documenttype_returns_none(self):
        metrics = _make_metrics([
            ({'documenttype': 'index_a'}, {'content.proton.documentdb.documentstore.disk_usage.last': 1000}),
        ])

        self.assertIsNone(metrics.documentDb_diskUsage_bytes('index_b'))


if __name__ == '__main__':
    unittest.main()
