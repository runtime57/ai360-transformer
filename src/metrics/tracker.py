from collections import defaultdict


class MetricTracker:
    """
    Class to aggregate metrics from many batches.
    """

    def __init__(self, *keys, writer=None):
        """
        Args:
            *keys (list[str]): list (as positional arguments) of metric
                names (may include the names of losses)
            writer (WandBWriter | CometMLWriter | None): experiment tracker.
                Not used in this code version. Can be used to log metrics
                from each batch.
        """
        self.writer = writer
        self._data = defaultdict(lambda: {"total": 0.0, "count": 0.0})

        for key in keys:
            _ = self._data[key]

    def reset(self):
        for key in self._data.keys():
            self._data[key]["total"] = 0.0
            self._data[key]["count"] = 0.0

    def update(self, key, value, n=1):
        self.update_sum(key, value, n)

    def update_sum(self, key, sum_value, n=1):
        metric_data = self._data[key]
        metric_data['total'] += sum_value
        metric_data['count'] += n

    def update_avg(self, key, avg_value, n=1):
        self.update_sum(key, avg_value * n, n)

    def count(self, key):
        return self._data.get(key, {'count': 0})['count']

    def avg(self, key):
        metric_data = self._data[key]
        return metric_data['total'] / metric_data['count'] if metric_data['count'] > 0 else 0.0

    def result(self):
        return {key: self.avg(key) for key in self._data.keys()}

    def keys(self):
        return self._data.keys()
