class Scaler:
    def __init__(
        self, min_value: float, max_value: float, num_buckets: int, eps: float = 1e-6,
    ):
        self.min_value = min_value
        self.max_value = max_value
        self.num_buckets = num_buckets
        self.eps = eps
        self.bucket_size = (max_value - min_value) / num_buckets

    def scale(self, value: float):
        assert (
            self.min_value <= value <= self.max_value
        ), f"Value ${value} is not in expected range [{self.min_value}, {self.max_value}]"
        return int((value - self.min_value) // self.bucket_size)
