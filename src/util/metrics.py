import os
import json

from typing import Iterator, List, Optional

from numpy.typing import NDArray
from util.data import Data


class MetricPoint:
    def __init__(self) -> None:
        # NOTE: velocity is NOT correctly typed here
        self.velocity_norm: Optional[float] = None
        self.velocity_vector: Optional[NDArray] = None
        self.features_num: Optional[int] = None
        self.cosine_similarity: Optional[float] = None
        self.rotation_error: Optional[float] = None
        self.inference_time_ms: float = 0

        # i didn't want anymore losing time so here it is
        self.extra = {}


class Metrics:
    def __init__(self, model_name: str, configuration) -> None:
        self.model_name: str = model_name
        self._metrics: List[MetricPoint] = []
        self.data = Data()
        self.input_size = 0
        self._metric_size: int = 0
        self.configuration = configuration

    def save(self, path: str):
        """Save the collected metrics to a JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        buff = {
            "version": "1.0",
            "model_name": self.model_name,
            "input_size": self.input_size,
            "record_count": self._metric_size,
            "metrics": [],
            "configuration": {},
        }
        for m in self._metrics:
            metric_json = {
                "volicity_vector": m.velocity_vector,
                "velocity_norm": m.velocity_norm,
                "features_num": m.features_num,
                "cosine_similarity": m.cosine_similarity,
                "rotation_error": m.rotation_error,
                "inference_time_ms": m.inference_time_ms,
            }
            buff["metrics"].append(metric_json)

        with open(path, "w") as f:
            json.dump(buff, f, indent=4)

    def add(self, mp: MetricPoint):
        self._metric_size += 1
        self._metrics.append(mp)

    # probably never used
    def remove(self, mp: MetricPoint):
        self._metric_size -= 1
        self._metrics.remove(mp)
