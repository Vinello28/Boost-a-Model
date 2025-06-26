from argparse import Namespace
import os
from typing import Optional


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


class State:
    def __init__(self) -> None:
        self.is_gui_enabled: Optional[bool] = None
        self.is_cuda_enabled: Optional[bool] = None


@singleton
class Data:
    def __init__(self):
        self.value = 0
        self.bam_root = "/workspace/Boost-a-Model/"
        self.bam_store = os.path.join(self.bam_root, ".store")
        self.bam_config = os.path.join(self.bam_root, ".config")
        self._method = "undefined"
        self.result_path = os.path.join(self.bam_root, self._method, "results")
        self.config_path = ""
        self.cmd_args: Optional[Namespace] = None

        self.state = State()

        # path to reference video (goal video)
        self.goal_path: Optional[str] = ""
        # path to input video (current video)
        self.input_path: Optional[str] = ""

        # Gpu device used, or cpu
        self.device: Optional[str] = ""

        # Path to custom configuration of model used
        self.config_path: Optional[str] = ""

        # Keeps track of how many frames were processed
        self.progress: int = 0
        # Keeps track of the goal frame position
        self.gf_position: int = 0
        # Keeps track of the input frame position
        self.inf_position: int = 0

    def set_method(self, method):
        self._method = method
        self.result_path = os.path.join(self.bam_root, self._method, "results")

    def get_method(self) -> str:
        return self._method
