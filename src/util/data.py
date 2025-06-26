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
        self.bam_root = os.getcwd()
        self.bam_store = os.path.join(self.bam_root, ".store")
        self.bam_config = os.path.join(self.bam_root, ".config")
        self._method = "undefined"
        self.result_path = os.path.join(self.bam_root, "results")
        self.config_path = ""
        self.cmd_args: Optional[Namespace] = None

        self.state = State()

        self.goal_path: Optional[str] = ""
        self.input_path: Optional[str] = ""
        self.device: Optional[str] = ""
        self.config_path: Optional[str] = ""

        self.progress: int = 0
        self.gf_position: int = 0
        self.inf_position: int = 0

    def set_method(self, method):
        self._method = method

    def get_method(self) -> str:
        return self._method
