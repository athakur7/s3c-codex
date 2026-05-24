from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(slots=True)
class S3CConfig:
    """Global configuration for baseline and S3C pipelines."""

    data_dir: str
    images_subdir: str
    masks_subdir: str
    num_images: str
    seed: int | None
    reduction: float
    shadow_mode: Literal["gt", "auto"]
    methods: tuple[str, ...]
    alpha: float = 1.0
    beta: float = 2.0
    gamma: float = 2.5
    delta: float = 2.0
    p_shadow: float = 10.0
    p_struct: float = 10.0
    tau_struct: float = 0.5
    lambda_sh: float = 1.0
    sigma_tensor: float = 1.5
    w_line: float = 1.5
    tau_r: float = 0.65
    tau_c: float = 0.12
    local_window: int = 51
    bilateral_window: int = 5
    bilateral_sigma_spatial: float = 1.5
    bilateral_sigma_range: float = 15.0
    output_root: str = "./outputs"
    workers: int = 4
    executor: Literal["serial", "thread", "process"] = "process"
    opencv_threads_per_worker: int = 1
    verbose: bool = False

    def to_dict(self) -> dict:
        return asdict(self)
