from abc import ABC, abstractmethod
import numpy as np

class BaseTracker(ABC):
    """
    Abstract Base Class that enforces a unified interface for all tracking models
    benchmarked in this framework (SAM-3, DEVA, YOLOv11-World).
    """
    def __init__(self, device="cpu", use_mock=False):
        self.device = device
        self.use_mock = use_mock
        self.model = None
        self.active_tracks = {} # Track ID mapping / persistent states

    @abstractmethod
    def initialize(self, prompt="apple"):
        """
        Initializes the model with the target class/prompt (e.g. "apple" or "fruit").
        Loads weights, applies VRAM configs, and sets up state.
        """
        pass

    @abstractmethod
    def process_frame(self, image):
        """
        Processes a single frame.
        Input:
            image: numpy array (BGR format, shape [H, W, 3])
        Returns:
            list of dicts, where each dict represents a tracked apple:
            {
                "id": int (unique persistent ID),
                "box": [ymin, xmin, ymax, xmax] (normalized or pixel coordinates),
                "mask": numpy binary mask array (shape [H, W], boolean/uint8 0/1) or None,
                "score": float (confidence score)
            }
        """
        pass

    @abstractmethod
    def clear_memory(self):
        """
        Resets/truncates the model's internal memory bank (e.g., SAM-3's memory keys,
        DEVA's spatial-temporal association frames, or Kalman filters in BoT-SORT).
        Crucial for mitigating long-term VRAM build-up and testing re-identification.
        """
        pass
