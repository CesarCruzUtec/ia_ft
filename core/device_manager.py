"""
Device manager for selecting and configuring compute devices (CUDA, MPS, CPU).
"""

import torch


class DeviceManager:
    """
    Singleton manager for compute device selection and configuration.
    Handles GPU (CUDA/MPS) and CPU device setup with appropriate optimizations.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not DeviceManager._initialized:
            self._device = self._select_device()
            self._configure_device()
            DeviceManager._initialized = True

    def _select_device(self) -> torch.device:
        """
        Select the best available compute device.

        Priority: CUDA > MPS > CPU

        Returns:
            torch.device: Selected device
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"✓ Using CUDA device: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("✓ Using MPS (Apple Silicon) device")
            print(
                "⚠ Note: MPS support is preliminary. SAM2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS."
            )
        else:
            device = torch.device("cpu")
            print("✓ Using CPU device")

        return device

    def _configure_device(self):
        """Configure device-specific optimizations."""
        if self._device.type == "cuda":
            # Enable autocast for CUDA
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

            # Enable TF32 for Ampere and newer GPUs
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("✓ TF32 optimizations enabled")

    @property
    def device(self) -> torch.device:
        """Get the selected device."""
        return self._device

    def is_cuda(self) -> bool:
        """Check if CUDA is being used."""
        return self._device.type == "cuda"

    def is_mps(self) -> bool:
        """Check if MPS is being used."""
        return self._device.type == "mps"

    def is_cpu(self) -> bool:
        """Check if CPU is being used."""
        return self._device.type == "cpu"

    def clear_cache(self):
        """Clear GPU memory cache if available."""
        if self.is_cuda():
            torch.cuda.empty_cache()
            print("✓ CUDA cache cleared")
