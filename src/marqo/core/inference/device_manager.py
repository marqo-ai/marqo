from enum import Enum
from functools import cached_property
from typing import List, Optional

import torch

from marqo.base_model import ImmutableBaseModel
from marqo.core.exceptions import CudaDeviceNotAvailableError, CudaOutOfMemoryError
from marqo.logging import get_logger

logger = get_logger('device_manager')


class DeviceType(str, Enum):
    CPU = 'cpu'
    CUDA = 'cuda'


class Device(ImmutableBaseModel):
    id: int
    name: str
    type: DeviceType
    total_memory: Optional[int] = None

    @property
    def full_name(self) -> str:
        return f'{self.type.value}:{self.id}({self.name})'

    @classmethod
    def cpu(cls) -> 'Device':
        return Device(id=-1, name='cpu', type=DeviceType.CPU)

    @classmethod
    def cuda(cls, device_id, name, total_memory) -> 'Device':
        return Device(id=device_id, name=name, type=DeviceType.CUDA, total_memory=total_memory)


class DeviceManager:
    """
    Device manager collects information and stats of CPU and GPU devices to facilitate the preprocessing and
    vectorisation processes. Based on the information, we will choose the best device to load the embedding models,
    process media files and vectorise the content to achieve optimal performance for search and document ingestion.
    """
    def __init__(self):
        self._is_cuda_available_at_startup: bool = torch.cuda.is_available()
        self.devices: List[Device] = [Device.cpu()]
        self.best_available_device_type = DeviceType.CPU

        if self._is_cuda_available_at_startup:
            self.best_available_device_type = DeviceType.CUDA
            device_count = torch.cuda.device_count()
            for device_id in range(device_count):
                self.devices.append(Device.cuda(device_id,
                                                torch.cuda.get_device_name(device_id),
                                                torch.cuda.get_device_properties(device_id).total_memory))

        logger.debug(f'Found devices {self.devices}. Best available device set to: '
                     f'{self.best_available_device_type.value}.')

    @cached_property
    def cuda_devices(self):
        return [device for device in self.devices if device.type == DeviceType.CUDA]

    def cuda_device_health_check(self) -> None:
        """
        Checks the status of the CUDA devices, and raises exceptions if it becomes
        not available or out of memory.

        raises
          - CudaDeviceNotAvailableError if CUDA device is not available.
          - CudaOutOfMemoryError if any CUDA device is out of memory.
        """
        if not self._is_cuda_available_at_startup:
            # If the instance is initialised without cuda devices, skip the check
            return

        if not torch.cuda.is_available():
            # CUDA devices could become unavailable/unreachable if the docker container running Marqo loses access
            # to the device symlinks. There is no way to recover from this, we will need to restart the container.
            # See https://github.com/NVIDIA/nvidia-container-toolkit/issues/48 for more details.
            raise CudaDeviceNotAvailableError('CUDA device(s) have become unavailable')

        oom_errors = []
        for device in self.cuda_devices:
            try:
                cuda_device = torch.device(f'cuda:{device.id}')
                memory_stats = torch.cuda.memory_stats(cuda_device)
                logger.debug(f'CUDA device {device.full_name} with total memory {device.total_memory}. '
                             f'Memory stats: {str(memory_stats)}')

                torch.randn(3, device=cuda_device)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    logger.error(f'CUDA device {device.full_name} is out of memory. Total memory: {device.total_memory}. '
                                 f'Memory stats: {str(memory_stats)}')
                    allocated_mem = memory_stats.get("allocated.all.current", None) if memory_stats else None
                    oom_errors.append(f'CUDA device {device.full_name} is out of memory:'
                                      f' ({allocated_mem}/{device.total_memory})')
                else:
                    # Log out a warning message when encounter other transient errors.
                    logger.error(f'Encountered issue inspecting CUDA device {device.full_name}: {str(e)}')
            except Exception as e:
                # Log out a warning message when encounter other transient errors.
                logger.error(f'Encountered issue inspecting CUDA device {device.full_name}: {str(e)}')

        if oom_errors:
            # We error out if any CUDA device is out of memory. If this happens consistently, the memory might be held
            # by a long-running thread, and Marqo will need to be restarted to get to a healthy status
            raise CudaOutOfMemoryError(';'.join(oom_errors))
