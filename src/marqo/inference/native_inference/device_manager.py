from enum import Enum
from functools import cached_property
from typing import List, Optional

import torch

from marqo.base_model import ImmutableBaseModel
from marqo.core.exceptions import CudaDeviceNotAvailableError, CudaOutOfMemoryError, DeviceError
from marqo.logging import get_logger

logger = get_logger(__name__)


class DeviceType(str, Enum):
    CPU = 'cpu'
    CUDA = 'cuda'


class Device(ImmutableBaseModel):
    id: int
    name: str
    type: DeviceType
    total_memory: Optional[int] = None

    def __repr__(self) -> str:
        if self.type == DeviceType.CPU:
            return self.type.value
        else:
            return f'{self.type.value}:{self.id}'

    def __str__(self) -> str:
        if self.type == DeviceType.CPU:
            return self.type.value
        else:
            return f'{self.type.value}:{self.id}({self.name})'

    def matches(self, device: str):
        return device == self.type.value or device == f'{self.type.value}:{self.id}'

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
        self._best_available_device = DeviceType.CPU.value

        if self._is_cuda_available_at_startup:
            self._best_available_device = DeviceType.CUDA.value
            device_count = torch.cuda.device_count()
            for device_id in range(device_count):
                self.devices.append(Device.cuda(device_id,
                                                torch.cuda.get_device_name(device_id),
                                                torch.cuda.get_device_properties(device_id).total_memory))

        logger.info(f'Found devices {self.devices}. Best available device set to: {self._best_available_device}.')

    def pick_and_validate_device(self, device: Optional[str] = None) -> str:
        if device is None:
            logger.debug(f'Device is not provided, pick the default device `{self._best_available_device}`')
            return self._best_available_device

        # otherwise, check if the device passed in is valid
        if any([d.matches(device) for d in self.devices]):
            logger.debug(f'Device {device} matches one of the devices: {self.devices}')
            return device

        raise DeviceError(f'`{device}` is not a valid device. Valid devices are {self.devices}')

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
            memory_stats = None
            try:
                cuda_device = torch.device(f'cuda:{device.id}')
                memory_stats = torch.cuda.memory_stats(cuda_device)
                logger.debug(f'CUDA device {device} with total memory {device.total_memory}. '
                             f'Memory stats: {str(memory_stats)}')

                # Marqo usually allocates 20MiB cuda memory at a time when processing media files. When OOM happens,
                # there are usually a few MiB free in reserved memory. To consistently trigger OOM from this health
                # check request, we try to create a tensor that won't fit in the reserved memory to always trigger
                # another allocation. Please note this is not leaky since we don't store the reference to this tensor.
                # Once it's out of scope, the memory it takes will be returned to reserved space. Our assumption is
                # that this method will not be called too often or with high concurrency (only used by liveness check
                # for now which run once every few seconds).
                tensor_size = int(20 * 1024 * 1024 / 4)  # 20MiB, float32 (4 bytes)
                torch.randn(tensor_size, device=cuda_device)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    logger.error(f'CUDA device {device} is out of memory. Original error: {str(e)}. '
                                 f'Memory stats: {str(memory_stats)}.')
                    allocated_mem = memory_stats.get("allocated_bytes.all.current", None) if memory_stats else None
                    reserved_mem = memory_stats.get("reserved_bytes.all.current", None) if memory_stats else None
                    oom_errors.append(f'CUDA device {device} is out of memory (reserved: {reserved_mem}, '
                                      f'allocated: {allocated_mem}, total: {device.total_memory})')
                else:
                    # Log out a warning message when encounter other transient errors.
                    logger.error(f'Encountered issue inspecting CUDA device {device}: {str(e)}')
            except Exception as e:
                # Log out a warning message when encounter other transient errors.
                logger.error(f'Encountered issue inspecting CUDA device {device}: {str(e)}')

        if oom_errors:
            # We error out if any CUDA device is out of memory. If this happens consistently, the memory might be held
            # by a long-running thread, and Marqo will need to be restarted to get to a healthy status
            raise CudaOutOfMemoryError(';'.join(oom_errors))
