from enum import Enum
from typing import List, Optional

import torch
from pydantic import BaseModel

from marqo.core.exceptions import CudaDeviceNotAvailableError, CudaOutOfMemoryError
from marqo.logging import get_logger

logger = get_logger('device_manager')


class DeviceType(str, Enum):
    cpu = 'cpu'
    cuda = 'cuda'


class Device(BaseModel):
    id: int
    name: str
    type: DeviceType
    total_memory: Optional[int] = 0

    @classmethod
    def cpu(cls) -> 'Device':
        return Device(id=-1, name='cpu', type=DeviceType.cpu)

    @classmethod
    def cuda(cls, device_id, name, total_memory) -> 'Device':
        return Device(id=device_id, name=name, type=DeviceType.cuda, total_memory=total_memory)


class DeviceManager:
    def __init__(self):
        self._is_cuda_available_at_startup: bool = torch.cuda.is_available()
        self.devices: List[Device] = [Device.cpu()]
        self.best_available_device_type = DeviceType.cpu

        if self._is_cuda_available_at_startup:
            self.best_available_device_type = DeviceType.cuda
            device_count = torch.cuda.device_count()
            for device_id in range(device_count):
                self.devices.append(Device.cuda(device_id,
                                                torch.cuda.get_device_name(device_id),
                                                torch.cuda.get_device_properties(device_id).total_memory))

        logger.debug(f'Found devices {self.devices}. Best available device set to: '
                     f'{self.best_available_device_type.value}.')

    @property
    def cuda_devices(self):
        return [device for device in self.devices if device.type == DeviceType.cuda]

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
            logger.error('Cuda device becomes unavailable')
            raise CudaDeviceNotAvailableError('Cuda device becomes unavailable')

        # TODO confirm whether we should check all devices or just the default one
        for device in self.cuda_devices:
            try:
                cuda_device = torch.device(device.name)
                memory_stats = torch.cuda.memory_stats(cuda_device)
                logger.debug(f'Cuda device {device.name} with total memory {device.total_memory}. '
                             f'Memory stats: {str(memory_stats)}')

                torch.randn(3, device=cuda_device)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    # If we encounter 'CUDA error: out of memory' error consistently, it means some threads are
                    # holding the memory
                    logger.error(f'Cuda device {device.name} is out of memory. Total memory: {device.total_memory}. '
                                 f'Memory stats: {str(memory_stats)}')
                    allocated_mem = memory_stats.get("allocated.all.current", None) if memory_stats else None
                    raise CudaOutOfMemoryError(f'Cuda device {device.name} is out of memory: '
                                               f'({allocated_mem}/{device.total_memory})')
            except Exception as e:
                # Log out a warning message when encounter other transient errors.
                logger.warning(f'Encountered issue inspecting Cuda device {device.name}: {str(e)}')
