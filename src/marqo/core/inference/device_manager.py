from enum import Enum
from typing import List

import psutil
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
    total_memory: int

    @classmethod
    def cpu(cls) -> 'Device':
        return Device(id=-1, name='cpu', type=DeviceType.cpu, total_memory=psutil.virtual_memory().total)

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
        if not self._is_cuda_available_at_startup:
            return

        if not torch.cuda.is_available():
            logger.error('Cuda device becomes unavailable.')
            raise CudaDeviceNotAvailableError('Cuda device becomes unavailable.')

        for device in self.cuda_devices:
            cuda_device = torch.device(device.name)
            memory_stats = torch.cuda.memory_stats(cuda_device)
            logger.debug(f'Cuda device {device.name} with total memory {device.total_memory}. '
                         f'Memory stats: {memory_stats}')
            
            try:
                torch.randn(3).to(cuda_device)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    logger.error(f'Cuda device {device.name} is out of memory. Total memory: {device.total_memory}. '
                                 f'Memory stats: {memory_stats}')
                    raise CudaOutOfMemoryError(f'Cuda device {device.name} is out of memory. '
                                               f'({memory_stats["allocated.all.current"]}/{device.total_memory})')
