import unittest
from collections import OrderedDict
from types import SimpleNamespace
from unittest import mock

import torch

from marqo.core.exceptions import CudaDeviceNotAvailableError, CudaOutOfMemoryError, DeviceError
from marqo.inference.native_inference.device_manager import DeviceManager, Device


class TestDeviceManager(unittest.TestCase):

    def _device_manager_without_cuda(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            return DeviceManager()

    def _device_manager_with_cuda(self, total_memory: int = 1_000_000):
        with mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("torch.cuda.device_count", return_value=1), \
             mock.patch("torch.cuda.get_device_name", return_value='Tesla T4'), \
             mock.patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(total_memory=total_memory)):

            return DeviceManager()

    def _device_manager_with_multiple_cuda_devices(self, total_memory: int = 1_000_000):
        with mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("torch.cuda.device_count", return_value=2), \
             mock.patch("torch.cuda.get_device_name", side_effect=['Tesla T4', 'Tesla H200']), \
             mock.patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(total_memory=total_memory)):

            return DeviceManager()

    def _mem_stats(self, allocated: int, reserved: int) -> OrderedDict:
        return OrderedDict({
            "allocated_bytes.all.current": allocated,
            "reserved_bytes.all.current": reserved
        })

    def test_init_with_cpu(self):
        device_manager = self._device_manager_without_cuda()

        self.assertEqual(device_manager._best_available_device, 'cpu')
        self.assertEqual(device_manager.devices, [Device.cpu()])
        self.assertFalse(device_manager._is_cuda_available_at_startup)

    def test_init_with_gpu(self):
        device_manager = self._device_manager_with_cuda(total_memory=1_000_000)

        self.assertEqual(device_manager._best_available_device, 'cuda')
        self.assertEqual(device_manager.devices, [Device.cpu(), Device.cuda(0, 'Tesla T4', 1_000_000)])
        self.assertTrue(device_manager._is_cuda_available_at_startup)

    def test_cuda_health_check_should_skip_without_cuda_devices(self):
        device_manager = self._device_manager_without_cuda()

        with mock.patch("marqo.inference.native_inference.device_manager.torch") as mock_cuda:
            device_manager.cuda_device_health_check()
        self.assertEqual(0, len(mock_cuda.mock_calls))

    def test_cuda_health_check_should_pass_when_cuda_device_is_healthy(self):
        device_manager = self._device_manager_with_cuda()

        with mock.patch("torch.cuda.is_available", return_value=True), \
                mock.patch("torch.randn", return_value=torch.tensor([1, 2, 3])), \
                mock.patch("marqo.inference.native_inference.device_manager.logger") as mock_logger:
            device_manager.cuda_device_health_check()

        # verify there's no warning or error level logging
        for mock_logger_calls in mock_logger.mock_calls:
            logger_call_method_name = mock_logger_calls[0]
            self.assertNotIn(logger_call_method_name, ['warning', 'error'])

    def test_cuda_health_check_should_fail_when_cuda_device_becomes_unavailable(self):
        device_manager = self._device_manager_with_cuda()

        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(CudaDeviceNotAvailableError) as err:
                device_manager.cuda_device_health_check()

            self.assertEqual(str(err.exception), "CUDA device(s) have become unavailable")

    def test_cuda_health_check_should_fail_when_cuda_device_is_out_of_memory(self):
        device_manager = self._device_manager_with_cuda(total_memory=1_000_000)

        with mock.patch("torch.cuda.is_available", return_value=True), \
                mock.patch("torch.randn", side_effect=RuntimeError("CUDA error: out of memory")), \
                mock.patch("torch.cuda.memory_stats", return_value=self._mem_stats(900_000, 950_000)):
            with self.assertRaises(CudaOutOfMemoryError) as err:
                device_manager.cuda_device_health_check()

            self.assertEqual(str(err.exception), "CUDA device cuda:0(Tesla T4) is out of memory "
                                                 "(reserved: 950000, allocated: 900000, total: 1000000)")

    def test_cuda_health_check_should_fail_when_any_cuda_device_is_out_of_memory(self):
        device_manager = self._device_manager_with_multiple_cuda_devices(total_memory=1_000_000)

        with mock.patch("torch.cuda.is_available", return_value=True), \
                mock.patch("torch.randn", side_effect=[torch.tensor([1, 2, 3]), RuntimeError("CUDA error: out of memory")]), \
                mock.patch("torch.cuda.memory_stats", return_value=self._mem_stats(900_000, 950_000)):
            with self.assertRaises(CudaOutOfMemoryError) as err:
                device_manager.cuda_device_health_check()

            self.assertEqual(str(err.exception), "CUDA device cuda:1(Tesla H200) is out of memory "
                                                 "(reserved: 950000, allocated: 900000, total: 1000000)")

    def test_cuda_health_check_should_check_if_all_cuda_devices_are_out_of_memory(self):
        device_manager = self._device_manager_with_multiple_cuda_devices(total_memory=1_000_000)

        with mock.patch("torch.cuda.is_available", return_value=True), \
                mock.patch("torch.randn",
                           side_effect=[RuntimeError("CUDA error: out of memory"), RuntimeError("CUDA error: out of memory")]), \
                mock.patch("torch.cuda.memory_stats", return_value=self._mem_stats(900_000, 950_000)):
            with self.assertRaises(CudaOutOfMemoryError) as err:
                device_manager.cuda_device_health_check()

            self.assertEqual(str(err.exception), "CUDA device cuda:0(Tesla T4) is out of memory "
                                                 "(reserved: 950000, allocated: 900000, total: 1000000);"
                                                 "CUDA device cuda:1(Tesla H200) is out of memory "
                                                 "(reserved: 950000, allocated: 900000, total: 1000000)")

    def test_cuda_health_check_should_pass_and_log_error_message_when_cuda_calls_encounter_issue_other_than_oom(self):
        device_manager = self._device_manager_with_multiple_cuda_devices()

        with mock.patch("torch.cuda.is_available", return_value=True), \
                mock.patch("torch.cuda.memory_stats", side_effect=[RuntimeError("not a memory issue"), Exception("random exception")]), \
                mock.patch("marqo.inference.native_inference.device_manager.logger") as mock_logger:
            device_manager.cuda_device_health_check()

        self.assertEqual('error', mock_logger.mock_calls[0][0])
        self.assertEqual('Encountered issue inspecting CUDA device cuda:0(Tesla T4): not a memory issue',
                         mock_logger.mock_calls[0][1][0])

        self.assertEqual('error', mock_logger.mock_calls[1][0])
        self.assertEqual('Encountered issue inspecting CUDA device cuda:1(Tesla H200): random exception',
                         mock_logger.mock_calls[1][1][0])

    def test_pick_and_validate_device_should_pick_best_device_when_not_provided(self):
        with self.subTest('should pick cuda when available'):
            device_manager = self._device_manager_with_cuda()

            device = device_manager.pick_and_validate_device(None)
            self.assertEqual("cuda", device)

        with self.subTest('should pick cpu when cuda not available'):
            device_manager = self._device_manager_without_cuda()

            device = device_manager.pick_and_validate_device(None)
            self.assertEqual("cpu", device)

    def test_pick_and_validate_device_should_return_device_if_valid(self):
        device_manager = self._device_manager_with_multiple_cuda_devices()

        for device in [
            "cuda:0", "cuda:1", "cuda", "cpu"
        ]:
            with self.subTest(device=device):
                self.assertEqual(device, device_manager.pick_and_validate_device(device))

    def test_pick_and_validate_device_should_raise_device_error_if_not_valid(self):
        device_manager = self._device_manager_with_multiple_cuda_devices()

        for device in [
            "cuda:2", "gpu", "tpu", "dish washer"
        ]:
            with self.subTest(device=device):
                with self.assertRaises(DeviceError) as context:
                    device_manager.pick_and_validate_device(device)
                self.assertEqual(f'`{device}` is not a valid device. Valid devices are [cpu, cuda:0, cuda:1]',
                                 str(context.exception))
