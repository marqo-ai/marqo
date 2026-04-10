from enum import StrEnum
from typing import Optional

from inference_orchestrator.schemas.base_model import AppBaseModel


class GRPCCompressionAlgorithm(StrEnum):
    GZIP = "gzip"
    DEFLATE = "deflate"


class TritonChannelArgs(AppBaseModel):
    """
    A class to hold the arguments for a channel in Triton Inference Server.

    Default values are set based on Triton Inference Server recommendations.
    https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/_reference/tritonclient/tritonclient.grpc.html#tritonclient.grpc.KeepAliveOptions

    Attributes:
        grpc_keep_alive_time_ms (int): The time (in milliseconds) after which a keepalive ping is sent on the transport.
        grpc_keep_alive_timeout_ms (int): The time (in milliseconds) the sender of the
            keepalive ping waits for an acknowledgment.
        grpc_keep_alive_permit_without_calls (int): If true, allows keepalive pings
            to be sent even if there are no calls in flight.
        grpc_http2_max_pings_without_data (int): The maximum number of pings that
            can be sent when there is no data/header frame to be sent.
        grpc_http2_min_time_between_pings_ms (int): The minimum time (in milliseconds)
            between successive pings without data/header frames.
        grpc_http2_min_ping_interval_without_data_ms (int): The minimum time (in milliseconds)
            between successive pings without data/header frames.
        grpc_max_receive_message_length (int): The maximum message size (in bytes)
            that the client can receive.
        grpc_max_send_message_length (int): The maximum message size (in bytes)
            that the client can send.
        grpc_initial_reconnect_backoff_ms (int): The initial backoff time (in milliseconds)
            for the first reconnect attempt.
        grpc_max_reconnect_backoff_ms (int): The maximum backoff time (in milliseconds)
            for reconnect attempts.
        grpc_enable_retries (int): If true, enables automatic retries on failed
            streams.
    Methods:
        build_channel_args: Builds the channel arguments for Triton Inference Server.
    """

    grpc_keep_alive_time_ms: int = 300_000
    grpc_keep_alive_timeout_ms: int = 20_000
    grpc_keep_alive_permit_without_calls: int = 0
    grpc_http2_max_pings_without_data: int = 2
    grpc_http2_min_time_between_pings_ms: int = 10_000
    grpc_http2_min_ping_interval_without_data_ms: int = 10_000
    grpc_max_receive_message_length: int = 128 * 1024 * 1024
    grpc_max_send_message_length: int = 128 * 1024 * 1024
    grpc_initial_reconnect_backoff_ms: int = 500
    grpc_max_reconnect_backoff_ms: int = 10_000
    grpc_enable_retries: int = 1
    grpc_compression_algorithm: Optional[GRPCCompressionAlgorithm] = None

    def build_channel_args(self) -> list[tuple[str, int]]:
        """
        Build the channel arguments for Triton Inference Server.

        Returns:
            list[tuple[str, int]]: A list of tuples containing the channel argument names and their values.
        """
        return [
            ("grpc.keepalive_time_ms", self.grpc_keep_alive_time_ms),
            ("grpc.keepalive_timeout_ms", self.grpc_keep_alive_timeout_ms),
            (
                "grpc.keepalive_permit_without_calls",
                self.grpc_keep_alive_permit_without_calls,
            ),
            (
                "grpc.http2.max_pings_without_data",
                self.grpc_http2_max_pings_without_data,
            ),
            (
                "grpc.http2.min_time_between_pings_ms",
                self.grpc_http2_min_time_between_pings_ms,
            ),
            (
                "grpc.http2.min_ping_interval_without_data_ms",
                self.grpc_http2_min_ping_interval_without_data_ms,
            ),
            ("grpc.max_receive_message_length", self.grpc_max_receive_message_length),
            ("grpc.max_send_message_length", self.grpc_max_send_message_length),
            (
                "grpc.initial_reconnect_backoff_ms",
                self.grpc_initial_reconnect_backoff_ms,
            ),
            ("grpc.max_reconnect_backoff_ms", self.grpc_max_reconnect_backoff_ms),
            ("grpc.enable_retries", self.grpc_enable_retries),
        ]
