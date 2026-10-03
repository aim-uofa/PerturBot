"""Loopback protocol smoke test; no model weights or robot hardware."""

import asyncio

import numpy as np
from openpi_client import base_policy
from openpi_client import websocket_client_policy
from typing_extensions import override
import websockets.asyncio.server

from openpi.serving import websocket_policy_server


class DummyPolicy(base_policy.BasePolicy):
    @override
    def infer(self, obs):
        assert obs["observation/state"].shape == (14,)
        return {"actions": np.zeros((10, 14), dtype=np.float32)}


def test_websocket_round_trip_and_health_check():
    async def check():
        policy_server = websocket_policy_server.WebsocketPolicyServer(DummyPolicy(), metadata={"test_only": True})
        async with websockets.asyncio.server.serve(
            policy_server._handler,  # noqa: SLF001
            "127.0.0.1",
            0,
            process_request=websocket_policy_server._health_check,  # noqa: SLF001
            compression=None,
        ) as server:
            port = server.sockets[0].getsockname()[1]

            def infer():
                client = websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=port)
                try:
                    return client.infer({"observation/state": np.zeros(14, dtype=np.float32)})
                finally:
                    client._ws.close()  # noqa: SLF001

            result = await asyncio.to_thread(infer)
            assert result["actions"].shape == (10, 14)
            assert np.isfinite(result["actions"]).all()
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(b"GET /healthz HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            await writer.drain()
            response = await asyncio.wait_for(reader.read(), timeout=5)
            writer.close()
            await writer.wait_closed()
            assert b"200 OK" in response
            assert response.endswith(b"OK\n")

    asyncio.run(check())
