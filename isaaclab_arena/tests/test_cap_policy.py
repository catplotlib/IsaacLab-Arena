# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CAP protocol framing and per-channel startup safety."""

import importlib.util

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_cap_client(_simulation_app):
    import numpy as np
    import socket
    import struct
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    import msgpack
    import msgpack_numpy

    from isaaclab_arena_environments.isaac_cap.cap_policy import CapPolicy, CapPolicyCfg

    client = CapPolicy(CapPolicyCfg())
    left, right = socket.socketpair()
    client._socket = left
    frame = {"camera": np.arange(18, dtype=np.uint8).reshape(2, 3, 3)}

    def serve():
        size = struct.unpack("!I", right.recv(4))[0]
        payload = b""
        while len(payload) < size:
            payload += right.recv(size - len(payload))
        received = msgpack.unpackb(payload, object_hook=msgpack_numpy.decode, raw=False)
        np.testing.assert_array_equal(received["camera"], frame["camera"])
        response = msgpack.packb({"left": {"joint_pos": [0.1] * 7, "gripper": 0.25}, "arm_valid": True})
        # Exercise fragmented headers and bodies.
        for byte in struct.pack("!I", len(response)) + response:
            right.sendall(bytes([byte]))
        right.close()

    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(serve)
        reply = client._exchange(frame)
        future.result()
    assert client._valid(reply, "arm_valid")
    assert not client._valid(reply, "gripper_valid")
    assert client._valid({"gripper_valid": {"left": True}}, "gripper_valid")
    try:
        client._receive(4)
    except EOFError:
        pass
    else:
        raise AssertionError("Disconnected sockets must raise EOFError")
    client.close()

    # An uncommanded channel must hold the measured state, not CAP's seed pose.
    robot = SimpleNamespace(
        joint_names=[*[f"fr3_joint{i}" for i in range(1, 8)], "left_driver_joint"],
        data=SimpleNamespace(joint_pos=torch.tensor([[0.2] * 7 + [0.4]])),
    )
    env = SimpleNamespace(num_envs=1, device="cpu", scene={"robot": robot}, step_dt=0.02)
    env.unwrapped = env
    client._socket = SimpleNamespace(close=lambda: None)
    client._camera = lambda *args: {}
    client._tip_reach = lambda *args: 0.01
    client._exchange = lambda frame: {"left": {"joint_pos": [99] * 7, "gripper": 0.25}, "gripper_valid": True}
    action = client.get_action(env, {})
    torch.testing.assert_close(action[0, :7], torch.full((7,), 0.2))
    assert action[0, 7] == 0.75
    client._finished = True
    for _ in range(100):
        action = client.get_action(env, {})
    assert env.cap_episode_finished and action[0, 7] == 0.75
    client.reset()
    assert not env.cap_episode_finished and not client._finished
    return True


@pytest.mark.skipif(importlib.util.find_spec("msgpack_numpy") is None, reason="Install the cap client extra")
def test_cap_client():
    assert run_function_with_persistent_simulation_app(_test_cap_client)
