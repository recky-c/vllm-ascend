"""POC 8a: Barrier Bridge verification (no MTE).

Verifies whether a capture-stream HCCL collective can correctly depend on a
captured side-stream (comm_stream) event, and vice versa.

DAG (all_reduce stays on capture stream, dummy ops on comm stream):

    Capture Stream C                 Comm Stream M

    READY all_reduce
          |
          v
    record push_start
          | ----------------------->
                                    wait push_start
                                          |
                                          v
                                      dummy op
                                          |
                                    record push_done
          |<-------------------------------
          |
    wait push_done
          |
          v
    DONE all_reduce
          |
          v
    record recv_start
          | ----------------------->
                                    wait recv_start
                                          |
                                          v
                                      dummy op
                                          |
                                    record recv_done
          |<-------------------------------
          |
    wait recv_done
          |
          v
    final op

PASS criteria: capture ok, 100 replays ok, no hang, no 107024/107028,
result correct, both ranks print PASS.
"""
import os

import torch
import torch_npu
import torch.distributed as dist


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, f"POC 8a requires world_size==2, got {world_size}"

    device = torch.npu.current_device()
    print(f"[POC8A][rank={rank}] device={device}", flush=True)

    capture_stream = torch.npu.Stream()
    comm_stream = torch.npu.Stream()

    push_start = torch.npu.Event()
    push_done = torch.npu.Event()
    recv_start = torch.npu.Event()
    recv_done = torch.npu.Event()

    ready_token = torch.zeros(1, dtype=torch.int32, device="npu")
    done_token = torch.zeros(1, dtype=torch.int32, device="npu")

    comm_buffer = torch.zeros(4096, dtype=torch.float32, device="npu")
    final_buffer = torch.zeros_like(comm_buffer)

    torch.npu.synchronize()
    dist.barrier()

    graph = torch.npu.NPUGraph()

    print(f"[POC8A][rank={rank}] starting graph capture", flush=True)

    with torch.npu.graph(graph, stream=capture_stream):
        # phase 1: READY all_reduce on capture stream
        dist.all_reduce(ready_token, group=dist.group.WORLD)
        push_start.record(capture_stream)

        # phase 2: side stream dummy op
        with torch.npu.stream(comm_stream):
            comm_stream.wait_event(push_start)
            comm_buffer.add_(1.0)
            push_done.record(comm_stream)

        # phase 3: capture stream waits side stream BEFORE second collective
        capture_stream.wait_event(push_done)
        dist.all_reduce(done_token, group=dist.group.WORLD)

        # phase 4: second capture -> comm bridge
        recv_start.record(capture_stream)
        with torch.npu.stream(comm_stream):
            comm_stream.wait_event(recv_start)
            comm_buffer.add_(2.0)
            recv_done.record(comm_stream)

        # phase 5: capture waits recv_done, then final op
        capture_stream.wait_event(recv_done)
        final_buffer.copy_(comm_buffer)

    print(f"[POC8A][rank={rank}] capture OK", flush=True)

    torch.npu.synchronize()
    # capture itself may have executed one kernel, so clear buffers before replay
    comm_buffer.zero_()
    final_buffer.zero_()
    torch.npu.synchronize()
    dist.barrier()

    REPLAY_COUNT = 100
    for i in range(REPLAY_COUNT):
        if rank == 0 and i % 10 == 0:
            print(f"[POC8A] replay {i}", flush=True)
        graph.replay()

    torch.npu.synchronize()
    dist.barrier()

    expected = REPLAY_COUNT * 3.0
    actual = final_buffer.cpu()
    assert torch.allclose(
        actual, torch.full_like(actual, expected)
    ), f"rank={rank}: expected={expected}, actual={actual[:8]}"

    print(f"[POC8A][rank={rank}] PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
