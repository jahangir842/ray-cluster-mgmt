"""Calculate a sum of squares across every live Ray node."""

import socket
from time import perf_counter

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


@ray.remote(num_cpus=1, num_gpus=0)
def sum_squares(start, stop):
    return socket.gethostname(), sum(i * i for i in range(start, stop))


def main():
    ray.init(address="auto")
    try:
        nodes = sorted(
            (node for node in ray.nodes() if node["Alive"]),
            key=lambda node: node["NodeManagerAddress"],
        )
        print(f"Connected to {len(nodes)} nodes", flush=True)
        chunk_size = 1_000_000
        started = perf_counter()
        tasks = [
            sum_squares.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node["NodeID"], soft=False
                )
            ).remote(i * chunk_size, (i + 1) * chunk_size)
            for i, node in enumerate(nodes)
        ]
        results = ray.get(tasks, timeout=120)
        for node, (hostname, subtotal) in zip(nodes, results):
            print(f"{node['NodeManagerAddress']} ({hostname}): {subtotal}")
        count = len(nodes) * chunk_size
        actual = sum(subtotal for _, subtotal in results)
        expected = (count - 1) * count * (2 * count - 1) // 6
        assert actual == expected, (actual, expected)
        print(f"PASS: sum of squares for {count:,} integers = {actual}")
        print(f"Distributed tasks completed in {perf_counter() - started:.2f}s")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
