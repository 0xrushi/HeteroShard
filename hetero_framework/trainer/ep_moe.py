"""
Expert-Parallel MoE Worker (stub)

Placeholder for a future worker that would support expert-parallel MoE with
intra-layer token routing across multiple GPUs (all-to-all). This is not used
in the current pipeline trainer, which performs layer-parallel (pipeline) only.
"""


class ExpertParallelMoEWorker:
    def __init__(self):
        raise NotImplementedError(
            "ExpertParallelMoEWorker is a stub. Expert-parallel MoE requires "
            "collective all-to-all communication and is outside the current "
            "socket-based pipeline scope."
        )
