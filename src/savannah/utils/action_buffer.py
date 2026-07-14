import collections

import torch


class ActionBuffer:
    def __init__(self, execute_steps: int):
        """
        execute_steps (n): How many actions to execute before querying the model again.
        """
        self.execute_steps = execute_steps
        self.queue = collections.deque()

    def is_empty(self) -> bool:
        return len(self.queue) == 0

    def push(self, action_chunk: torch.Tensor):
        """
        Takes the (T, action_dim) chunk from the policy.
        Slices it to only keep the first `execute_steps`, and adds them to the queue.
        """
        # (T, action_dim) -> numpy array
        chunk_np = action_chunk.cpu()

        # Slicing the receding horizon
        actions_to_execute = chunk_np[: self.execute_steps]

        # Add to our execution queue
        self.queue.extend(actions_to_execute)

    def pop(self) -> torch.Tensor:
        """Returns the next single action to step the environment."""
        return self.queue.popleft()
