"""
Command integers for multi-stage pipeline RPC over raw tensor transport.

We encode the command as a 0-dim int32 tensor sent before the payload.
"""

CMD_FWD = 1  # Forward only: recv x, attn -> send y
CMD_FWD_LOSS = 2  # Last stage forward+loss: recv x, attn, labels -> send loss, dL/dx
CMD_BWD = 3  # Backward for intermediate: recv x, attn, upstream_grad -> send dL/dx
CMD_OPT_STEP = 4  # Optimizer step + zero_grad (gradient sync boundary)

# Checkpoint command ids, not the step number.
# The save frequency is controlled by save_every_n_steps and a modulo check in the coordinatorhow to tesst it

CMD_SAVE_CKPT = 5  # Save checkpoint at given step: recv step(int32) -> send ack(int32)
CMD_LOAD_CKPT = 6  # Load checkpoint at given step: recv step(int32) -> send ack(int32)
CMD_REPORT_STATE = 7  # Optional: send current opt step/int state
CMD_HAS_CKPT = 8  # Check if checkpoint for step exists: recv step(int32) -> send ack(int32)
