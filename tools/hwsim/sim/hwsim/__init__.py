"""hwsim -- hardware what-if discrete-event simulator for Sirius query traces (WS6, v0).

Replays the task graph recorded by Quent telemetry under a resource model
(executor threads, FIFO task queue, GPU memory-pool admission, transfer
channels) and re-times / re-flows it when hardware knobs change.

See tools/hwsim/docs/simulator-design.md for the model description.
"""

__version__ = "0.1.0"
