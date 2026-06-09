"""Format-drift tracking.

We don't crash on a single bad line; we record a warning so the skill can
surface "this log may have been parsed with a stale parser" to the user.
"""

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class FormatWarnings:
    # Maps anchor name -> count of lines that matched the loose prefix
    # but failed the strict regex.
    drift_counts: Counter = field(default_factory=Counter)
    # Up to 3 sample lines per anchor for forensic inspection.
    drift_samples: dict = field(default_factory=dict)

    def record_drift(self, anchor_name: str, line: str) -> None:
        self.drift_counts[anchor_name] += 1
        samples = self.drift_samples.setdefault(anchor_name, [])
        if len(samples) < 3:
            samples.append(line.rstrip())

    def to_dict(self) -> dict:
        return {
            "drift_counts": dict(self.drift_counts),
            "drift_samples": self.drift_samples,
        }

    @property
    def has_drift(self) -> bool:
        return bool(self.drift_counts)
