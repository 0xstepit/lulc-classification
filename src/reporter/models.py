from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ReportEntry:
    step: str
    data: Any
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)
