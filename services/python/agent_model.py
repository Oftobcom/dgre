from dataclasses import dataclass
from typing import Optional

@dataclass
class RiskState:
    agent_id: str
    driver_risk: float = 0.0
    passenger_risk: float = 0.0
    platform_risk: float = 0.0

@dataclass
class Agent:
    agent_id: str
    type: str  # "DRIVER" or "PASSENGER"
    active: bool = True
    current_risk: RiskState = None
    driver_fields: Optional[dict] = None
    passenger_fields: Optional[dict] = None

    def __post_init__(self):
        if self.current_risk is None:
            self.current_risk = RiskState(agent_id=self.agent_id)
