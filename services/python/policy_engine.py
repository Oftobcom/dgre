from services.agent_model import Agent, RiskState
from typing import List

class PolicyEngine:
    @staticmethod
    def apply_policies(agents: List[Agent], policies: List[dict]) -> List[RiskState]:
        updated_risks = []
        for agent in agents:
            risk = agent.current_risk
            # Example: naive risk adjustment
            if agent.type == "DRIVER":
                risk.driver_risk = min(1.0, risk.driver_risk + 0.1)
            if agent.type == "PASSENGER":
                risk.passenger_risk = min(1.0, risk.passenger_risk + 0.05)
            # Apply platform policies
            for policy in policies:
                if agent.agent_id in policy.get("target_agent_ids", []):
                    risk.driver_risk = min(1.0, risk.driver_risk + policy.get("adjustment_factor", 0))
            updated_risks.append(risk)
        return updated_risks
