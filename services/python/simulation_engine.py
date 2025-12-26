import numpy as np
from services.agent_model import Agent, RiskState

class SimulationEngine:
    @staticmethod
    def run_simulation(agents: list, steps: int, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
        simulation_result = []
        for step in range(steps):
            step_risks = []
            for agent in agents:
                risk = agent.current_risk
                if agent.type == "DRIVER":
                    risk.driver_risk = min(1.0, risk.driver_risk + np.random.rand()*0.05)
                if agent.type == "PASSENGER":
                    risk.passenger_risk = min(1.0, risk.passenger_risk + np.random.rand()*0.03)
                risk.platform_risk = min(1.0, np.random.rand()*0.02)
                step_risks.append(risk)
            simulation_result.append(step_risks)
        return simulation_result
