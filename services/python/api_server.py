from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from services.agent_model import Agent, RiskState
from services.policy_engine import PolicyEngine
from services.simulation_engine import SimulationEngine

app = FastAPI(title="DG RE MVP API")

# Pydantic models for request/response
class PolicyRequest(BaseModel):
    agent_ids: List[str]
    types: List[str]  # "DRIVER" or "PASSENGER"
    steps: Optional[int] = 1
    policies: Optional[List[dict]] = []

class RiskResponse(BaseModel):
    agent_id: str
    driver_risk: float
    passenger_risk: float
    platform_risk: float

@app.post("/evaluate_policy", response_model=List[RiskResponse])
def evaluate_policy(request: PolicyRequest):
    # Create Agent objects
    agents = []
    for i, agent_id in enumerate(request.agent_ids):
        agents.append(Agent(agent_id=agent_id, type=request.types[i]))
    
    # Apply policies
    updated_risks = PolicyEngine.apply_policies(agents, request.policies)
    # Convert to response
    return [
        RiskResponse(
            agent_id=r.agent_id,
            driver_risk=r.driver_risk,
            passenger_risk=r.passenger_risk,
            platform_risk=r.platform_risk
        )
        for r in updated_risks
    ]

@app.post("/simulate", response_model=List[List[RiskResponse]])
def simulate(request: PolicyRequest):
    agents = []
    for i, agent_id in enumerate(request.agent_ids):
        agents.append(Agent(agent_id=agent_id, type=request.types[i]))
    
    sim_result = SimulationEngine.run_simulation(agents, steps=request.steps or 5, seed=42)
    
    # Convert to response
    result = []
    for step_risks in sim_result:
        step_response = [
            RiskResponse(
                agent_id=r.agent_id,
                driver_risk=r.driver_risk,
                passenger_risk=r.passenger_risk,
                platform_risk=r.platform_risk
            )
            for r in step_risks
        ]
        result.append(step_response)
    return result
