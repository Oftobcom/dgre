from services.agent_model import Agent
from services.policy_engine import PolicyEngine
from services.simulation_engine import SimulationEngine

# Create example agents
agents = [
    Agent(agent_id="driver_1", type="DRIVER", driver_fields={"vehicle_id":"v1"}),
    Agent(agent_id="passenger_1", type="PASSENGER", passenger_fields={"payment_method_id":"pm1"})
]

# Example policies
policies = [{"policy_id":"p1", "target_agent_ids":["driver_1"], "adjustment_factor":0.05}]

# Apply policy
updated_risks = PolicyEngine.apply_policies(agents, policies)
print("Updated Risks after Policy:")
for r in updated_risks:
    print(r)

# Run simulation
sim_result = SimulationEngine.run_simulation(agents, steps=5, seed=42)
print("\nSimulation Result:")
for step_num, step_risks in enumerate(sim_result):
    print(f"Step {step_num+1}:")
    for r in step_risks:
        print(r)
