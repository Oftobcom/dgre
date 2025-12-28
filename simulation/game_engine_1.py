# dgre/simulation/game_engine_1.py
class DifferentialGameEngine:
    def step(self, t: float, dt: float) -> GameState:
        # Platform computes optimal control
        platform_action = self.platform_controller.optimize(self.state)
        
        # Agents compute best responses
        agent_actions = [
            agent.best_response(platform_action) 
            for agent in self.agents
        ]
        
        # State evolves via differential equations
        new_state = self.dynamics(self.state, platform_action, agent_actions)
        
        return GameState(new_state, self.compute_risk(new_state))