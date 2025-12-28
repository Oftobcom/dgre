# game_engine.py

"""
DG RE - Core Differential Game Engine
Implements the strategic interaction between platform and agents as a differential game.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# 1. STRATEGIC PRIMITIVES (Must be implemented first)
# ============================================================================

class AgentType(Enum):
    DRIVER = "driver"
    PASSENGER = "passenger"

class Action(Enum):
    """Minimal strategic action space"""
    COOPERATE = "cooperate"
    DEFECT = "defect"
    WAIT = "wait"
    ACCELERATE = "accelerate"  # Take more risk
    DECELERATE = "decelerate"  # Take less risk

@dataclass
class StrategicParams:
    """Agent's strategic type"""
    risk_tolerance: float = 0.5  # θ ∈ [0,1]
    discount_factor: float = 0.95  # δ ∈ (0,1)
    rationality: float = 1.0  # λ in quantal response
    reciprocity: float = 0.3  # Tit-for-tat tendency
    
    def quantal_response(self, expected_utilities: Dict[Action, float]) -> Dict[Action, float]:
        """Bounded rationality: logit choice probabilities"""
        if not expected_utilities:
            return {}
        
        # Quantal response function: P(a) ∝ exp(λ * EU(a))
        exp_values = {a: np.exp(self.rationality * eu) 
                     for a, eu in expected_utilities.items()}
        total = sum(exp_values.values())
        
        return {a: v/total for a, v in exp_values.items()}

@dataclass 
class RiskState:
    """Dynamic risk following controlled diffusion"""
    agent_id: str
    fatigue_risk: float = 0.0  # x₁ ∈ [0,1]
    reliability_risk: float = 0.0  # x₂ ∈ [0,1]
    strategic_risk: float = 0.0  # x₃ ∈ [0,1] - risk from game play
    timestamp: float = 0.0
    
    def to_vector(self) -> np.ndarray:
        return np.array([self.fatigue_risk, self.reliability_risk, self.strategic_risk])
    
    @classmethod
    def from_vector(cls, agent_id: str, vector: np.ndarray, timestamp: float):
        return cls(
            agent_id=agent_id,
            fatigue_risk=vector[0],
            reliability_risk=vector[1],
            strategic_risk=vector[2],
            timestamp=timestamp
        )

# ============================================================================
# 2. DIFFERENTIAL GAME DYNAMICS (Core Equations)
# ============================================================================

class RiskDynamics:
    """
    Implements the differential equation for risk evolution:
    dx = (A*x + B_platform*u_platform + B_agent*u_agent)dt + σ dW
    """
    
    def __init__(self):
        # State matrix A: how risks evolve naturally
        self.A = np.array([
            [-0.1, 0.02, 0.01],   # Fatigue risk decays slowly, affects others
            [0.01, -0.15, 0.03],  # Reliability risk
            [0.0, 0.0, -0.2]      # Strategic risk decays faster
        ])
        
        # Control matrices: how platform/agent actions affect risk
        self.B_platform = np.array([
            [0.1, 0.0, 0.0],   # Platform can reduce fatigue
            [0.0, 0.15, 0.0],  # Platform can improve reliability
            [0.0, 0.0, -0.1]   # Platform can reduce strategic risk
        ])
        
        self.B_agent = np.array([
            [0.3, 0.0, 0.0],   # Agent actions strongly affect fatigue
            [0.0, 0.25, 0.0],  # Agent affects reliability
            [0.1, 0.1, 0.4]    # Agent strongly affects strategic risk
        ])
        
        # Noise volatility
        self.sigma = 0.05
    
    def compute_drift(self, x: np.ndarray, u_platform: np.ndarray, 
                     u_agent: np.ndarray) -> np.ndarray:
        """Drift term: μ(t,x,u) = A*x + B_platform*u_platform + B_agent*u_agent"""
        return self.A @ x + self.B_platform @ u_platform + self.B_agent @ u_agent
    
    def step(self, x: np.ndarray, u_platform: np.ndarray, u_agent: np.ndarray,
             dt: float, noise: bool = True) -> np.ndarray:
        """
        Euler-Maruyama discretization:
        x_{t+1} = x_t + μ(t,x,u)dt + σ√dt * ξ
        """
        drift = self.compute_drift(x, u_platform, u_agent)
        
        # Deterministic step
        x_next = x + drift * dt
        
        # Add stochastic component if needed
        if noise:
            noise_term = self.sigma * np.sqrt(dt) * np.random.randn(3)
            x_next += noise_term
        
        # Ensure bounds [0,1]
        x_next = np.clip(x_next, 0.0, 1.0)
        
        return x_next

# ============================================================================
# 3. STRATEGIC AGENT (Implements Best Response)
# ============================================================================

class StrategicAgent:
    """Agent that plays the differential game"""
    
    def __init__(self, agent_id: str, agent_type: AgentType, 
                 params: StrategicParams):
        self.agent_id = agent_id
        self.type = agent_type
        self.params = params
        self.risk_state = RiskState(agent_id=agent_id)
        
        # Belief about platform's action
        self.belief_platform_action = np.zeros(3)
        
        # Memory of past interactions
        self.history: List[Tuple[float, Action, float]] = []  # (time, action, payoff)
    
    def expected_utility(self, action: Action, platform_action: np.ndarray,
                        current_risk: np.ndarray) -> float:
        """
        Compute expected utility of an action given beliefs.
        Simple linear-quadratic utility for now.
        """
        # Map action to control vector
        if action == Action.COOPERATE:
            u_agent = np.array([-0.2, -0.2, -0.1])  # Reduce risks
        elif action == Action.DEFECT:
            u_agent = np.array([0.3, 0.3, 0.4])     # Increase risks (short-term gain)
        elif action == Action.ACCELERATE:
            u_agent = np.array([0.2, 0.1, 0.2])     # Moderate risk increase
        elif action == Action.DECELERATE:
            u_agent = np.array([-0.1, -0.15, -0.2]) # Risk reduction
        else:  # WAIT
            u_agent = np.zeros(3)
        
        # Simple utility: U = -xᵀQx + Rᵀu - penalty*defection
        Q = np.diag([1.0, 1.0, 2.0])  # Penalize risk
        R = np.array([0.5, 0.5, 0.3]) # Reward control effort
        
        # Immediate payoff
        immediate = -current_risk.T @ Q @ current_risk + R.T @ u_agent
        
        # Future payoff (discounted)
        dynamics = RiskDynamics()
        future_risk = dynamics.step(current_risk, platform_action, u_agent, dt=1.0, noise=False)
        future = -future_risk.T @ Q @ future_risk * self.params.discount_factor
        
        # Defection penalty if reciprocating
        if action == Action.DEFECT and len(self.history) > 0:
            last_platform_action = self.belief_platform_action
            if np.all(last_platform_action < 0):  # Platform was cooperative
                immediate -= 0.5  # Guilt/reciprocity cost
        
        return immediate + future
    
    def best_response(self, platform_action: np.ndarray, 
                     current_risk: np.ndarray) -> Action:
        """
        Compute best response to platform action.
        Uses quantal response (bounded rationality).
        """
        # Update belief
        self.belief_platform_action = platform_action
        
        # Available actions
        available_actions = [Action.COOPERATE, Action.DEFECT, 
                           Action.ACCELERATE, Action.DECELERATE]
        
        # Compute expected utilities
        expected_utilities = {}
        for action in available_actions:
            eu = self.expected_utility(action, platform_action, current_risk)
            expected_utilities[action] = eu
        
        # Quantal response probabilities
        action_probs = self.params.quantal_response(expected_utilities)
        
        # Choose action (with some exploration for learning)
        if np.random.random() < 0.1:  # 10% exploration
            chosen = np.random.choice(available_actions)
        else:
            chosen = max(action_probs.items(), key=lambda x: x[1])[0]
        
        # Store history
        payoff = expected_utilities[chosen]
        self.history.append((self.risk_state.timestamp, chosen, payoff))
        
        return chosen
    
    def action_to_control(self, action: Action) -> np.ndarray:
        """Map discrete action to continuous control vector"""
        if action == Action.COOPERATE:
            return np.array([-0.2, -0.2, -0.1])
        elif action == Action.DEFECT:
            return np.array([0.3, 0.3, 0.4])
        elif action == Action.ACCELERATE:
            return np.array([0.2, 0.1, 0.2])
        elif action == Action.DECELERATE:
            return np.array([-0.1, -0.15, -0.2])
        else:  # WAIT
            return np.zeros(3)

# ============================================================================
# 4. PLATFORM CONTROLLER (Solves Optimal Control)
# ============================================================================

class PlatformController:
    """Platform that optimizes risk management"""
    
    def __init__(self):
        self.dynamics = RiskDynamics()
        
        # Cost matrices for linear-quadratic control
        self.Q = np.diag([1.0, 1.5, 2.0])  # State cost (penalize high risk)
        self.R = np.diag([0.1, 0.1, 0.1])  # Control cost (penalize intervention)
        
        # Prediction horizon
        self.horizon = 5
    
    def optimize_control(self, agent_states: Dict[str, np.ndarray],
                        agent_types: Dict[str, AgentType]) -> Dict[str, np.ndarray]:
        """
        Simple myopic optimization for MVP.
        In full version, this would solve Hamilton-Jacobi-Bellman.
        """
        platform_actions = {}
        
        for agent_id, x in agent_states.items():
            # Simple heuristic: more aggressive control for higher risk
            if agent_types[agent_id] == AgentType.DRIVER:
                # Drivers: focus on fatigue
                u = np.array([
                    -0.3 if x[0] > 0.5 else -0.1,  # Fatigue control
                    -0.2 if x[1] > 0.4 else -0.05, # Reliability control
                    -0.1 if x[2] > 0.3 else 0.0     # Strategic risk control
                ])
            else:
                # Passengers: focus on reliability
                u = np.array([
                    -0.1 if x[0] > 0.6 else 0.0,    # Fatigue control
                    -0.3 if x[1] > 0.5 else -0.1,   # Reliability control  
                    -0.15 if x[2] > 0.4 else -0.05  # Strategic risk control
                ])
            
            # Add noise to prevent perfect prediction by agents
            u += 0.05 * np.random.randn(3)
            u = np.clip(u, -0.5, 0.5)
            
            platform_actions[agent_id] = u
        
        return platform_actions

# ============================================================================
# 5. DIFFERENTIAL GAME ENGINE (Core Loop)
# ============================================================================

class DifferentialGameEngine:
    """
    Core differential game loop.
    Implements Stackelberg game: platform moves first, agents best respond.
    """
    
    def __init__(self, dt: float = 0.1, noise: bool = True):
        self.dt = dt
        self.noise = noise
        self.time = 0.0
        
        # Core components
        self.dynamics = RiskDynamics()
        self.platform = PlatformController()
        
        # Game state
        self.agents: Dict[str, StrategicAgent] = {}
        self.risk_states: Dict[str, RiskState] = {}
        
        # History for analysis
        self.history: List[Dict] = []
    
    def add_agent(self, agent: StrategicAgent, initial_risk: Optional[np.ndarray] = None):
        """Add agent to the game"""
        self.agents[agent.agent_id] = agent
        
        if initial_risk is None:
            initial_risk = np.random.rand(3) * 0.3  # Start with low risk
        
        agent.risk_state = RiskState.from_vector(
            agent.agent_id, initial_risk, self.time
        )
        self.risk_states[agent.agent_id] = agent.risk_state
    
    def game_step(self) -> Dict:
        """
        One step of the differential game:
        1. Platform computes optimal control
        2. Agents compute best responses
        3. Risk states evolve via differential equations
        4. Record outcomes
        """
        # Get current states
        current_states = {aid: a.risk_state.to_vector() 
                         for aid, a in self.agents.items()}
        agent_types = {aid: a.type for aid, a in self.agents.items()}
        
        # 1. PLATFORM OPTIMIZATION (Leader)
        platform_actions = self.platform.optimize_control(current_states, agent_types)
        
        # 2. AGENT BEST RESPONSES (Followers)
        agent_actions = {}
        agent_controls = {}
        
        for agent_id, agent in self.agents.items():
            # Agent observes platform action and chooses best response
            platform_action = platform_actions[agent_id]
            current_risk = current_states[agent_id]
            
            action = agent.best_response(platform_action, current_risk)
            control = agent.action_to_control(action)
            
            agent_actions[agent_id] = action
            agent_controls[agent_id] = control
        
        # 3. STATE EVOLUTION (Differential Equations)
        new_risk_states = {}
        for agent_id, agent in self.agents.items():
            x_current = current_states[agent_id]
            u_platform = platform_actions[agent_id]
            u_agent = agent_controls[agent_id]
            
            # Evolve risk via stochastic differential equation
            x_next = self.dynamics.step(
                x_current, u_platform, u_agent, 
                self.dt, noise=self.noise
            )
            
            # Update agent's risk state
            new_state = RiskState.from_vector(
                agent_id, x_next, self.time + self.dt
            )
            agent.risk_state = new_state
            new_risk_states[agent_id] = new_state
        
        # 4. RECORD GAME STATE
        game_state = {
            'time': self.time,
            'platform_actions': platform_actions,
            'agent_actions': agent_actions,
            'risk_states': new_risk_states,
            'agent_controls': agent_controls
        }
        self.history.append(game_state)
        
        # 5. ADVANCE TIME
        self.time += self.dt
        self.risk_states = new_risk_states
        
        return game_state
    
    def run_simulation(self, steps: int, warmup: int = 10) -> List[Dict]:
        """
        Run the differential game for multiple steps.
        Includes warmup for agents to learn.
        """
        logger.info(f"Starting differential game simulation: {steps} steps")
        
        # Warmup phase (agents learn without recording)
        for _ in range(warmup):
            self.game_step()
        
        # Main simulation
        results = []
        for step in range(steps):
            game_state = self.game_step()
            
            if step % 10 == 0:
                avg_risk = np.mean([s.to_vector().mean() 
                                  for s in game_state['risk_states'].values()])
                logger.info(f"Step {step}: Time={self.time:.1f}, Avg Risk={avg_risk:.3f}")
            
            results.append(game_state)
        
        logger.info(f"Simulation complete. Final time: {self.time:.1f}")
        return results
    
    def compute_equilibrium_metrics(self, window: int = 20) -> Dict:
        """
        Compute if game is converging to equilibrium.
        Simple metric: check if actions stabilize.
        """
        if len(self.history) < window:
            return {'converged': False, 'error': float('inf')}
        
        recent = self.history[-window:]
        
        # Check action convergence
        action_changes = []
        for i in range(1, len(recent)):
            prev = recent[i-1]['agent_actions']
            curr = recent[i]['agent_actions']
            
            # Count action changes
            changes = sum(1 for aid in prev if prev[aid] != curr.get(aid))
            action_changes.append(changes / len(prev))
        
        avg_change = np.mean(action_changes) if action_changes else 1.0
        
        return {
            'converged': avg_change < 0.1,  # Less than 10% action changes
            'action_stability': 1.0 - avg_change,
            'avg_risk': np.mean([s.to_vector().mean() 
                               for h in recent 
                               for s in h['risk_states'].values()])
        }

# ============================================================================
# 6. VISUALIZATION & ANALYSIS (Minimal for Debugging)
# ============================================================================

def analyze_strategic_behavior(results: List[Dict]):
    """Basic analysis of strategic interactions"""
    if not results:
        print("No results to analyze")
        return
    
    last_step = results[-1]
    
    print("\n" + "="*60)
    print("DIFFERENTIAL GAME ANALYSIS")
    print("="*60)
    
    # Action frequencies
    all_actions = []
    for step in results:
        all_actions.extend(list(step['agent_actions'].values()))
    
    action_counts = {a: all_actions.count(a) for a in set(all_actions)}
    total = len(all_actions)
    
    print("\nAgent Action Frequencies:")
    for action, count in action_counts.items():
        freq = count / total
        print(f"  {action.value}: {freq:.1%}")
    
    # Risk evolution
    initial_risk = results[0]['risk_states']
    final_risk = results[-1]['risk_states']
    
    print("\nRisk Evolution (Initial → Final):")
    for agent_id in initial_risk:
        init = initial_risk[agent_id].to_vector().mean()
        final = final_risk[agent_id].to_vector().mean()
        change = final - init
        print(f"  {agent_id}: {init:.3f} → {final:.3f} (Δ={change:+.3f})")
    
    # Platform intervention intensity
    platform_actions = [step['platform_actions'] for step in results]
    avg_intervention = np.mean([np.abs(u).mean() 
                               for step in platform_actions 
                               for u in step.values()])
    
    print(f"\nPlatform Intervention Intensity: {avg_intervention:.3f}")
    print("="*60)

# ============================================================================
# 7. MAIN EXECUTION (Test the Engine)
# ============================================================================

def test_differential_game():
    """Run a minimal differential game to prove strategic interaction"""
    
    # Initialize game engine
    engine = DifferentialGameEngine(dt=0.5, noise=True)
    
    # Create strategic agents with different parameters
    agents = [
        StrategicAgent(
            agent_id="driver_1",
            agent_type=AgentType.DRIVER,
            params=StrategicParams(
                risk_tolerance=0.7,  # Risk-seeking
                discount_factor=0.9,
                rationality=0.8,
                reciprocity=0.1
            )
        ),
        StrategicAgent(
            agent_id="driver_2", 
            agent_type=AgentType.DRIVER,
            params=StrategicParams(
                risk_tolerance=0.3,  # Risk-averse
                discount_factor=0.98,
                rationality=1.2,
                reciprocity=0.6
            )
        ),
        StrategicAgent(
            agent_id="passenger_1",
            agent_type=AgentType.PASSENGER,
            params=StrategicParams(
                risk_tolerance=0.5,
                discount_factor=0.95,
                rationality=1.0,
                reciprocity=0.4
            )
        )
    ]
    
    # Add agents with random initial risks
    for agent in agents:
        initial_risk = np.random.rand(3) * 0.4
        engine.add_agent(agent, initial_risk)
    
    # Run the differential game
    print("Starting differential game simulation...")
    results = engine.run_simulation(steps=50, warmup=5)
    
    # Analyze strategic behavior
    analyze_strategic_behavior(results)
    
    # Check for equilibrium
    metrics = engine.compute_equilibrium_metrics()
    print(f"\nEquilibrium Metrics:")
    print(f"  Converged: {metrics['converged']}")
    print(f"  Action Stability: {metrics['action_stability']:.3f}")
    print(f"  Average Risk: {metrics['avg_risk']:.3f}")
    
    # Strategic proof: did agents respond differently?
    last_actions = results[-1]['agent_actions']
    unique_actions = len(set(last_actions.values()))
    
    print(f"\nSTRATEGIC PROOF:")
    if unique_actions > 1:
        print("✓ Agents chose DIFFERENT actions (strategic differentiation)")
    else:
        print("✗ All agents chose SAME action (no strategic reasoning)")
    
    # Show agent-specific outcomes
    print("\nAgent-Specific Outcomes:")
    for agent_id, agent in engine.agents.items():
        cooperations = sum(1 for h in engine.history 
                          if h['agent_actions'].get(agent_id) == Action.COOPERATE)
        total = len([h for h in engine.history if agent_id in h['agent_actions']])
        
        if total > 0:
            coop_rate = cooperations / total
            final_risk = agent.risk_state.to_vector().mean()
            print(f"  {agent_id}: Cooperation={coop_rate:.1%}, Final Risk={final_risk:.3f}")
    
    return engine, results

if __name__ == "__main__":
    print("DG RE - Differential Game Engine Test")
    print("=" * 60)
    
    # Run test
    engine, results = test_differential_game()
    
    print("\n" + "="*60)
    print("Differential Game Engine initialized successfully!")
    print(f"Simulated {len(results)} steps with {len(engine.agents)} strategic agents.")
    print("="*60)