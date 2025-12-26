import React, { useState } from 'react';
import Controls from './components/Controls';
import AgentTable from './components/AgentTable';
import SimulationChart from './components/SimulationChart';
import { evaluatePolicy, simulate } from './api';

function App() {
    const [agents, setAgents] = useState([]);
    const [simResult, setSimResult] = useState([]);

    const runSimulation = async (agentIds, types, steps) => {
        const result = await simulate(agentIds, types, steps);
        setSimResult(result);
        setAgents(result[result.length - 1]); // show last step
    };

    return (
        <div style={{ padding: "20px" }}>
            <h1>DG RE Dashboard</h1>
            <Controls onRun={runSimulation} />
            <AgentTable agents={agents} />
            <SimulationChart simResult={simResult} />
        </div>
    );
}

export default App;
