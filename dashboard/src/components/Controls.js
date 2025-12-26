import React, { useState } from 'react';

const Controls = ({ onRun }) => {
    const [agents, setAgents] = useState("driver_1:DRIVER,passenger_1:PASSENGER");
    const [steps, setSteps] = useState(5);

    const handleRun = () => {
        const agentList = agents.split(",").map(a => a.split(":")[0]);
        const typeList = agents.split(",").map(a => a.split(":")[1]);
        onRun(agentList, typeList, steps);
    };

    return (
        <div style={{ marginBottom: "20px" }}>
            <div>
                <label>Agents (id:type, comma-separated): </label>
                <input type="text" value={agents} onChange={e => setAgents(e.target.value)} size={50}/>
            </div>
            <div>
                <label>Simulation Steps: </label>
                <input type="number" value={steps} onChange={e => setSteps(parseInt(e.target.value))}/>
            </div>
            <button onClick={handleRun}>Run Simulation</button>
        </div>
    );
};

export default Controls;
