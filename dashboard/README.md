Perfect! Let’s create a **React.js frontend dashboard** for DG RE MVP, which will **visualize agent risks and simulation steps live** by calling the FastAPI backend we already implemented.

---

# DG RE React.js Dashboard

## 1️⃣ Project Structure (React)

```
dgre_dashboard/
├─ public/
│  └─ index.html
├─ src/
│  ├─ App.js
│  ├─ index.js
│  ├─ api.js             # Axios calls to FastAPI
│  ├─ components/
│  │  ├─ AgentTable.js   # Table of agent risks
│  │  ├─ SimulationChart.js # Chart of simulation steps
│  │  └─ Controls.js     # Input form for agents/policies
├─ package.json
```

---

## 2️⃣ Install Dependencies

```bash
npx create-react-app dgre_dashboard
cd dgre_dashboard
npm install axios chart.js react-chartjs-2
```

---

## 3️⃣ `src/api.js` — FastAPI Calls

```javascript
import axios from 'axios';

const API_BASE = "http://localhost:8000";

export const evaluatePolicy = async (agentIds, types, policies=[]) => {
    const response = await axios.post(`${API_BASE}/evaluate_policy`, {
        agent_ids: agentIds,
        types: types,
        policies: policies
    });
    return response.data;
};

export const simulate = async (agentIds, types, steps=5) => {
    const response = await axios.post(`${API_BASE}/simulate`, {
        agent_ids: agentIds,
        types: types,
        steps: steps
    });
    return response.data;
};
```

---

## 4️⃣ `src/components/Controls.js` — Input Form

```javascript
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
```

---

## 5️⃣ `src/components/AgentTable.js` — Table of Agent Risks

```javascript
import React from 'react';

const AgentTable = ({ agents }) => {
    if (!agents || agents.length === 0) return null;

    return (
        <table border="1" cellPadding="5" style={{ marginBottom: "20px" }}>
            <thead>
                <tr>
                    <th>Agent ID</th>
                    <th>Driver Risk</th>
                    <th>Passenger Risk</th>
                    <th>Platform Risk</th>
                </tr>
            </thead>
            <tbody>
                {agents.map(agent => (
                    <tr key={agent.agent_id}>
                        <td>{agent.agent_id}</td>
                        <td>{agent.driver_risk.toFixed(2)}</td>
                        <td>{agent.passenger_risk.toFixed(2)}</td>
                        <td>{agent.platform_risk.toFixed(2)}</td>
                    </tr>
                ))}
            </tbody>
        </table>
    );
};

export default AgentTable;
```

---

## 6️⃣ `src/components/SimulationChart.js` — Chart of Simulation Steps

```javascript
import React from 'react';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const SimulationChart = ({ simResult }) => {
    if (!simResult || simResult.length === 0) return null;

    const labels = simResult.map((_, i) => `Step ${i+1}`);
    const datasets = simResult[0].map((_, idx) => ({
        label: simResult[0][idx].agent_id,
        data: simResult.map(step => step[idx].driver_risk),
        borderColor: `hsl(${idx*60}, 70%, 50%)`,
        fill: false
    }));

    return <Line data={{ labels, datasets }} />;
};

export default SimulationChart;
```

---

## 7️⃣ `src/App.js` — Main Dashboard

```javascript
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
```

---

## 8️⃣ Run the Dashboard

```bash
npm start
```

* Navigate to `http://localhost:3000`
* Enter agents: `driver_1:DRIVER,passenger_1:PASSENGER`
* Set steps: `5`
* Click **Run Simulation**
* View **table of agent risks** and **line chart over simulation steps**

---

## ✅ Features of this Dashboard

* **Interactive input** for agents and simulation steps
* **Live visualization** of agent risk evolution
* **Supports multiple agents**
* **Easy to extend** for new risk metrics, policies, or simulation results
* **Forward-compatible** with gRPC backend or C++ risk core

---

If you want, I can **enhance this dashboard** further to include:

1. **Policy editor**: add/edit platform policies dynamically
2. **Multi-metric chart**: show driver_risk, passenger_risk, platform_risk simultaneously
3. **Agent filtering / sorting** for large simulations

This will make DG RE fully **interactive and research-ready**.

Do you want me to do that next?
