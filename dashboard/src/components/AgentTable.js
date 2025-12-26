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
