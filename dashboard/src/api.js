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
