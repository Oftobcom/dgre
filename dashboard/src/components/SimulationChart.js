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
