import React from 'react';
import { useLocation } from 'react-router-dom';
import { PieChart, Pie, Cell, Tooltip, Legend } from 'recharts';
import './BillDashboard.css'; // Import the CSS file

const COLORS = ['#03E4DF', '#a81b23', '#0EF691']; // Simplified colors for pie chart

const BillDashboard = () => {
    const location = useLocation(); // access location object
    const { billData, congressType, summary, classification } = location.state || { billData: null, congressType: null, summary: null, classification: null };

    // Log the received state to debug
    console.log('Received billData:', billData);
    console.log('Received congressType:', congressType);
    console.log('Received summary:', summary);
    console.log('Received classification:', classification);

    if (!billData) {
        return <div>No data available</div>; // display message when no data is available
    }

    const { title } = billData; // destruct bill data

    // Ensure classification values are numbers and log them
    const data = [
        { name: 'Democratic', value: Number(classification.Democratic) },
        { name: 'Republican', value: Number(classification.Republican) },
        { name: 'Middle', value: Number(classification.Middle) },
    ];

    console.log('Pie chart data:', data); // Log pie chart data

    // Check if data is correctly formatted
    if (!Array.isArray(data) || data.length === 0 || data.some(entry => typeof entry.value !== 'number')) {
        console.error('Data is not correctly formatted for PieChart:', data);
        return <div>Invalid data for chart</div>;
    }

    // Custom tooltip to display percentage
    const CustomTooltip = ({ active, payload }) => {
        if (active && payload && payload.length) {
            const { name, value } = payload[0];
            const percentage = (value * 100).toFixed(2) + '%';
            return (
                <div className="custom-tooltip">
                    <p className="label">{`${name} : ${percentage}`}</p>
                </div>
            );
        }
        return null;
    };

    // styling of dashboard
    return (
        <div className={`dashboard-container ${congressType ? congressType.toLowerCase() : ''}`}>
            <h2 className="dashboard-title">{title}</h2>
            <div className="container-wrapper">
                <div className="chart-and-type-container">
                    <h3 className="congress-type">Congress Type: {congressType}</h3> {/* Display congress type */}
                    <div className="chart-container">
                        <PieChart width={400} height={400}>
                            <Pie
                                data={data}
                                cx={200}
                                cy={200}
                                labelLine={false}
                                outerRadius={80}
                                fill="#8884d8"
                                dataKey="value"
                            >
                                {data.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                ))}
                            </Pie>
                            <Tooltip content={<CustomTooltip />} />
                            <Legend />
                        </PieChart>
                    </div>
                </div>
                <div className="summary-content-container">
                    <h2 className="summary-title">Summary</h2>
                    <div className="summary-container">
                        <div className="summary-content">
                            <p>{summary}</p> {/* Display the summary */}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default BillDashboard;