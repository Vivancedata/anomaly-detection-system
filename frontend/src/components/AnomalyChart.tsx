import React, { useEffect, useRef } from 'react';
import { ChartData } from '../types';
import './AnomalyChart.css';

interface AnomalyChartProps {
  data: ChartData;
}

const AnomalyChart: React.FC<AnomalyChartProps> = ({ data }) => {
  const chartRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!chartRef.current) return;

    const canvas = chartRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Set canvas dimensions
    const width = canvas.width;
    const height = canvas.height;

    // Chart dimensions
    const padding = { top: 20, right: 20, bottom: 30, left: 50 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    // Find max value for scaling
    const maxDataValue = Math.max(
      ...data.normal,
      ...data.threshold,
      ...data.anomalies.map(val => val * 20) // Scale anomalies for visibility
    );

    // Draw axes
    ctx.beginPath();
    ctx.strokeStyle = '#ccc';
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, height - padding.bottom);
    ctx.lineTo(width - padding.right, height - padding.bottom);
    ctx.stroke();

    // Draw data
    if (data.timestamps.length === 0) return;

    const timeStep = chartWidth / (data.timestamps.length - 1);

    // Function to draw a line series
    const drawLine = (
      dataPoints: number[],
      color: string,
      lineWidth: number = 2,
      dashed: boolean = false
    ) => {
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      if (dashed) {
        ctx.setLineDash([5, 3]);
      } else {
        ctx.setLineDash([]);
      }

      dataPoints.forEach((value, index) => {
        const x = padding.left + index * timeStep;
        const y =
          height - padding.bottom - (value / maxDataValue) * chartHeight;

        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();
    };

    // Draw normal data line
    drawLine(data.normal, '#3498db');

    // Draw threshold line
    drawLine(data.threshold, '#e74c3c', 1, true);

    // Draw anomalies as points
    ctx.setLineDash([]);
    data.anomalies.forEach((value, index) => {
      if (value > 0) {
        const x = padding.left + index * timeStep;
        const y = height - padding.bottom - (data.normal[index] / maxDataValue) * chartHeight;
        
        ctx.beginPath();
        ctx.fillStyle = '#e74c3c';
        ctx.arc(x, y, value + 3, 0, Math.PI * 2);
        ctx.fill();
      }
    });

    // Draw time axis labels
    ctx.fillStyle = '#666';
    ctx.font = '10px Arial';
    ctx.textAlign = 'center';

    // Only show a subset of labels to avoid overcrowding
    const labelStep = Math.max(1, Math.floor(data.timestamps.length / 6));
    for (let i = 0; i < data.timestamps.length; i += labelStep) {
      const x = padding.left + i * timeStep;
      const time = new Date(data.timestamps[i]);
      const label = `${time.getHours()}:${time.getMinutes().toString().padStart(2, '0')}`;
      
      ctx.fillText(label, x, height - padding.bottom + 15);
    }

    // Add legend
    const legendY = padding.top + 15;
    
    // Normal line
    ctx.beginPath();
    ctx.strokeStyle = '#3498db';
    ctx.lineWidth = 2;
    ctx.setLineDash([]);
    ctx.moveTo(width - padding.right - 100, legendY);
    ctx.lineTo(width - padding.right - 70, legendY);
    ctx.stroke();
    
    ctx.fillStyle = '#666';
    ctx.textAlign = 'left';
    ctx.fillText('Normal', width - padding.right - 65, legendY + 4);
    
    // Threshold line
    ctx.beginPath();
    ctx.strokeStyle = '#e74c3c';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 3]);
    ctx.moveTo(width - padding.right - 100, legendY + 20);
    ctx.lineTo(width - padding.right - 70, legendY + 20);
    ctx.stroke();
    
    ctx.fillText('Threshold', width - padding.right - 65, legendY + 24);
    
    // Anomaly point
    ctx.beginPath();
    ctx.fillStyle = '#e74c3c';
    ctx.setLineDash([]);
    ctx.arc(width - padding.right - 85, legendY + 40, 5, 0, Math.PI * 2);
    ctx.fill();
    
    ctx.fillStyle = '#666';
    ctx.fillText('Anomaly', width - padding.right - 65, legendY + 44);
    
  }, [data]);

  return (
    <div className="anomaly-chart">
      <canvas 
        ref={chartRef} 
        width={700} 
        height={350} 
        className="chart-canvas"
      ></canvas>
    </div>
  );
};

export default AnomalyChart;
