# VivanceData Real-Time Anomaly Detection System

![Anomaly Detection](https://raw.githubusercontent.com/Vivancedata/anomaly-detection-system/master/frontend/src/assets/favicon.svg)

## Overview

VivanceData's Real-Time Anomaly Detection System is a powerful, scalable solution designed to monitor data streams and identify unusual patterns or anomalies in real-time. Built to handle high-throughput environments, this system helps organizations detect critical events such as fraud, security breaches, equipment failures, and other abnormal activities before they cause significant damage.

## Key Features

- **Real-Time Processing**: Millisecond-level detection of anomalies across multiple data streams
- **Multi-Model Approach**: Ensemble of statistical, machine learning, and deep learning algorithms
- **Adaptive Thresholds**: Automatically adjusts sensitivity based on data patterns and feedback
- **Multivariate Analysis**: Detects complex anomalies across correlated variables
- **Auto-Calibration**: Self-adjusts to evolving data distributions and seasonality
- **Explainable Results**: Clear visualizations and explanations of detected anomalies
- **Customizable Alerting**: Flexible notification system with prioritization and routing
- **Historical Analysis**: Compare current anomalies with historical patterns
- **Low False Positive Rate**: Advanced filtering to minimize alert fatigue

## Tech Stack

- **Backend**:
  - Python
  - FastAPI
  - Rust (for performance-critical components)
  - TensorFlow, PyTorch, Scikit-learn
  - Apache Kafka/Pulsar for streaming

- **Frontend**:
  - React
  - TypeScript
  - Vite
  - Tailwind CSS
  - Shadcn UI components

- **Storage**:
  - TimescaleDB
  - InfluxDB
  - Elasticsearch

- **Deployment**:
  - Docker
  - Kubernetes

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.8+
- Node.js 16+
- Kafka or Pulsar cluster (or use our included development setup)

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/vivancedata/anomaly-detection-system.git
cd anomaly-detection-system
```

2. Start the development environment:
```bash
docker-compose up -d
```

3. Access the dashboard at http://localhost:8080

### Local Development Setup

For developers contributing to the system:

1. Set up the backend:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

2. Set up the frontend:
```bash
cd frontend
npm install
npm run dev
```

## Project Structure

```
├── backend/               # Python FastAPI backend
│   ├── src/               # Source code
│   │   ├── api/           # API endpoints
│   │   ├── config/        # Configuration
│   │   ├── models/        # Anomaly detection models
│   │   ├── services/      # Business logic
│   │   └── utils/         # Utilities
│   └── Dockerfile         # Backend container
├── frontend/              # React/TypeScript frontend
│   ├── src/               # Source code
│   │   ├── components/    # Reusable components
│   │   ├── pages/         # Application pages
│   │   ├── lib/           # Utility libraries
│   │   ├── types/         # TypeScript types
│   │   └── utils/         # Utility functions
│   └── Dockerfile         # Frontend container
├── data/                  # Data directories
│   ├── input/             # Raw input data
│   ├── processed/         # Processed data
│   └── models/            # Saved models
└── docker-compose.yml     # Development environment
```

## Contributing

We welcome contributions to the VivanceData Anomaly Detection System! Please see our [Contributing Guide](docs/contributing.md) for details.

## License

This project is proprietary and confidential. Unauthorized copying, distribution, or use is strictly prohibited.

## Contact

For more information, please contact us at anomaly-detection@vivancedata.com or visit our [website](https://vivancedata.com).

---

© 2025 VivanceData, Inc. All Rights Reserved.
