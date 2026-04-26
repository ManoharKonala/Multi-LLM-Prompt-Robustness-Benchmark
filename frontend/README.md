# PRB Dashboard

The official visualization interface for the **Multi-LLM Prompt Robustness Benchmark**.

## Overview
This dashboard is a React-based web application that parses benchmark results and provides an interactive overview of model performance.

### Features
- **Robustness Matrix**: Real-time color-coded table of model scores across attack types.
*   **Rankings**: Dynamic bar charts showing which LLM is most resilient.
- **Run Log**: Real-time terminal output simulation.
- **Coverage Map**: Roadmap of implemented and planned perturbation types.

## Setup

1.  **Enter Directory**:
    ```bash
    cd frontend
    ```

2.  **Install Packages**:
    ```bash
    npm install
    ```

3.  **Launch Dashboard**:
    ```bash
    npm run dev
    ```

## Data Integration
The dashboard reads data from `public/results.csv`. When you run a benchmark on the backend:
1.  The Python script saves results to `data/results.csv`.
2.  Copy this file to `frontend/public/results.csv` to update the UI.

```bash
# Example copy command (Windows)
copy ..\data\results.csv .\public\results.csv
```

## Tech Stack
- **Framework**: React 18
- **Build Tool**: Vite
- **Styling**: Vanilla CSS (Terminal Aesthetic)
- **Icons**: Lucide React
