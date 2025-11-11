# Route Optimization System

A complete last-mile delivery route optimization system for logistics companies. Combines geographic clustering with TSP/VRP optimization to reduce delivery distances, balance courier workloads, and calculate business impact.

## Overview

This system optimizes delivery routes through:
1. **Data Loading & Cleaning** - Validates and cleans delivery data from CSV files
2. **Geographic Clustering** - Groups deliveries into balanced PODs (Points of Delivery)
3. **Route Optimization** - Uses OR-Tools to solve TSP/VRP for optimal delivery sequences
4. **Business Metrics** - Calculates cost savings, fuel savings, time savings, and CO2 reduction
5. **Interactive Visualizations** - Generates maps and dashboards showing optimization results

## Features

- ✅ **Web Interface** - Upload CSV and see results instantly with Streamlit
- ✅ **Geographic Clustering** using KMeans, DBSCAN, or Hierarchical algorithms
- ✅ **Fair Workload Distribution** via balanced assignment algorithm
- ✅ **TSP/VRP Optimization** using Google OR-Tools
- ✅ **Real-world Distance Calculations** with geodesic distances
- ✅ **Business Impact Analysis** with fuel, cost, time, and CO2 metrics
- ✅ **Interactive Maps** with Folium and Plotly
- ✅ **Comprehensive Reporting** with CSV exports and JSON metrics
- ✅ **One-Click Deployment** to Streamlit Cloud (FREE)

## Project Structure

```
routing/
├── data/
│   ├── raw/                    # Original CSV files
│   └── processed/              # Cleaned data
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data loading and validation
│   ├── clustering.py           # POD clustering system
│   ├── route_optimizer.py      # TSP/VRP solver
│   ├── metrics.py              # Business metrics calculator
│   └── visualization.py        # Map and chart generation
├── outputs/
│   ├── maps/                   # HTML map visualizations
│   ├── routes/                 # Route CSV exports
│   └── metrics/                # Metrics reports and dashboards
├── .streamlit/
│   └── config.toml             # Streamlit configuration
├── streamlit_app.py            # Web application (NEW!)
├── main.py                     # Command-line interface
├── requirements.txt            # Python dependencies
├── DEPLOYMENT.md               # Deployment guide
└── README.md
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/brzaa/routing.git
cd routing

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 🌐 Web App (Recommended)

**Try the live demo:** `https://[your-app].streamlit.app` (after deployment)

```bash
# Run locally
streamlit run streamlit_app.py
```

Upload your CSV file and see results instantly with interactive visualizations!

### 💻 Command Line

```bash
# Run optimization on your data
python main.py --input data/raw/sleman_depok_10_10.csv --city "Sleman Depok"
```

### Advanced Usage

```bash
# Customize optimization parameters
python main.py \
  --input data/raw/your_data.csv \
  --city "Jakarta Utara" \
  --method hierarchical \
  --pods-per-courier 5 \
  --max-packages 50 \
  --min-packages 15 \
  --time-limit 60 \
  --output custom_output/
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Path to input CSV file (required) | - |
| `--city` | City or area name (required) | - |
| `--method` | Clustering algorithm: `kmeans`, `dbscan`, `hierarchical` | `hierarchical` |
| `--pods-per-courier` | Target PODs per courier | `4` |
| `--max-packages` | Maximum packages per POD | `40` |
| `--min-packages` | Minimum packages per POD | `10` |
| `--time-limit` | Route optimization time limit (seconds) | `30` |
| `--output` | Output directory | `outputs` |

## Input Data Format

Your CSV file must contain these columns:

| Column | Description | Example |
|--------|-------------|---------|
| `AWB_NUMBER` | Unique delivery ID | "AWB123456" |
| `EMPLOYEE_ID` | Courier ID | "EMP001" |
| `NICKNAME` | Courier name | "John Doe" |
| `DO_POD_DELIVER_CODE` | Current POD assignment | "POD_A" |
| `BERATASLI` | Package weight (kg) | 2.5 |
| `SELECTED_LATITUDE` | Delivery latitude | -6.2088 |
| `SELECTED_LONGITUDE` | Delivery longitude | 106.8456 |
| `BRANCH_LATITUDE` | Branch/depot latitude | -6.2000 |
| `BRANCH_LONGITUDE` | Branch/depot longitude | 106.8000 |
| `BRANCH_NAME` | Branch name | "Jakarta Hub" |
| `BRANCH_CODE` | Branch code | "JKT01" |
| `DELIVERY_DATE` | Delivery date | "2024-10-10" |

## Output Files

After running the optimization, you'll find:

### Maps (`outputs/maps/`)
- `clustering_comparison.html` - Before/after POD assignments
- `cluster_*_route.html` - Optimized route for each cluster

### Routes (`outputs/routes/`)
- `optimized_routes.csv` - Detailed route sequences
- `cluster_summary.csv` - Per-cluster statistics
- `*_optimized_pods.csv` - New POD definitions
- `*_optimized_details.csv` - AWB-level assignments
- `*_optimized_courier_summary.csv` - Courier workload summary

### Metrics (`outputs/metrics/`)
- `dashboard.html` - Interactive metrics dashboard
- `courier_workload.html` - Workload comparison chart
- `report.txt` - Comprehensive text report
- `metrics.json` - Structured metrics data

## Example Results

Typical optimization results:

### Clustering Optimization
- **POD Reduction**: 45 → 38 PODs (-15.6%)
- **Workload Balance**: 34% → 12% CV (65% improvement)
- **Fairness Score**: 66% → 88%

### Route Optimization
- **Distance**: 285 km → 198 km
- **Savings**: 87 km (30.5% reduction)

### Business Impact (Daily)
- **Fuel Savings**: 8.7 liters
- **Cost Savings**: Rp 87,000
- **Time Savings**: 2.2 hours
- **CO2 Reduction**: 20 kg

### Annual Projections
- **Annual Savings**: Rp 22,968,000
- **Fuel Saved**: 2,296 liters
- **CO2 Reduced**: 5,281 kg

## Technical Details

### Clustering Algorithm

The system uses geographic clustering with these approaches:

1. **KMeans** - Fast, requires pre-specified cluster count
2. **DBSCAN** - Density-based, handles arbitrary shapes
3. **Hierarchical** - Builds cluster hierarchy, flexible merging

Clusters are assigned to couriers using a greedy balanced assignment algorithm to minimize workload variance.

### Route Optimization

Uses Google OR-Tools constraint solver:

- **TSP** (Traveling Salesman Problem) for single courier routes
- **VRP** (Vehicle Routing Problem) for multi-courier clusters
- **First Solution Strategy**: PATH_CHEAPEST_ARC
- **Local Search**: GUIDED_LOCAL_SEARCH
- **Distance Metric**: Geodesic (real-world distances)

### Business Metrics

Assumptions for Indonesian logistics:
- Fuel efficiency: 10 km/liter
- Fuel cost: Rp 10,000/liter
- Average speed: 40 km/hour
- CO2 emissions: 2.3 kg/liter
- Working days: 22 days/month

## 🚀 Web App Deployment

### Deploy to Streamlit Cloud (FREE)

1. **Fork/Clone this repository** to your GitHub account
2. **Sign up** at https://streamlit.io/cloud (free)
3. **Click "New app"** and select:
   - Repository: `your-username/routing`
   - Branch: `main`
   - Main file: `streamlit_app.py`
4. **Click "Deploy"** and wait 2-5 minutes
5. **Share your URL**: `https://your-app.streamlit.app`

📖 **Full deployment guide**: See [DEPLOYMENT.md](DEPLOYMENT.md)

### Web App Features

- 📤 **Drag & Drop Upload** - Simply upload your CSV file
- ⚙️ **Interactive Controls** - Adjust parameters in real-time
- 🗺️ **Live Visualizations** - See clustering and routes instantly
- 📊 **Business Dashboard** - View KPIs and metrics
- 📥 **Export Results** - Download optimized routes and metrics
- 🚀 **No Setup Required** - Works in any browser

## Development

### Running Tests

```bash
# Test data loading
python -c "from src.data_loader import DataLoader; loader = DataLoader(); df = loader.load_and_clean('data/raw/sleman_depok_10_10.csv')"

# Test full pipeline
python main.py --input data/raw/sleman_depok_10_10.csv --city "Test City"
```

### Project Architecture

```
DataLoader → PODClusteringSystem → RouteOptimizer → MetricsCalculator → Visualization
     ↓              ↓                     ↓                  ↓                ↓
Clean Data → Geographic Clusters → Optimized Routes → Business Impact → Maps/Reports
```

## Use Cases

This system is designed for:

- **Logistics Companies** - Optimize daily delivery routes
- **E-commerce Platforms** - Improve last-mile delivery efficiency
- **Courier Services** - Balance workload among couriers
- **Operations Research** - Portfolio projects demonstrating optimization skills
- **Data Science Projects** - Combine ML clustering with mathematical optimization

## Performance

Expected performance on typical datasets:

| Deliveries | Couriers | Processing Time | Memory Usage |
|------------|----------|-----------------|--------------|
| 100-300    | 5-10     | 30-60 seconds   | < 200 MB     |
| 300-500    | 10-15    | 1-2 minutes     | < 300 MB     |
| 500-1000   | 15-30    | 2-5 minutes     | < 500 MB     |

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'ortools'`
```bash
pip install ortools
```

**Issue**: CSV column missing
- Verify your CSV has all required columns
- Check column names match exactly (case-sensitive)

**Issue**: Optimization takes too long
- Reduce `--time-limit` for faster (less optimal) solutions
- Increase `--max-packages` to create fewer, larger clusters

**Issue**: No improvement shown
- Ensure your current routes aren't already optimal
- Try different clustering methods
- Adjust POD parameters

## Citation

If you use this system in your research or project, please cite:

```
@software{route_optimization_system,
  title = {Route Optimization System for Last-Mile Delivery},
  author = {brzaa},
  year = {2024},
  url = {https://github.com/brzaa/routing}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues, questions, or contributions:
- Open an issue on [GitHub Issues](https://github.com/brzaa/routing/issues)
- Check existing documentation in AGENT.MD
- Review sample data in `data/raw/`

## Acknowledgments

Built with:
- [Google OR-Tools](https://developers.google.com/optimization) for optimization
- [scikit-learn](https://scikit-learn.org/) for clustering
- [Folium](https://python-visualization.github.io/folium/) & [Plotly](https://plotly.com/) for visualization
- [GeoPy](https://geopy.readthedocs.io/) for geographic calculations

---

**Business Context**: Designed for SiCepat (Indonesian logistics) to demonstrate 20-30% distance reduction and fair workload balancing in last-mile delivery operations.
