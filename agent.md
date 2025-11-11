# ROUTING OPTIMIZATION SYSTEM - AGENT SPECIFICATION

## PROJECT OVERVIEW

Build a complete last-mile delivery route optimization system that:
1. Takes delivery data from local CSV files (extracted from Snowflake)
2. Performs geographic clustering for fair workload distribution
3. Optimizes delivery routes within each cluster using VRP/TSP
4. Calculates business impact (distance savings, cost reduction, time savings)
5. Generates visual reports and interactive maps

**Business Context:**
- Company: SiCepat (Indonesian logistics company)
- Problem: Inefficient courier workload distribution and sub-optimal delivery routes
- Goal: Demonstrate 20-30% distance reduction and fair workload balancing
- Use Case: Portfolio project to demonstrate data science + operations research skills

## DATA SOURCES

### Input Data Location
**File:** `sleman_depok_10_10.csv` (or similar city CSV)

**Required Columns:**
- `AWB_NUMBER` - Unique delivery ID
- `EMPLOYEE_ID` - Courier ID
- `NICKNAME` - Courier name
- `DO_POD_DELIVER_CODE` - Current POD assignment
- `BERATASLI` - Package weight (kg)
- `SELECTED_LATITUDE` - Delivery latitude
- `SELECTED_LONGITUDE` - Delivery longitude
- `BRANCH_LATITUDE` - Branch/depot latitude
- `BRANCH_LONGITUDE` - Branch/depot longitude
- `BRANCH_NAME` - Branch name (e.g., "Jakut Ancol Barat")
- `BRANCH_CODE` - Branch code
- `DELIVERY_DATE` - Date of delivery

**Data Characteristics:**
- Single city data (e.g., Jakarta Utara)
- One month of deliveries (e.g., October 2024)
- 300-1000 deliveries typical
- 10-30 couriers
- Real coordinates in Indonesia

## EXISTING CODE TO INTEGRATE

You have two Python files that need to be integrated:

### 1. clustering.py
Contains `PODClusteringSystem` class with these key features:
- Geographic clustering using KMeans/DBSCAN/Hierarchical
- Fair workload distribution via Hungarian algorithm
- Branch location detection
- Delivery categorization by vehicle type (motorcycle vs car based on weight)
- Visualization with Folium maps
- Export capabilities

**Key methods:**
- `__init__(df, city_name, branch_location)`
- `cluster_delivery_points(method, target_pods_per_courier, max_packages_per_pod, ...)`
- `assign_pods_to_couriers(method='balanced')`
- `evaluate_optimization()`
- `create_comparison_maps(output_dir)`
- `export_assignments(output_dir)`

### 2. tsp.py
Contains VRP/TSP solver using Google OR-Tools with:
- Distance calculation using geodesic (real-world distances)
- VRP for clusters with multiple couriers
- TSP for single courier routes
- Capacity constraints based on package weight
- Route visualization with Plotly

**Key components:**
- OR-Tools routing solver (pywrapcp)
- Geodesic distance calculations
- Route extraction and visualization
- Distance comparison (before vs after)

**Problem:** These are separate, not connected. Your job is to integrate them.

## PROJECT STRUCTURE

Create this structure:

```
routing-optimizer/
├── data/
│   ├── raw/                    # Original CSV from Snowflake
│   │   └── jakarta_oct_deliveries.csv
│   └── processed/              # Cleaned data
│       └── jakarta_clean.csv
├── src/
│   ├── __init__.py
│   ├── clustering.py           # Existing PODClusteringSystem
│   ├── route_optimizer.py      # NEW: TSP integration class
│   ├── data_loader.py          # NEW: Data cleaning
│   ├── metrics.py              # NEW: Savings calculation
│   └── visualization.py        # NEW: Report generation
├── outputs/
│   ├── maps/                   # HTML map visualizations
│   ├── routes/                 # Route CSVs
│   └── metrics/                # Metrics reports
├── main.py                     # Entry point
├── requirements.txt
└── README.md
```

## REQUIREMENTS

### Phase 1: Data Loading & Cleaning

**File:** `src/data_loader.py`

Create a `DataLoader` class that:
1. Reads CSV from `data/raw/`
2. Validates required columns exist
3. Removes null coordinates
4. Removes duplicate AWB_NUMBER (keep first)
5. Validates coordinates are within Indonesia bounds:
   - Latitude: -11 to 6
   - Longitude: 95 to 141
6. Adds `delivery_id` column (sequential)
7. Reports cleaning statistics
8. Saves to `data/processed/`

**Key method:**
```python
def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load, clean, validate delivery data"""
    # Implementation
```

### Phase 2: Route Optimizer Integration

**File:** `src/route_optimizer.py`

Create a `RouteOptimizer` class that wraps the TSP/VRP logic:

```python
class RouteOptimizer:
    def __init__(self, clustering_system: PODClusteringSystem):
        """
        Initialize with clustering results
        
        Args:
            clustering_system: PODClusteringSystem with clusters already created
        """
        self.clustering_system = clustering_system
        self.branch_location = clustering_system.branch_location
        self.routes = {}  # Store optimized routes per cluster
        self.metrics = {}  # Store distance metrics per cluster
    
    def solve_all_clusters(self, time_limit_seconds: int = 30):
        """
        Run VRP/TSP optimization for each cluster
        
        For each cluster:
        - If multiple couriers: solve as VRP
        - If single courier: solve as TSP
        - Store optimized routes
        - Calculate distances
        """
        pass
    
    def _solve_cluster_vrp(self, cluster_id: int, time_limit: int):
        """Solve VRP for a cluster with multiple couriers"""
        # Use OR-Tools routing solver
        # Similar to your existing tsp.py logic
        pass
    
    def _solve_single_courier_tsp(self, cluster_id: int, courier_id: str, time_limit: int):
        """Solve TSP for single courier"""
        pass
    
    def get_total_distance_savings(self) -> Dict:
        """Calculate overall distance improvement"""
        return {
            'baseline_distance_km': self._calculate_baseline_distance(),
            'optimized_distance_km': self._sum_optimized_routes(),
            'savings_km': ...,
            'savings_percent': ...
        }
    
    def _calculate_baseline_distance(self) -> float:
        """
        Estimate current distance by summing:
        branch -> delivery1 -> branch -> delivery2 -> branch, etc.
        (Assumes couriers return to branch after each delivery)
        """
        pass
    
    def export_routes_csv(self, output_dir: str):
        """Export optimized routes to CSV files"""
        pass
```

**Technical Requirements:**
- Use `ortools.constraint_solver` for routing
- Use `geopy.distance.geodesic` for real-world distances
- Convert distances to integers (meters) for OR-Tools
- Set reasonable time limits (10-30 seconds per cluster)
- Handle edge cases (single delivery, outliers)

### Phase 3: Metrics Calculation

**File:** `src/metrics.py`

Create a `MetricsCalculator` class:

```python
class MetricsCalculator:
    def __init__(self, clustering_system: PODClusteringSystem, 
                 route_optimizer: RouteOptimizer):
        self.clustering = clustering_system
        self.routing = route_optimizer
    
    def calculate_all_metrics(self) -> Dict:
        """
        Calculate comprehensive metrics
        
        Returns:
            {
                'clustering_metrics': {...},
                'routing_metrics': {...},
                'business_impact': {...}
            }
        """
        pass
    
    def _clustering_metrics(self) -> Dict:
        """
        Workload distribution metrics:
        - Packages per courier (before/after)
        - Coefficient of variation (CV)
        - Fairness improvement %
        - PODs before/after
        """
        pass
    
    def _routing_metrics(self) -> Dict:
        """
        Route optimization metrics:
        - Total distance before (km)
        - Total distance after (km)
        - Distance savings (km and %)
        - Average route length
        """
        pass
    
    def _business_impact(self) -> Dict:
        """
        Business value metrics:
        - Fuel savings (liters) - assume 10 km/liter
        - Cost savings (IDR) - assume 10,000 IDR/liter
        - Time savings (hours) - assume 40 km/hour avg speed
        - Annual projection (assuming 22 working days/month)
        - CO2 reduction (kg) - assume 2.3 kg CO2/liter
        """
        pass
    
    def generate_report(self, output_path: str):
        """Generate text report with all metrics"""
        pass
    
    def export_metrics_json(self, output_path: str):
        """Export metrics as JSON"""
        pass
```

### Phase 4: Visualization

**File:** `src/visualization.py`

Create visualization functions:

```python
def create_clustering_comparison_map(
    clustering_system: PODClusteringSystem,
    output_path: str
):
    """
    Create side-by-side comparison map:
    - Left: Current POD assignments
    - Right: Optimized POD assignments
    
    Show:
    - Delivery points (colored by courier)
    - Branch location (red star)
    - POD boundaries (convex hulls)
    - Legend with courier workload
    """
    pass

def create_route_optimization_maps(
    route_optimizer: RouteOptimizer,
    output_dir: str
):
    """
    Create one map per cluster showing:
    - Branch location
    - Delivery points
    - Optimized TSP/VRP route (lines connecting stops)
    - Route distance metric
    
    Use Plotly or Folium for interactive maps
    """
    pass

def create_metrics_dashboard(
    metrics: Dict,
    output_path: str
):
    """
    Create HTML dashboard with:
    - KPI cards (savings %, distance reduction, etc.)
    - Bar charts (before/after comparison)
    - Workload distribution charts
    - Route efficiency metrics
    
    Use Plotly for interactivity
    """
    pass
```

### Phase 5: Main Pipeline

**File:** `main.py`

Create the main execution script:

```python
import argparse
from pathlib import Path
from src.data_loader import DataLoader
from src.clustering import PODClusteringSystem
from src.route_optimizer import RouteOptimizer
from src.metrics import MetricsCalculator
from src.visualization import (
    create_clustering_comparison_map,
    create_route_optimization_maps,
    create_metrics_dashboard
)

def main(
    input_csv: str,
    city_name: str,
    clustering_method: str = 'hierarchical',
    target_pods_per_courier: int = 4,
    max_packages_per_pod: int = 40,
    min_packages_per_pod: int = 10
):
    """
    Main pipeline for route optimization
    
    Steps:
    1. Load and clean data
    2. Run clustering optimization
    3. Optimize routes within clusters
    4. Calculate metrics
    5. Generate visualizations
    6. Export results
    """
    
    print("="*80)
    print("LAST-MILE DELIVERY ROUTE OPTIMIZATION")
    print("="*80)
    
    # 1. Load data
    print("\n[1/6] Loading and cleaning data...")
    loader = DataLoader()
    df = loader.load_and_clean(input_csv)
    print(f"✓ Loaded {len(df)} deliveries")
    print(f"✓ Couriers: {df['EMPLOYEE_ID'].nunique()}")
    
    # 2. Run clustering
    print("\n[2/6] Running geographic clustering...")
    clustering_system = PODClusteringSystem(
        df=df,
        city_name=city_name
    )
    clustering_system.analyze_current_state()
    clustering_system.cluster_delivery_points(
        method=clustering_method,
        target_pods_per_courier=target_pods_per_courier,
        max_packages_per_pod=max_packages_per_pod,
        min_packages_per_pod=min_packages_per_pod
    )
    clustering_system.assign_pods_to_couriers(method='balanced')
    clustering_system.evaluate_optimization()
    print("✓ Clustering complete")
    
    # 3. Optimize routes
    print("\n[3/6] Optimizing delivery routes...")
    route_optimizer = RouteOptimizer(clustering_system)
    route_optimizer.solve_all_clusters(time_limit_seconds=30)
    print("✓ Route optimization complete")
    
    # 4. Calculate metrics
    print("\n[4/6] Calculating metrics...")
    calculator = MetricsCalculator(clustering_system, route_optimizer)
    metrics = calculator.calculate_all_metrics()
    print("✓ Metrics calculated")
    
    # 5. Generate visualizations
    print("\n[5/6] Generating visualizations...")
    Path('outputs/maps').mkdir(parents=True, exist_ok=True)
    Path('outputs/routes').mkdir(parents=True, exist_ok=True)
    Path('outputs/metrics').mkdir(parents=True, exist_ok=True)
    
    create_clustering_comparison_map(
        clustering_system,
        'outputs/maps/clustering_comparison.html'
    )
    create_route_optimization_maps(
        route_optimizer,
        'outputs/maps'
    )
    create_metrics_dashboard(
        metrics,
        'outputs/metrics/dashboard.html'
    )
    print("✓ Visualizations created")
    
    # 6. Export results
    print("\n[6/6] Exporting results...")
    clustering_system.export_assignments('outputs/routes')
    route_optimizer.export_routes_csv('outputs/routes')
    calculator.generate_report('outputs/metrics/report.txt')
    calculator.export_metrics_json('outputs/metrics/metrics.json')
    print("✓ Results exported")
    
    # Summary
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE - SUMMARY")
    print("="*80)
    print(f"\n📊 Clustering:")
    print(f"  • PODs: {metrics['clustering_metrics']['pods_before']} → {metrics['clustering_metrics']['pods_after']}")
    print(f"  • Workload balance improvement: {metrics['clustering_metrics']['cv_improvement']:.1f}%")
    
    print(f"\n🚗 Routing:")
    print(f"  • Distance: {metrics['routing_metrics']['distance_before_km']:.1f} km → {metrics['routing_metrics']['distance_after_km']:.1f} km")
    print(f"  • Savings: {metrics['routing_metrics']['savings_km']:.1f} km ({metrics['routing_metrics']['savings_percent']:.1f}%)")
    
    print(f"\n💰 Business Impact:")
    print(f"  • Fuel savings: {metrics['business_impact']['fuel_savings_liters']:.1f} liters/day")
    print(f"  • Cost savings: {metrics['business_impact']['cost_savings_idr']:,.0f} IDR/day")
    print(f"  • Annual projection: {metrics['business_impact']['annual_savings_idr']:,.0f} IDR/year")
    
    print(f"\n📁 Outputs saved to:")
    print(f"  • Maps: outputs/maps/")
    print(f"  • Routes: outputs/routes/")
    print(f"  • Metrics: outputs/metrics/")
    print("\n" + "="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Route Optimization System')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--city', type=str, required=True, help='City name')
    parser.add_argument('--method', type=str, default='hierarchical', 
                       choices=['kmeans', 'dbscan', 'hierarchical'])
    
    args = parser.parse_args()
    main(input_csv=args.input, city_name=args.city, clustering_method=args.method)
```

## DEPENDENCIES

**File:** `requirements.txt`

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.11.0
folium>=0.14.0
plotly>=5.17.0
geopy>=2.4.0
ortools>=9.7.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

## SUCCESS CRITERIA

### Functional Requirements
✅ System loads CSV data and validates it
✅ Clustering produces balanced workload distribution
✅ Route optimization runs without errors
✅ Metrics show measurable improvements:
   - Distance reduction: 15-35%
   - Workload CV reduction: 40-70%
✅ Visualizations are clear and professional
✅ All outputs export correctly

### Code Quality
✅ Clean class structure (no notebook-style code)
✅ Proper error handling
✅ Logging/progress updates
✅ Type hints where appropriate
✅ Docstrings for all classes/methods

### Output Quality
✅ Interactive HTML maps that render in browser
✅ CSV exports with proper headers
✅ JSON metrics with correct structure
✅ Text report that's readable and well-formatted

## VALIDATION STEPS

After implementation, validate:

1. **Data Loading**
   ```python
   loader = DataLoader()
   df = loader.load_and_clean('data/raw/jakarta_oct_deliveries.csv')
   assert len(df) > 0
   assert df['SELECTED_LATITUDE'].between(-11, 6).all()
   ```

2. **Clustering**
   ```python
   system = PODClusteringSystem(df, 'Jakarta')
   system.cluster_delivery_points()
   system.assign_pods_to_couriers()
   assert system.optimization_metrics['improvements']['cv_improvement'] > 0
   ```

3. **Routing**
   ```python
   optimizer = RouteOptimizer(system)
   optimizer.solve_all_clusters()
   savings = optimizer.get_total_distance_savings()
   assert savings['savings_percent'] > 10
   ```

4. **Outputs**
   ```bash
   ls outputs/maps/*.html  # Should have multiple HTML files
   ls outputs/routes/*.csv  # Should have route CSVs
   ls outputs/metrics/*.json  # Should have metrics.json
   ```

## NOTES FOR AGENT

### Key Integration Points
- The `PODClusteringSystem.df` will have a `new_pod_label` column after clustering
- This column identifies which cluster each delivery belongs to
- Use this to group deliveries for TSP/VRP optimization
- The `courier_assignments` attribute maps clusters to couriers

### OR-Tools Tips
- Use `PATH_CHEAPEST_ARC` as first solution strategy
- Use `GUIDED_LOCAL_SEARCH` for local search metaheuristic
- Set time limit of 10-30 seconds per cluster
- Convert distances to integer meters for OR-Tools
- Handle depot (branch) as index 0

### Common Pitfalls to Avoid
- Don't recalculate coordinates - they're already in the data
- Don't re-implement clustering - use the existing class
- Remember to handle edge cases (single delivery, empty clusters)
- Test with small dataset first (50 deliveries)
- Validate all file paths exist before writing

### Performance Expectations
- 500 deliveries should process in < 2 minutes
- Most time spent on TSP solving (expected)
- Memory usage should be < 500MB

## EXAMPLE USAGE

Once implemented:

```bash
# Install dependencies
pip install -r requirements.txt

# Run optimization
python main.py \
  --input data/raw/jakarta_oct_deliveries.csv \
  --city "Jakarta Utara" \
  --method hierarchical

# View results
open outputs/maps/clustering_comparison.html
open outputs/metrics/dashboard.html
cat outputs/metrics/report.txt
```

## DELIVERABLES

When complete, you should have:
1. ✅ Working Python package with all modules
2. ✅ README.md with usage instructions
3. ✅ Sample outputs in outputs/ directory
4. ✅ requirements.txt
5. ✅ Validated on at least one real dataset

---

**END OF SPECIFICATION**

This agent.md provides everything needed to build a production-quality route optimization system. Follow it sequentially, validate at each step, and you'll have a portfolio-worthy project.