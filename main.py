#!/usr/bin/env python3
"""
Main Pipeline for Route Optimization System

Last-mile delivery route optimization system that performs:
1. Data loading and cleaning
2. Geographic clustering for workload distribution
3. Route optimization using TSP/VRP
4. Business impact calculation
5. Interactive visualization generation

Usage:
    python main.py --input data/raw/sleman_depok_10_10.csv --city "Sleman Depok"

    python main.py --input data/raw/your_data.csv --city "Your City" --method hierarchical
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import DataLoader
from src.clustering import PODClusteringSystem
from src.route_optimizer import RouteOptimizer
from src.metrics import MetricsCalculator
from src.visualization import create_all_visualizations


def main(
    input_csv: str,
    city_name: str,
    clustering_method: str = 'hierarchical',
    target_pods_per_courier: int = 4,
    max_packages_per_pod: int = 40,
    min_packages_per_pod: int = 10,
    time_limit: int = 30,
    output_dir: str = 'outputs'
):
    """
    Main pipeline for route optimization.

    Args:
        input_csv: Path to input CSV file
        city_name: Name of the city/area
        clustering_method: Clustering algorithm (kmeans, dbscan, hierarchical)
        target_pods_per_courier: Target number of PODs per courier
        max_packages_per_pod: Maximum packages per POD
        min_packages_per_pod: Minimum packages per POD
        time_limit: Time limit for TSP/VRP solving (seconds)
        output_dir: Base directory for all outputs
    """

    print("\n" + "="*80)
    print("LAST-MILE DELIVERY ROUTE OPTIMIZATION SYSTEM")
    print("="*80)
    print(f"City: {city_name}")
    print(f"Input: {input_csv}")
    print(f"Clustering Method: {clustering_method}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    try:
        # ====================================================================
        # STEP 1: Load and Clean Data
        # ====================================================================
        print("\n" + "="*80)
        print("[1/6] LOADING AND CLEANING DATA")
        print("="*80)

        loader = DataLoader()
        df = loader.load_and_clean(input_csv, output_dir='data/processed')

        print(f"\n✅ Step 1 Complete:")
        print(f"   • Loaded {len(df):,} deliveries")
        print(f"   • Couriers: {df['EMPLOYEE_ID'].nunique()}")
        print(f"   • Current PODs: {df['DO_POD_DELIVER_CODE'].nunique()}")

        # ====================================================================
        # STEP 2: Geographic Clustering
        # ====================================================================
        print("\n" + "="*80)
        print("[2/6] RUNNING GEOGRAPHIC CLUSTERING")
        print("="*80)

        clustering_system = PODClusteringSystem(df=df, city_name=city_name)

        # Analyze current state
        clustering_system.analyze_current_state()

        # Create new clusters
        clustering_system.cluster_delivery_points(
            method=clustering_method,
            target_pods_per_courier=target_pods_per_courier,
            max_packages_per_pod=max_packages_per_pod,
            min_packages_per_pod=min_packages_per_pod,
            separate_heavy_packages=True
        )

        # Assign clusters to couriers
        clustering_system.assign_pods_to_couriers(method='balanced')

        # Evaluate clustering results
        clustering_system.evaluate_optimization()

        print(f"\n✅ Step 2 Complete:")
        print(f"   • Created {len(clustering_system.new_pods)} optimized PODs")
        print(f"   • Workload balance improved by {clustering_system.optimization_metrics['improvements']['cv_improvement']:.1f}%")

        # ====================================================================
        # STEP 3: Route Optimization
        # ====================================================================
        print("\n" + "="*80)
        print("[3/6] OPTIMIZING DELIVERY ROUTES")
        print("="*80)

        route_optimizer = RouteOptimizer(clustering_system)
        route_optimizer.solve_all_clusters(time_limit_seconds=time_limit)

        savings = route_optimizer.get_total_distance_savings()

        print(f"\n✅ Step 3 Complete:")
        print(f"   • Optimized {len(route_optimizer.routes)} clusters")
        print(f"   • Distance reduced by {savings['savings_km']:.1f} km ({savings['savings_percent']:.1f}%)")

        # ====================================================================
        # STEP 4: Calculate Metrics
        # ====================================================================
        print("\n" + "="*80)
        print("[4/6] CALCULATING BUSINESS METRICS")
        print("="*80)

        calculator = MetricsCalculator(clustering_system, route_optimizer)
        metrics = calculator.calculate_all_metrics()

        print(f"\n✅ Step 4 Complete:")
        print(f"   • Daily cost savings: Rp {metrics['business_impact']['cost_savings_idr']:,.0f}")
        print(f"   • Annual savings: Rp {metrics['business_impact']['annual_cost_savings_idr']:,.0f}")

        # ====================================================================
        # STEP 5: Generate Visualizations
        # ====================================================================
        print("\n" + "="*80)
        print("[5/6] GENERATING VISUALIZATIONS")
        print("="*80)

        # Create output directories
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{output_dir}/maps").mkdir(parents=True, exist_ok=True)
        Path(f"{output_dir}/routes").mkdir(parents=True, exist_ok=True)
        Path(f"{output_dir}/metrics").mkdir(parents=True, exist_ok=True)

        # Create all visualizations
        viz_outputs = create_all_visualizations(
            clustering_system,
            route_optimizer,
            metrics,
            output_dir
        )

        print(f"\n✅ Step 5 Complete:")
        print(f"   • Created {len(viz_outputs)} visualization sets")

        # ====================================================================
        # STEP 6: Export Results
        # ====================================================================
        print("\n" + "="*80)
        print("[6/6] EXPORTING RESULTS")
        print("="*80)

        # Export clustering assignments
        clustering_system.export_assignments(f"{output_dir}/routes")

        # Export optimized routes
        route_optimizer.export_routes_csv(f"{output_dir}/routes")

        # Export metrics report
        calculator.generate_report(f"{output_dir}/metrics/report.txt")
        calculator.export_metrics_json(f"{output_dir}/metrics/metrics.json")

        print(f"\n✅ Step 6 Complete:")
        print(f"   • All results exported to {output_dir}/")

        # ====================================================================
        # FINAL SUMMARY
        # ====================================================================
        print("\n" + "="*80)
        print("🎉 OPTIMIZATION COMPLETE - SUMMARY")
        print("="*80)

        print(f"\n📊 Clustering Results:")
        print(f"   • PODs: {metrics['clustering_metrics']['pods_before']} → "
              f"{metrics['clustering_metrics']['pods_after']} "
              f"({metrics['clustering_metrics']['pods_change']:+d})")
        print(f"   • Workload Balance: {metrics['clustering_metrics']['cv_improvement']:.1f}% improvement")
        print(f"   • Fairness Score: {metrics['clustering_metrics']['fairness_score_before']:.1f}% → "
              f"{metrics['clustering_metrics']['fairness_score_after']:.1f}%")

        print(f"\n🚗 Routing Results:")
        print(f"   • Distance: {metrics['routing_metrics']['distance_before_km']:.1f} km → "
              f"{metrics['routing_metrics']['distance_after_km']:.1f} km")
        print(f"   • Savings: {metrics['routing_metrics']['savings_km']:.1f} km "
              f"({metrics['routing_metrics']['savings_percent']:.1f}%)")
        print(f"   • Clusters Optimized: {metrics['routing_metrics']['total_clusters']}")

        print(f"\n💰 Business Impact (Daily):")
        print(f"   • Fuel Savings: {metrics['business_impact']['fuel_savings_liters']:.1f} liters")
        print(f"   • Cost Savings: Rp {metrics['business_impact']['cost_savings_idr']:,.0f}")
        print(f"   • Time Savings: {metrics['business_impact']['time_savings_hours']:.1f} hours")
        print(f"   • CO2 Reduction: {metrics['business_impact']['co2_reduction_kg']:.1f} kg")

        print(f"\n💰 Annual Projections:")
        print(f"   • Annual Savings: Rp {metrics['business_impact']['annual_cost_savings_idr']:,.0f}")
        print(f"   • Fuel Saved: {metrics['business_impact']['annual_fuel_savings_liters']:,.0f} liters")
        print(f"   • CO2 Reduced: {metrics['business_impact']['annual_co2_reduction_kg']:,.0f} kg")

        print(f"\n📁 Output Locations:")
        print(f"   • Maps:    {output_dir}/maps/")
        print(f"   • Routes:  {output_dir}/routes/")
        print(f"   • Metrics: {output_dir}/metrics/")

        print("\n" + "="*80)
        print("✅ ALL PROCESSING COMPLETE!")
        print("="*80)

        return {
            'success': True,
            'metrics': metrics,
            'outputs': viz_outputs
        }

    except Exception as e:
        print("\n" + "="*80)
        print("❌ ERROR OCCURRED")
        print("="*80)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print("="*80)

        return {
            'success': False,
            'error': str(e)
        }


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Route Optimization System for Last-Mile Delivery',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python main.py --input data/raw/sleman_depok_10_10.csv --city "Sleman Depok"

  # With custom parameters
  python main.py --input data/raw/your_data.csv --city "Jakarta" \\
    --method hierarchical --pods-per-courier 5 --time-limit 60

  # Minimal POD count
  python main.py --input data/raw/your_data.csv --city "Bandung" \\
    --max-packages 50 --min-packages 15
        """
    )

    # Required arguments
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input CSV file (required columns: AWB_NUMBER, EMPLOYEE_ID, etc.)'
    )

    parser.add_argument(
        '--city',
        type=str,
        required=True,
        help='City or area name for the deliveries'
    )

    # Optional arguments
    parser.add_argument(
        '--method',
        type=str,
        default='hierarchical',
        choices=['kmeans', 'dbscan', 'hierarchical'],
        help='Clustering algorithm (default: hierarchical)'
    )

    parser.add_argument(
        '--pods-per-courier',
        type=int,
        default=4,
        help='Target number of PODs per courier (default: 4)'
    )

    parser.add_argument(
        '--max-packages',
        type=int,
        default=40,
        help='Maximum packages per POD (default: 40)'
    )

    parser.add_argument(
        '--min-packages',
        type=int,
        default=10,
        help='Minimum packages per POD (default: 10)'
    )

    parser.add_argument(
        '--time-limit',
        type=int,
        default=30,
        help='Time limit for route optimization in seconds (default: 30)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='outputs',
        help='Output directory for results (default: outputs)'
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    result = main(
        input_csv=args.input,
        city_name=args.city,
        clustering_method=args.method,
        target_pods_per_courier=args.pods_per_courier,
        max_packages_per_pod=args.max_packages,
        min_packages_per_pod=args.min_packages,
        time_limit=args.time_limit,
        output_dir=args.output
    )

    # Exit with appropriate code
    sys.exit(0 if result['success'] else 1)
