"""
Metrics Calculation Module

Calculates comprehensive optimization metrics including:
- Clustering workload distribution metrics
- Route optimization distance metrics
- Business impact (cost, fuel, time, CO2 savings)
"""

import pandas as pd
import numpy as np
import json
from typing import Dict
from pathlib import Path


class MetricsCalculator:
    """
    Calculate comprehensive metrics for optimization results.

    Combines clustering and routing metrics to provide business impact analysis.
    """

    # Business constants for Indonesia logistics
    FUEL_EFFICIENCY_KM_PER_LITER = 10  # Average motorcycle/car efficiency
    FUEL_COST_IDR_PER_LITER = 10000  # Approximate fuel cost in IDR
    AVG_SPEED_KM_PER_HOUR = 40  # Average delivery speed in urban areas
    CO2_KG_PER_LITER = 2.3  # CO2 emissions per liter of fuel
    WORKING_DAYS_PER_MONTH = 22  # Typical working days

    def __init__(self, clustering_system, route_optimizer):
        """
        Initialize MetricsCalculator.

        Args:
            clustering_system: PODClusteringSystem instance with clustering results
            route_optimizer: RouteOptimizer instance with route optimization results
        """
        self.clustering = clustering_system
        self.routing = route_optimizer
        self.all_metrics = None

    def calculate_all_metrics(self) -> Dict:
        """
        Calculate comprehensive metrics.

        Returns:
            Dictionary containing all metrics:
            - clustering_metrics: Workload distribution metrics
            - routing_metrics: Distance optimization metrics
            - business_impact: Business value metrics
        """
        print("\n" + "="*80)
        print("METRICS CALCULATION")
        print("="*80)

        clustering_metrics = self._clustering_metrics()
        routing_metrics = self._routing_metrics()
        business_impact = self._business_impact(routing_metrics)

        self.all_metrics = {
            'clustering_metrics': clustering_metrics,
            'routing_metrics': routing_metrics,
            'business_impact': business_impact,
            'summary': self._create_summary(clustering_metrics, routing_metrics, business_impact)
        }

        self._print_metrics_summary()

        return self.all_metrics

    def _clustering_metrics(self) -> Dict:
        """
        Calculate workload distribution metrics.

        Returns:
            Dictionary of clustering metrics
        """
        # Before metrics (from current state)
        current_couriers = self.clustering.current_state['couriers']
        packages_before = current_couriers['total_packages'].values
        cv_before = self.clustering.current_state['metrics']['cv_packages']
        max_min_ratio_before = self.clustering.current_state['metrics']['max_min_ratio']

        # After metrics (from optimized assignments)
        packages_after = self.clustering.courier_assignments['packages'].values
        cv_after = (packages_after.std() / packages_after.mean()) * 100 if packages_after.mean() > 0 else 0
        max_min_ratio_after = (packages_after.max() / packages_after.min()) if packages_after.min() > 0 else 0

        # Improvements
        cv_improvement = ((cv_before - cv_after) / cv_before * 100) if cv_before > 0 else 0
        ratio_improvement = ((max_min_ratio_before - max_min_ratio_after) / max_min_ratio_before * 100) \
            if max_min_ratio_before > 0 else 0

        return {
            'pods_before': self.clustering.current_state['metrics']['n_pods'],
            'pods_after': len(self.clustering.new_pods),
            'pods_change': len(self.clustering.new_pods) - self.clustering.current_state['metrics']['n_pods'],
            'pods_change_percent': ((len(self.clustering.new_pods) -
                                   self.clustering.current_state['metrics']['n_pods']) /
                                  self.clustering.current_state['metrics']['n_pods'] * 100)
                                  if self.clustering.current_state['metrics']['n_pods'] > 0 else 0,
            'packages_per_courier_before': {
                'mean': packages_before.mean(),
                'std': packages_before.std(),
                'min': packages_before.min(),
                'max': packages_before.max(),
                'cv': cv_before
            },
            'packages_per_courier_after': {
                'mean': packages_after.mean(),
                'std': packages_after.std(),
                'min': packages_after.min(),
                'max': packages_after.max(),
                'cv': cv_after
            },
            'max_min_ratio_before': max_min_ratio_before,
            'max_min_ratio_after': max_min_ratio_after,
            'cv_improvement': cv_improvement,
            'ratio_improvement': ratio_improvement,
            'fairness_score_before': 100 - cv_before,  # Higher is better
            'fairness_score_after': 100 - cv_after,
            'fairness_improvement': cv_improvement
        }

    def _routing_metrics(self) -> Dict:
        """
        Calculate route optimization metrics.

        Returns:
            Dictionary of routing metrics
        """
        distance_savings = self.routing.get_total_distance_savings()

        # Per-cluster statistics
        cluster_distances = [m['optimized_distance_meters'] for m in self.routing.metrics.values()]
        cluster_deliveries = [m['num_deliveries'] for m in self.routing.metrics.values()]

        return {
            'distance_before_km': distance_savings['baseline_distance_km'],
            'distance_after_km': distance_savings['optimized_distance_km'],
            'savings_km': distance_savings['savings_km'],
            'savings_percent': distance_savings['savings_percent'],
            'distance_before_meters': distance_savings['baseline_distance_meters'],
            'distance_after_meters': distance_savings['optimized_distance_meters'],
            'savings_meters': distance_savings['savings_meters'],
            'total_clusters': len(self.routing.routes),
            'avg_cluster_distance_km': np.mean(cluster_distances) / 1000 if cluster_distances else 0,
            'avg_deliveries_per_cluster': np.mean(cluster_deliveries) if cluster_deliveries else 0,
            'total_deliveries': sum(cluster_deliveries),
            'optimization_success_rate': sum(1 for s in self.routing.optimization_status.values()
                                            if s == 'success') / len(self.routing.optimization_status) * 100
                                           if self.routing.optimization_status else 0
        }

    def _business_impact(self, routing_metrics: Dict) -> Dict:
        """
        Calculate business value metrics.

        Args:
            routing_metrics: Dictionary of routing metrics

        Returns:
            Dictionary of business impact metrics
        """
        savings_km = routing_metrics['savings_km']

        # Fuel savings
        fuel_savings_liters = savings_km / self.FUEL_EFFICIENCY_KM_PER_LITER

        # Cost savings
        cost_savings_idr = fuel_savings_liters * self.FUEL_COST_IDR_PER_LITER

        # Time savings
        time_savings_hours = savings_km / self.AVG_SPEED_KM_PER_HOUR

        # CO2 reduction
        co2_reduction_kg = fuel_savings_liters * self.CO2_KG_PER_LITER

        # Annual projections (assuming consistent daily deliveries)
        days_per_year = self.WORKING_DAYS_PER_MONTH * 12

        return {
            # Daily metrics
            'fuel_savings_liters': fuel_savings_liters,
            'cost_savings_idr': cost_savings_idr,
            'time_savings_hours': time_savings_hours,
            'co2_reduction_kg': co2_reduction_kg,

            # Monthly projections
            'monthly_fuel_savings_liters': fuel_savings_liters * self.WORKING_DAYS_PER_MONTH,
            'monthly_cost_savings_idr': cost_savings_idr * self.WORKING_DAYS_PER_MONTH,
            'monthly_time_savings_hours': time_savings_hours * self.WORKING_DAYS_PER_MONTH,
            'monthly_co2_reduction_kg': co2_reduction_kg * self.WORKING_DAYS_PER_MONTH,

            # Annual projections
            'annual_fuel_savings_liters': fuel_savings_liters * days_per_year,
            'annual_cost_savings_idr': cost_savings_idr * days_per_year,
            'annual_time_savings_hours': time_savings_hours * days_per_year,
            'annual_co2_reduction_kg': co2_reduction_kg * days_per_year,

            # Constants used
            'fuel_efficiency_km_per_liter': self.FUEL_EFFICIENCY_KM_PER_LITER,
            'fuel_cost_idr_per_liter': self.FUEL_COST_IDR_PER_LITER,
            'avg_speed_km_per_hour': self.AVG_SPEED_KM_PER_HOUR,
            'co2_kg_per_liter': self.CO2_KG_PER_LITER
        }

    def _create_summary(self, clustering_metrics: Dict, routing_metrics: Dict,
                       business_impact: Dict) -> Dict:
        """Create executive summary of key metrics."""
        return {
            'optimization_type': 'Clustering + Route Optimization',
            'city': self.clustering.city_name,
            'total_deliveries': routing_metrics['total_deliveries'],
            'num_couriers': len(self.clustering.courier_assignments),
            'key_improvements': {
                'workload_balance': f"{clustering_metrics['cv_improvement']:.1f}%",
                'distance_reduction': f"{routing_metrics['savings_percent']:.1f}%",
                'cost_savings_per_day': f"Rp {business_impact['cost_savings_idr']:,.0f}",
                'annual_savings': f"Rp {business_impact['annual_cost_savings_idr']:,.0f}"
            }
        }

    def _print_metrics_summary(self) -> None:
        """Print comprehensive metrics summary."""
        cm = self.all_metrics['clustering_metrics']
        rm = self.all_metrics['routing_metrics']
        bi = self.all_metrics['business_impact']

        print("\n" + "-"*80)
        print("📊 CLUSTERING METRICS")
        print("-"*80)
        print(f"  PODs: {cm['pods_before']} → {cm['pods_after']} ({cm['pods_change']:+d}, {cm['pods_change_percent']:+.1f}%)")
        print(f"  Workload Balance (CV): {cm['packages_per_courier_before']['cv']:.1f}% → "
              f"{cm['packages_per_courier_after']['cv']:.1f}% ({cm['cv_improvement']:+.1f}% improvement)")
        print(f"  Max/Min Ratio: {cm['max_min_ratio_before']:.2f}x → {cm['max_min_ratio_after']:.2f}x")
        print(f"  Fairness Score: {cm['fairness_score_before']:.1f}% → {cm['fairness_score_after']:.1f}%")

        print("\n" + "-"*80)
        print("🚗 ROUTING METRICS")
        print("-"*80)
        print(f"  Distance: {rm['distance_before_km']:.1f} km → {rm['distance_after_km']:.1f} km")
        print(f"  Savings: {rm['savings_km']:.1f} km ({rm['savings_percent']:.1f}%)")
        print(f"  Clusters Optimized: {rm['total_clusters']}")
        print(f"  Optimization Success Rate: {rm['optimization_success_rate']:.1f}%")
        print(f"  Avg Distance/Cluster: {rm['avg_cluster_distance_km']:.1f} km")

        print("\n" + "-"*80)
        print("💰 BUSINESS IMPACT (Per Day)")
        print("-"*80)
        print(f"  Fuel Savings: {bi['fuel_savings_liters']:.1f} liters")
        print(f"  Cost Savings: Rp {bi['cost_savings_idr']:,.0f}")
        print(f"  Time Savings: {bi['time_savings_hours']:.1f} hours")
        print(f"  CO2 Reduction: {bi['co2_reduction_kg']:.1f} kg")

        print("\n" + "-"*80)
        print("📅 ANNUAL PROJECTIONS (22 days/month)")
        print("-"*80)
        print(f"  Fuel Savings: {bi['annual_fuel_savings_liters']:,.0f} liters/year")
        print(f"  Cost Savings: Rp {bi['annual_cost_savings_idr']:,.0f}/year")
        print(f"  Time Savings: {bi['annual_time_savings_hours']:,.0f} hours/year")
        print(f"  CO2 Reduction: {bi['annual_co2_reduction_kg']:,.0f} kg/year")

    def generate_report(self, output_path: str) -> None:
        """
        Generate text report with all metrics.

        Args:
            output_path: Path to save the report file
        """
        if self.all_metrics is None:
            self.calculate_all_metrics()

        cm = self.all_metrics['clustering_metrics']
        rm = self.all_metrics['routing_metrics']
        bi = self.all_metrics['business_impact']

        report = []
        report.append("="*80)
        report.append("ROUTE OPTIMIZATION SYSTEM - COMPREHENSIVE REPORT")
        report.append("="*80)
        report.append(f"\nCity: {self.clustering.city_name}")
        report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Deliveries: {rm['total_deliveries']}")
        report.append(f"Number of Couriers: {len(self.clustering.courier_assignments)}")

        report.append("\n" + "="*80)
        report.append("1. CLUSTERING OPTIMIZATION")
        report.append("="*80)
        report.append(f"\nPOD Configuration:")
        report.append(f"  Before: {cm['pods_before']} PODs")
        report.append(f"  After:  {cm['pods_after']} PODs")
        report.append(f"  Change: {cm['pods_change']:+d} ({cm['pods_change_percent']:+.1f}%)")

        report.append(f"\nWorkload Distribution (Packages per Courier):")
        report.append(f"  Before: Mean={cm['packages_per_courier_before']['mean']:.1f}, "
                     f"Std={cm['packages_per_courier_before']['std']:.1f}, "
                     f"Min={cm['packages_per_courier_before']['min']}, "
                     f"Max={cm['packages_per_courier_before']['max']}")
        report.append(f"  After:  Mean={cm['packages_per_courier_after']['mean']:.1f}, "
                     f"Std={cm['packages_per_courier_after']['std']:.1f}, "
                     f"Min={cm['packages_per_courier_after']['min']}, "
                     f"Max={cm['packages_per_courier_after']['max']}")

        report.append(f"\nBalance Metrics:")
        report.append(f"  Coefficient of Variation: {cm['packages_per_courier_before']['cv']:.1f}% → "
                     f"{cm['packages_per_courier_after']['cv']:.1f}% "
                     f"({cm['cv_improvement']:+.1f}% improvement)")
        report.append(f"  Max/Min Ratio: {cm['max_min_ratio_before']:.2f}x → "
                     f"{cm['max_min_ratio_after']:.2f}x")
        report.append(f"  Fairness Score: {cm['fairness_score_before']:.1f}% → "
                     f"{cm['fairness_score_after']:.1f}%")

        report.append("\n" + "="*80)
        report.append("2. ROUTE OPTIMIZATION")
        report.append("="*80)
        report.append(f"\nDistance Metrics:")
        report.append(f"  Current Distance:   {rm['distance_before_km']:,.1f} km")
        report.append(f"  Optimized Distance: {rm['distance_after_km']:,.1f} km")
        report.append(f"  Distance Savings:   {rm['savings_km']:,.1f} km ({rm['savings_percent']:.1f}%)")

        report.append(f"\nOptimization Details:")
        report.append(f"  Total Clusters: {rm['total_clusters']}")
        report.append(f"  Success Rate: {rm['optimization_success_rate']:.1f}%")
        report.append(f"  Avg Cluster Distance: {rm['avg_cluster_distance_km']:.1f} km")
        report.append(f"  Avg Deliveries/Cluster: {rm['avg_deliveries_per_cluster']:.1f}")

        report.append("\n" + "="*80)
        report.append("3. BUSINESS IMPACT")
        report.append("="*80)
        report.append(f"\nDaily Savings:")
        report.append(f"  Fuel:  {bi['fuel_savings_liters']:.1f} liters")
        report.append(f"  Cost:  Rp {bi['cost_savings_idr']:,.0f}")
        report.append(f"  Time:  {bi['time_savings_hours']:.1f} hours")
        report.append(f"  CO2:   {bi['co2_reduction_kg']:.1f} kg")

        report.append(f"\nMonthly Projections ({self.WORKING_DAYS_PER_MONTH} working days):")
        report.append(f"  Fuel:  {bi['monthly_fuel_savings_liters']:,.0f} liters")
        report.append(f"  Cost:  Rp {bi['monthly_cost_savings_idr']:,.0f}")
        report.append(f"  Time:  {bi['monthly_time_savings_hours']:,.0f} hours")
        report.append(f"  CO2:   {bi['monthly_co2_reduction_kg']:,.0f} kg")

        report.append(f"\nAnnual Projections ({self.WORKING_DAYS_PER_MONTH * 12} working days):")
        report.append(f"  Fuel:  {bi['annual_fuel_savings_liters']:,.0f} liters")
        report.append(f"  Cost:  Rp {bi['annual_cost_savings_idr']:,.0f}")
        report.append(f"  Time:  {bi['annual_time_savings_hours']:,.0f} hours")
        report.append(f"  CO2:   {bi['annual_co2_reduction_kg']:,.0f} kg")

        report.append(f"\nAssumptions:")
        report.append(f"  • Fuel efficiency: {bi['fuel_efficiency_km_per_liter']} km/liter")
        report.append(f"  • Fuel cost: Rp {bi['fuel_cost_idr_per_liter']:,}/liter")
        report.append(f"  • Average speed: {bi['avg_speed_km_per_hour']} km/hour")
        report.append(f"  • CO2 emissions: {bi['co2_kg_per_liter']} kg/liter")

        report.append("\n" + "="*80)
        report.append("END OF REPORT")
        report.append("="*80)

        # Write report
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            f.write('\n'.join(report))

        print(f"\n📄 Report generated: {output_path}")

    def export_metrics_json(self, output_path: str) -> None:
        """
        Export metrics as JSON.

        Args:
            output_path: Path to save the JSON file
        """
        if self.all_metrics is None:
            self.calculate_all_metrics()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to native Python types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        metrics_native = convert_to_native(self.all_metrics)

        with open(output_file, 'w') as f:
            json.dump(metrics_native, f, indent=2)

        print(f"📄 Metrics JSON exported: {output_path}")


if __name__ == "__main__":
    print("MetricsCalculator module - use via main.py pipeline")
