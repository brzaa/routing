"""
Visualization Module

Creates interactive visualizations for optimization results:
- Clustering comparison maps (before/after)
- Route optimization maps per cluster
- Metrics dashboard with KPIs and charts
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict


def create_clustering_comparison_map(clustering_system, output_path: str) -> str:
    """
    Create side-by-side comparison map using existing PODClusteringSystem method.

    Args:
        clustering_system: PODClusteringSystem instance
        output_path: Path to save the HTML file

    Returns:
        Path to the saved HTML file
    """
    print("\n🗺️  Creating clustering comparison map...")

    # Use the existing method from PODClusteringSystem
    output_file = clustering_system.create_comparison_maps(str(Path(output_path).parent))

    # The existing method saves with a different naming convention, so we rename if needed
    if output_file != output_path:
        import shutil
        try:
            shutil.move(output_file, output_path)
            print(f"✓ Clustering comparison map saved: {output_path}")
            return output_path
        except Exception as e:
            print(f"⚠️  Could not rename file: {e}")
            print(f"✓ Clustering comparison map saved: {output_file}")
            return output_file

    return output_file


def create_route_optimization_maps(route_optimizer, output_dir: str) -> list:
    """
    Create interactive maps showing optimized routes for each cluster.

    Args:
        route_optimizer: RouteOptimizer instance with solved routes
        output_dir: Directory to save the HTML maps

    Returns:
        List of paths to saved HTML files
    """
    print("\n🗺️  Creating route optimization maps...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = []

    branch_lat = route_optimizer.branch_location['latitude']
    branch_lon = route_optimizer.branch_location['longitude']
    branch_name = route_optimizer.branch_location.get('branch_name', 'Branch')

    for cluster_id, route_data in route_optimizer.routes.items():
        # Get route information
        route = route_data['route']
        courier_id = route_data['courier_id']
        total_distance = route_data['total_distance_meters'] / 1000  # Convert to km

        # Create figure
        fig = go.Figure()

        # Add branch location
        fig.add_trace(go.Scattermapbox(
            lat=[branch_lat],
            lon=[branch_lon],
            mode='markers',
            marker=dict(size=15, color='red', symbol='star'),
            name=f'Branch: {branch_name}',
            text=[f'<b>{branch_name}</b>'],
            hoverinfo='text'
        ))

        # Add delivery points
        delivery_lats = [loc[0] for loc in route[1:-1]]  # Exclude start/end depot
        delivery_lons = [loc[1] for loc in route[1:-1]]

        if delivery_lats:
            fig.add_trace(go.Scattermapbox(
                lat=delivery_lats,
                lon=delivery_lons,
                mode='markers',
                marker=dict(size=8, color='blue'),
                name='Delivery Points',
                text=[f'Stop {i+1}' for i in range(len(delivery_lats))],
                hoverinfo='text'
            ))

        # Add optimized route line
        route_lats = [loc[0] for loc in route]
        route_lons = [loc[1] for loc in route]

        fig.add_trace(go.Scattermapbox(
            lat=route_lats,
            lon=route_lons,
            mode='lines',
            line=dict(width=3, color='green'),
            name='Optimized Route',
            hoverinfo='skip'
        ))

        # Update layout
        fig.update_layout(
            title=f'Cluster {cluster_id} - Courier {courier_id}<br>'
                  f'Distance: {total_distance:.1f} km | Deliveries: {len(delivery_lats)}',
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=branch_lat, lon=branch_lon),
                zoom=12
            ),
            showlegend=True,
            height=600,
            margin={"r":0,"t":50,"l":0,"b":0}
        )

        # Save map
        output_file = output_path / f'cluster_{cluster_id}_route.html'
        fig.write_html(str(output_file))
        saved_files.append(str(output_file))

    print(f"✓ Created {len(saved_files)} route optimization maps")

    return saved_files


def create_metrics_dashboard(metrics: Dict, output_path: str) -> str:
    """
    Create interactive HTML dashboard with KPIs and charts.

    Args:
        metrics: Dictionary of calculated metrics
        output_path: Path to save the HTML dashboard

    Returns:
        Path to the saved HTML file
    """
    print("\n📊 Creating metrics dashboard...")

    cm = metrics['clustering_metrics']
    rm = metrics['routing_metrics']
    bi = metrics['business_impact']

    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Workload Distribution (Packages per Courier)',
            'Distance Comparison',
            'POD Count Change',
            'Annual Cost Savings',
            'Balance Metrics (Coefficient of Variation)',
            'Business Impact Summary'
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "indicator"}, {"type": "indicator"}],
            [{"type": "bar"}, {"type": "table"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.15
    )

    # 1. Workload Distribution
    fig.add_trace(
        go.Bar(
            x=['Before', 'After'],
            y=[cm['packages_per_courier_before']['mean'],
               cm['packages_per_courier_after']['mean']],
            name='Mean Packages',
            marker_color=['#FF6B6B', '#4ECDC4'],
            text=[f"{cm['packages_per_courier_before']['mean']:.1f}",
                  f"{cm['packages_per_courier_after']['mean']:.1f}"],
            textposition='auto',
        ),
        row=1, col=1
    )

    # 2. Distance Comparison
    fig.add_trace(
        go.Bar(
            x=['Baseline', 'Optimized'],
            y=[rm['distance_before_km'], rm['distance_after_km']],
            name='Distance (km)',
            marker_color=['#FF6B6B', '#4ECDC4'],
            text=[f"{rm['distance_before_km']:.1f} km",
                  f"{rm['distance_after_km']:.1f} km"],
            textposition='auto',
        ),
        row=1, col=2
    )

    # 3. POD Count Indicator
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=cm['pods_after'],
            title={"text": "POD Count"},
            delta={'reference': cm['pods_before'], 'relative': True,
                   'valueformat': '.1%'},
            domain={'x': [0, 1], 'y': [0, 1]}
        ),
        row=2, col=1
    )

    # 4. Cost Savings Indicator
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=bi['annual_cost_savings_idr'],
            title={"text": "Annual Savings (IDR)"},
            number={'valueformat': ',.0f', 'prefix': 'Rp '},
            domain={'x': [0, 1], 'y': [0, 1]}
        ),
        row=2, col=2
    )

    # 5. CV Comparison
    fig.add_trace(
        go.Bar(
            x=['Before', 'After'],
            y=[cm['packages_per_courier_before']['cv'],
               cm['packages_per_courier_after']['cv']],
            name='CV %',
            marker_color=['#FF6B6B', '#4ECDC4'],
            text=[f"{cm['packages_per_courier_before']['cv']:.1f}%",
                  f"{cm['packages_per_courier_after']['cv']:.1f}%"],
            textposition='auto',
        ),
        row=3, col=1
    )

    # 6. Business Impact Summary Table
    impact_data = {
        'Metric': [
            'Distance Reduction',
            'Workload Balance',
            'Daily Fuel Savings',
            'Daily Cost Savings',
            'Daily Time Savings',
            'Annual Cost Savings'
        ],
        'Value': [
            f"{rm['savings_percent']:.1f}%",
            f"{cm['cv_improvement']:.1f}%",
            f"{bi['fuel_savings_liters']:.1f} L",
            f"Rp {bi['cost_savings_idr']:,.0f}",
            f"{bi['time_savings_hours']:.1f} hrs",
            f"Rp {bi['annual_cost_savings_idr']:,.0f}"
        ]
    }

    fig.add_trace(
        go.Table(
            header=dict(
                values=['<b>Metric</b>', '<b>Value</b>'],
                fill_color='#4ECDC4',
                align='left',
                font=dict(size=12, color='white')
            ),
            cells=dict(
                values=[impact_data['Metric'], impact_data['Value']],
                fill_color='#F7F7F7',
                align='left',
                font=dict(size=11)
            )
        ),
        row=3, col=2
    )

    # Update layout
    fig.update_layout(
        title_text="<b>Route Optimization Dashboard</b>",
        title_font_size=20,
        showlegend=False,
        height=1200,
        font=dict(family="Arial, sans-serif")
    )

    # Update axes
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_yaxes(title_text="Packages", row=1, col=1)
    fig.update_xaxes(title_text="", row=1, col=2)
    fig.update_yaxes(title_text="Distance (km)", row=1, col=2)
    fig.update_xaxes(title_text="", row=3, col=1)
    fig.update_yaxes(title_text="CV %", row=3, col=1)

    # Save dashboard
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_file))

    print(f"✓ Metrics dashboard saved: {output_path}")

    return str(output_file)


def create_courier_workload_chart(clustering_system, output_path: str) -> str:
    """
    Create detailed courier workload comparison chart.

    Args:
        clustering_system: PODClusteringSystem instance
        output_path: Path to save the HTML chart

    Returns:
        Path to the saved HTML file
    """
    print("\n📊 Creating courier workload chart...")

    # Get before data
    current_couriers = clustering_system.current_state['couriers'].copy()
    current_couriers['state'] = 'Before'
    current_couriers = current_couriers.rename(columns={'total_packages': 'packages'})

    # Get after data
    new_couriers = clustering_system.courier_assignments.copy()
    new_couriers['state'] = 'After'
    new_couriers['courier_name'] = new_couriers['name']

    # Combine data
    combined = pd.concat([
        current_couriers[['courier_name', 'packages', 'state']],
        new_couriers[['courier_name', 'packages', 'state']]
    ])

    # Create figure
    fig = go.Figure()

    # Add bars for before
    before_data = combined[combined['state'] == 'Before'].sort_values('packages', ascending=False)
    fig.add_trace(go.Bar(
        x=before_data['courier_name'],
        y=before_data['packages'],
        name='Before Optimization',
        marker_color='#FF6B6B'
    ))

    # Add bars for after
    after_data = combined[combined['state'] == 'After'].sort_values('courier_name')
    fig.add_trace(go.Bar(
        x=after_data['courier_name'],
        y=after_data['packages'],
        name='After Optimization',
        marker_color='#4ECDC4'
    ))

    # Update layout
    fig.update_layout(
        title='Courier Workload Distribution - Before vs After',
        xaxis_title='Courier',
        yaxis_title='Number of Packages',
        barmode='group',
        height=500,
        showlegend=True,
        hovermode='x unified'
    )

    # Save chart
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_file))

    print(f"✓ Courier workload chart saved: {output_path}")

    return str(output_file)


def create_all_visualizations(clustering_system, route_optimizer, metrics: Dict,
                              output_dir: str) -> Dict[str, str]:
    """
    Create all visualizations in one go.

    Args:
        clustering_system: PODClusteringSystem instance
        route_optimizer: RouteOptimizer instance
        metrics: Dictionary of calculated metrics
        output_dir: Base directory for all outputs

    Returns:
        Dictionary mapping visualization names to file paths
    """
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)

    output_path = Path(output_dir)
    maps_dir = output_path / 'maps'
    metrics_dir = output_path / 'metrics'

    maps_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}

    # 1. Clustering comparison map
    outputs['clustering_comparison'] = create_clustering_comparison_map(
        clustering_system,
        str(maps_dir / 'clustering_comparison.html')
    )

    # 2. Route optimization maps
    route_files = create_route_optimization_maps(
        route_optimizer,
        str(maps_dir)
    )
    outputs['route_maps'] = route_files

    # 3. Metrics dashboard
    outputs['metrics_dashboard'] = create_metrics_dashboard(
        metrics,
        str(metrics_dir / 'dashboard.html')
    )

    # 4. Courier workload chart
    outputs['workload_chart'] = create_courier_workload_chart(
        clustering_system,
        str(metrics_dir / 'courier_workload.html')
    )

    print("\n" + "="*80)
    print(f"✅ All visualizations created ({len(outputs)} items)")
    print("="*80)

    return outputs


if __name__ == "__main__":
    print("Visualization module - use via main.py pipeline")
