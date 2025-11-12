"""
Route Optimization System - Streamlit Web App

Upload delivery data and visualize optimized routes with clustering and TSP/VRP optimization.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import base64
from io import BytesIO
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import DataLoader
from src.clustering import PODClusteringSystem
from src.route_optimizer import RouteOptimizer
from src.metrics import MetricsCalculator

# Page configuration
st.set_page_config(
    page_title="Route Optimization System",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🚚 Route Optimization System</h1>', unsafe_allow_html=True)
st.markdown("### Optimize last-mile delivery routes with AI-powered clustering and route planning")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# File uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload Delivery Data (CSV)",
    type=['csv'],
    help="Upload a CSV file with delivery data. Required columns: AWB_NUMBER, EMPLOYEE_ID, coordinates, etc."
)

# City name input
city_name = st.sidebar.text_input(
    "City/Area Name",
    value="My City",
    help="Name of the city or area for the deliveries"
)

# Clustering parameters
st.sidebar.subheader("Clustering Parameters")

clustering_method = st.sidebar.selectbox(
    "Clustering Algorithm",
    options=['hierarchical', 'kmeans', 'dbscan'],
    index=0,
    help="Algorithm used for geographic clustering"
)

target_pods_per_courier = st.sidebar.slider(
    "Target PODs per Courier",
    min_value=2,
    max_value=10,
    value=4,
    help="Target number of delivery zones per courier"
)

max_packages_per_pod = st.sidebar.slider(
    "Max Packages per POD",
    min_value=20,
    max_value=100,
    value=50,
    help="Maximum packages in a single POD. Higher = fewer clusters = faster processing"
)

min_packages_per_pod = st.sidebar.slider(
    "Min Packages per POD",
    min_value=5,
    max_value=30,
    value=10,
    help="Minimum packages in a single POD"
)

# Route optimization parameters
st.sidebar.subheader("Route Optimization")

optimize_routes = st.sidebar.checkbox(
    "🚗 Optimize Routes (TSP/VRP)",
    value=False,
    help="Enable route optimization. Uncheck to only run clustering (much faster!)"
)

if optimize_routes:
    time_limit = st.sidebar.slider(
        "Optimization Time Limit (seconds)",
        min_value=5,
        max_value=60,
        value=15,
        help="Maximum time for route optimization per cluster. Lower = faster but less optimal. Recommended: 10-20s"
    )

    # Advanced routing options
    with st.sidebar.expander("⚡ Advanced Routing Options"):
        use_ensemble = st.checkbox(
            "Enable Ensemble Solving",
            value=False,
            help="Run 3 different TSP strategies in parallel and pick the best solution (same time, better quality)"
        )

        road_distance_factor = st.slider(
            "Road Distance Factor",
            min_value=1.0,
            max_value=2.0,
            value=1.35,
            step=0.05,
            help="Multiplier for straight-line distance to approximate real road distance (1.35 = 35% longer)"
        )

        use_osrm = st.checkbox(
            "Use OSRM for Real Road Distances",
            value=False,
            help="Query OSRM API for actual road network distances (slower but more accurate)"
        )

        if use_osrm:
            osrm_server = st.text_input(
                "OSRM Server URL",
                value="http://router.project-osrm.org",
                help="OSRM server endpoint (use self-hosted server for production)"
            )
        else:
            osrm_server = "http://router.project-osrm.org"

    # Performance tips
    with st.sidebar.expander("💡 Speed Up Tips"):
        st.markdown("""
        **For faster results:**
        - Set Time Limit to **10-15s**
        - Increase Max Packages to **60-80**
        - Use **hierarchical** clustering

        **For best quality:**
        - Set Time Limit to **30-60s**
        - Keep Max Packages at **30-40**
        - Expect longer wait times
        """)
else:
    st.sidebar.info("💡 Clustering only mode - **much faster!** Enable routing above for full optimization.")
    time_limit = 15  # Default value when disabled
    use_ensemble = False
    road_distance_factor = 1.35
    use_osrm = False
    osrm_server = "http://router.project-osrm.org"

# Run button
run_optimization = st.sidebar.button("🚀 Run Optimization", type="primary", width='stretch')

# Main content area
if uploaded_file is None:
    # Instructions when no file uploaded
    st.info("👈 Upload a CSV file to get started!")

    st.markdown("### 📋 Required CSV Columns")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Essential Columns:**
        - `AWB_NUMBER` - Unique delivery ID
        - `EMPLOYEE_ID` - Courier ID
        - `NICKNAME` - Courier name
        - `DO_POD_DELIVER_CODE` - Current POD assignment
        - `BERATASLI` - Package weight (kg)
        - `SELECTED_LATITUDE` - Delivery latitude
        - `SELECTED_LONGITUDE` - Delivery longitude
        """)

    with col2:
        st.markdown("""
        **Branch Information:**
        - `BRANCH_LATITUDE` - Branch/depot latitude
        - `BRANCH_LONGITUDE` - Branch/depot longitude
        - `GERAI` or `BRANCH_NAME` - Branch name
        - `KODE_GERAI` or `BRANCH_CODE` - Branch code
        - `DO_POD_DELIVER_DATE` or `DELIVERY_DATE` - Date
        """)

    st.markdown("### 📊 What You'll Get")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **📈 Optimization Metrics**
        - Distance reduction %
        - Cost savings (daily & annual)
        - Fuel savings
        - CO2 reduction
        - Workload balance improvement
        """)

    with col2:
        st.markdown("""
        **🗺️ Interactive Maps**
        - Before/After clustering comparison
        - Optimized routes per cluster
        - Delivery point visualization
        - Branch location markers
        """)

    with col3:
        st.markdown("""
        **📥 Export Options**
        - Optimized route CSVs
        - Cluster assignments
        - Business metrics (JSON)
        - Comprehensive report
        """)

    # Sample data information
    st.markdown("### 🔍 Sample Data")
    st.markdown("""
    Your CSV should look like this:

    | AWB_NUMBER | EMPLOYEE_ID | NICKNAME | BERATASLI | SELECTED_LATITUDE | SELECTED_LONGITUDE | ... |
    |------------|-------------|----------|-----------|-------------------|-------------------|-----|
    | AWB123456  | EMP001      | John Doe | 2.5       | -6.2088          | 106.8456          | ... |
    | AWB123457  | EMP002      | Jane Smith | 3.2     | -6.2100          | 106.8500          | ... |
    """)

else:
    # File uploaded - show preview
    st.success(f"✅ File uploaded: {uploaded_file.name}")

    # Preview data
    with st.expander("📄 Preview Data", expanded=False):
        try:
            df_preview = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)  # Reset file pointer

            st.write(f"**Rows:** {len(df_preview):,} | **Columns:** {len(df_preview.columns)}")
            st.dataframe(df_preview.head(10), width='stretch')
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

    # Run optimization when button clicked
    if run_optimization:

        # Create temporary directory for outputs
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Save uploaded file
            input_file = temp_path / "input.csv"
            with open(input_file, "wb") as f:
                f.write(uploaded_file.getvalue())

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # Step 1: Load and clean data
                status_text.text("📂 Step 1/5: Loading and cleaning data...")
                progress_bar.progress(10)

                loader = DataLoader()
                df = loader.load_and_clean(str(input_file))

                progress_bar.progress(20)

                # Show data stats
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Deliveries", f"{len(df):,}")
                col2.metric("Couriers", f"{df['EMPLOYEE_ID'].nunique()}")
                col3.metric("Current PODs", f"{df['DO_POD_DELIVER_CODE'].nunique()}")
                col4.metric("Total Weight", f"{df['BERATASLI'].sum():.1f} kg")

                # Performance warning for large datasets
                if len(df) > 1500:
                    st.warning(f"""
                    ⚠️ **Large Dataset Detected ({len(df):,} deliveries)**

                    This may take 5-15 minutes to optimize. To speed up:
                    - Reduce **Time Limit** to 10-15 seconds (sidebar)
                    - Increase **Max Packages per POD** to 60-80 (sidebar)
                    - Or process in smaller batches
                    """)
                elif len(df) > 1000:
                    st.info(f"""
                    ℹ️ **Medium Dataset ({len(df):,} deliveries)** - Estimated time: 3-8 minutes

                    💡 Tip: Reduce **Time Limit** in sidebar for faster results.
                    """)

                # Step 2: Clustering
                status_text.text("🗺️ Step 2/5: Running geographic clustering...")
                progress_bar.progress(35)

                clustering_system = PODClusteringSystem(df=df, city_name=city_name)
                clustering_system.analyze_current_state()

                clustering_system.cluster_delivery_points(
                    method=clustering_method,
                    target_pods_per_courier=target_pods_per_courier,
                    max_packages_per_pod=max_packages_per_pod,
                    min_packages_per_pod=min_packages_per_pod,
                    separate_heavy_packages=True
                )

                clustering_system.assign_pods_to_couriers(method='balanced')
                clustering_system.evaluate_optimization()

                progress_bar.progress(50)

                # Step 3: Route optimization (conditional)
                if optimize_routes:
                    status_text.text("🚗 Step 3/5: Optimizing delivery routes...")
                    progress_bar.progress(60)

                    route_optimizer = RouteOptimizer(
                        clustering_system,
                        use_ensemble=use_ensemble,
                        road_distance_factor=road_distance_factor,
                        use_osrm=use_osrm,
                        osrm_server=osrm_server
                    )
                    route_optimizer.solve_all_clusters(time_limit_seconds=time_limit)

                    progress_bar.progress(75)

                    # Step 4: Calculate metrics
                    status_text.text("📊 Step 4/5: Calculating business metrics...")
                    progress_bar.progress(85)

                    calculator = MetricsCalculator(clustering_system, route_optimizer)
                    metrics = calculator.calculate_all_metrics()

                    progress_bar.progress(95)

                    # Step 5: Generate outputs
                    status_text.text("📁 Step 5/5: Generating visualizations and reports...")
                else:
                    # Skip routing, only show clustering results
                    status_text.text("✅ Clustering complete! (Route optimization skipped)")
                    progress_bar.progress(100)

                    route_optimizer = None
                    calculator = None
                    metrics = None

                # Create maps
                maps_dir = temp_path / "maps"
                maps_dir.mkdir(exist_ok=True)

                map_file = clustering_system.create_comparison_maps(str(maps_dir))

                progress_bar.progress(100)
                status_text.text("✅ Optimization complete!")

                # Display results
                st.markdown("---")

                if optimize_routes and metrics:
                    st.markdown("## 🎉 Optimization Results")

                    # Key metrics
                    st.markdown("### 📊 Key Performance Indicators")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            "Distance Reduction",
                            f"{metrics['routing_metrics']['savings_percent']:.1f}%",
                            f"-{metrics['routing_metrics']['savings_km']:.1f} km"
                        )

                    with col2:
                        st.metric(
                            "Daily Cost Savings",
                            f"Rp {metrics['business_impact']['cost_savings_idr']:,.0f}",
                            f"{metrics['business_impact']['fuel_savings_liters']:.1f} L fuel"
                        )

                    with col3:
                        st.metric(
                            "Annual Savings",
                            f"Rp {metrics['business_impact']['annual_cost_savings_idr']/1_000_000:.1f}M",
                            f"{metrics['business_impact']['annual_fuel_savings_liters']:,.0f} L/year"
                        )

                    with col4:
                        st.metric(
                            "Workload Balance",
                            f"+{metrics['clustering_metrics']['cv_improvement']:.1f}%",
                            "Improvement"
                        )
                else:
                    st.markdown("## 🗂️ Clustering Results")

                    # Clustering-only metrics
                    opt_metrics = clustering_system.optimization_metrics

                    # Calculate pod change (absolute number)
                    pod_change = opt_metrics['after']['n_pods'] - opt_metrics['before']['n_pods']

                    # Calculate average packages per POD
                    total_packages = len(df)
                    avg_packages = total_packages / opt_metrics['after']['n_pods']

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            "PODs Created",
                            opt_metrics['after']['n_pods'],
                            f"{pod_change:+d} from current"
                        )

                    with col2:
                        st.metric(
                            "Workload Balance",
                            f"{100 - opt_metrics['after']['cv']:.1f}%",
                            f"+{opt_metrics['improvements']['cv_improvement']:.1f}%"
                        )

                    with col3:
                        st.metric(
                            "Max/Min Ratio",
                            f"{opt_metrics['after']['max_min_ratio']:.2f}x",
                            f"-{opt_metrics['improvements']['ratio_improvement']:.1f}%"
                        )

                    with col4:
                        st.metric(
                            "Avg POD Size",
                            f"{avg_packages:.1f}",
                            "packages"
                        )

                    st.info("💡 Enable **'Optimize Routes'** in the sidebar to see distance savings and routing optimization!")

                # Detailed metrics tabs
                if optimize_routes and metrics:
                    tab1, tab2, tab3 = st.tabs(["📈 Clustering", "🚗 Routing", "💰 Business Impact"])

                    with tab1:
                        st.markdown("### Clustering Optimization")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**Before Optimization:**")
                            st.write(f"- PODs: {metrics['clustering_metrics']['pods_before']}")
                            st.write(f"- Avg Packages/Courier: {metrics['clustering_metrics']['packages_per_courier_before']['mean']:.1f}")
                            st.write(f"- Coefficient of Variation: {metrics['clustering_metrics']['packages_per_courier_before']['cv']:.1f}%")
                            st.write(f"- Max/Min Ratio: {metrics['clustering_metrics']['max_min_ratio_before']:.2f}x")

                        with col2:
                            st.markdown("**After Optimization:**")
                            st.write(f"- PODs: {metrics['clustering_metrics']['pods_after']}")
                            st.write(f"- Avg Packages/Courier: {metrics['clustering_metrics']['packages_per_courier_after']['mean']:.1f}")
                            st.write(f"- Coefficient of Variation: {metrics['clustering_metrics']['packages_per_courier_after']['cv']:.1f}%")
                            st.write(f"- Max/Min Ratio: {metrics['clustering_metrics']['max_min_ratio_after']:.2f}x")

                    with tab2:
                        st.markdown("### Route Optimization")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**Distance Metrics:**")
                            st.write(f"- Baseline Distance: {metrics['routing_metrics']['distance_before_km']:.1f} km")
                            st.write(f"- Optimized Distance: {metrics['routing_metrics']['distance_after_km']:.1f} km")
                            st.write(f"- Savings: {metrics['routing_metrics']['savings_km']:.1f} km ({metrics['routing_metrics']['savings_percent']:.1f}%)")

                        with col2:
                            st.markdown("**Optimization Details:**")
                            st.write(f"- Clusters Optimized: {metrics['routing_metrics']['total_clusters']}")
                            st.write(f"- Success Rate: {metrics['routing_metrics']['optimization_success_rate']:.1f}%")
                            st.write(f"- Avg Cluster Distance: {metrics['routing_metrics']['avg_cluster_distance_km']:.1f} km")

                    with tab3:
                        st.markdown("### Business Impact")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**Daily Impact:**")
                            st.write(f"- Fuel Savings: {metrics['business_impact']['fuel_savings_liters']:.1f} liters")
                            st.write(f"- Cost Savings: Rp {metrics['business_impact']['cost_savings_idr']:,.0f}")
                            st.write(f"- Time Savings: {metrics['business_impact']['time_savings_hours']:.1f} hours")
                            st.write(f"- CO2 Reduction: {metrics['business_impact']['co2_reduction_kg']:.1f} kg")

                        with col2:
                            st.markdown("**Annual Projection:**")
                            st.write(f"- Fuel Savings: {metrics['business_impact']['annual_fuel_savings_liters']:,.0f} liters")
                            st.write(f"- Cost Savings: Rp {metrics['business_impact']['annual_cost_savings_idr']:,.0f}")
                            st.write(f"- Time Savings: {metrics['business_impact']['annual_time_savings_hours']:,.0f} hours")
                            st.write(f"- CO2 Reduction: {metrics['business_impact']['annual_co2_reduction_kg']:,.0f} kg")
                else:
                    # Clustering-only detailed view
                    with st.expander("📊 Detailed Clustering Metrics", expanded=True):
                        opt_metrics = clustering_system.optimization_metrics

                        # Calculate average packages per POD
                        total_packages = len(df)
                        avg_packages_before = total_packages / opt_metrics['before']['n_pods']
                        avg_packages_after = total_packages / opt_metrics['after']['n_pods']

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**Before Optimization:**")
                            st.write(f"- PODs: {opt_metrics['before']['n_pods']}")
                            st.write(f"- Avg Packages/POD: {avg_packages_before:.1f}")
                            st.write(f"- Coefficient of Variation: {opt_metrics['before']['cv']:.1f}%")
                            st.write(f"- Max/Min Ratio: {opt_metrics['before']['max_min_ratio']:.2f}x")
                            st.write(f"- Avg Spread: {opt_metrics['before']['avg_spread']:.0f}m")

                        with col2:
                            st.markdown("**After Optimization:**")
                            st.write(f"- PODs: {opt_metrics['after']['n_pods']}")
                            st.write(f"- Avg Packages/POD: {avg_packages_after:.1f}")
                            st.write(f"- Coefficient of Variation: {opt_metrics['after']['cv']:.1f}%")
                            st.write(f"- Max/Min Ratio: {opt_metrics['after']['max_min_ratio']:.2f}x")
                            st.write(f"- Avg Spread: {opt_metrics['after']['avg_spread']:.0f}m")

                # Display map
                st.markdown("---")
                st.markdown("### 🗺️ Clustering Comparison Map")

                if Path(map_file).exists():
                    with open(map_file, 'r', encoding='utf-8') as f:
                        map_html = f.read()
                    st.components.v1.html(map_html, height=600, scrolling=True)
                else:
                    st.warning("Map file not found")

                # Display route optimization maps (only if routing was enabled)
                if optimize_routes and route_optimizer and route_optimizer.routes:
                    st.markdown("---")
                    st.markdown("### 🚗 Optimized Routes per Cluster")
                    st.markdown(f"Total clusters optimized: **{len(route_optimizer.routes)}**")
                    # Group clusters for better display (max 5 tabs at a time)
                    cluster_ids = list(route_optimizer.routes.keys())

                    # Show first 5 clusters in tabs
                    if len(cluster_ids) <= 5:
                        tabs = st.tabs([f"Cluster {cid}" for cid in cluster_ids[:5]])

                        for idx, cluster_id in enumerate(cluster_ids[:5]):
                            with tabs[idx]:
                                route_data = route_optimizer.routes[cluster_id]

                                # Show cluster info
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Courier ID", route_data['courier_id'])
                                col2.metric("Deliveries", len(route_data['delivery_sequence']))
                                col3.metric("Distance", f"{route_data['total_distance_meters']/1000:.1f} km")

                                # Create Plotly map for this cluster
                                import plotly.graph_objects as go

                                route = route_data['route']
                                branch_lat = route[0][0]
                                branch_lon = route[0][1]

                                fig = go.Figure()

                                # Add branch location
                                fig.add_trace(go.Scattermapbox(
                                    lat=[branch_lat],
                                    lon=[branch_lon],
                                    mode='markers',
                                    marker=dict(size=15, color='red', symbol='star'),
                                    name='Branch',
                                    text=['Branch'],
                                    hoverinfo='text'
                                ))

                                # Add delivery points
                                if len(route) > 2:
                                    delivery_lats = [loc[0] for loc in route[1:-1]]
                                    delivery_lons = [loc[1] for loc in route[1:-1]]

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
                                # Use actual road geometry if available (OSRM), otherwise straight lines
                                if route_data.get('route_geometry') is not None:
                                    # Use actual road path from OSRM
                                    road_path = route_data['route_geometry']
                                    route_lats = [loc[0] for loc in road_path]
                                    route_lons = [loc[1] for loc in road_path]
                                    route_name = 'Optimized Route (Road Path)'
                                    route_color = 'darkgreen'
                                else:
                                    # Use straight lines between waypoints
                                    route_lats = [loc[0] for loc in route]
                                    route_lons = [loc[1] for loc in route]
                                    route_name = 'Optimized Route (Straight Line)'
                                    route_color = 'green'

                                fig.add_trace(go.Scattermapbox(
                                    lat=route_lats,
                                    lon=route_lons,
                                    mode='lines',
                                    line=dict(width=3, color=route_color),
                                    name=route_name,
                                    hoverinfo='skip'
                                ))

                                # Update layout
                                fig.update_layout(
                                    mapbox=dict(
                                        style="open-street-map",
                                        center=dict(lat=branch_lat, lon=branch_lon),
                                        zoom=12
                                    ),
                                    showlegend=True,
                                    height=500,
                                    margin={"r":0,"t":0,"l":0,"b":0}
                                )

                                st.plotly_chart(fig, width='stretch')

                    # If more than 5 clusters, show them in an expander
                    if len(cluster_ids) > 5:
                        st.markdown(f"**Showing first 5 clusters. {len(cluster_ids)-5} more clusters available.**")

                        with st.expander(f"View remaining {len(cluster_ids)-5} clusters"):
                            for cluster_id in cluster_ids[5:]:
                                route_data = route_optimizer.routes[cluster_id]

                                st.markdown(f"#### Cluster {cluster_id}")
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Courier ID", route_data['courier_id'])
                                col2.metric("Deliveries", len(route_data['delivery_sequence']))
                                col3.metric("Distance", f"{route_data['total_distance_meters']/1000:.1f} km")

                                # Create Plotly map
                                import plotly.graph_objects as go

                                route = route_data['route']
                                branch_lat = route[0][0]
                                branch_lon = route[0][1]

                                fig = go.Figure()

                                # Add branch
                                fig.add_trace(go.Scattermapbox(
                                    lat=[branch_lat],
                                    lon=[branch_lon],
                                    mode='markers',
                                    marker=dict(size=15, color='red', symbol='star'),
                                    name='Branch'
                                ))

                                # Add deliveries
                                if len(route) > 2:
                                    delivery_lats = [loc[0] for loc in route[1:-1]]
                                    delivery_lons = [loc[1] for loc in route[1:-1]]

                                    fig.add_trace(go.Scattermapbox(
                                        lat=delivery_lats,
                                        lon=delivery_lons,
                                        mode='markers',
                                        marker=dict(size=8, color='blue'),
                                        name='Deliveries'
                                    ))

                                # Add route line
                                # Use actual road geometry if available (OSRM), otherwise straight lines
                                if route_data.get('route_geometry') is not None:
                                    # Use actual road path from OSRM
                                    road_path = route_data['route_geometry']
                                    route_lats = [loc[0] for loc in road_path]
                                    route_lons = [loc[1] for loc in road_path]
                                    route_name = 'Route (Road Path)'
                                    route_color = 'darkgreen'
                                else:
                                    # Use straight lines between waypoints
                                    route_lats = [loc[0] for loc in route]
                                    route_lons = [loc[1] for loc in route]
                                    route_name = 'Route (Straight Line)'
                                    route_color = 'green'

                                fig.add_trace(go.Scattermapbox(
                                    lat=route_lats,
                                    lon=route_lons,
                                    mode='lines',
                                    line=dict(width=3, color=route_color),
                                    name=route_name
                                ))

                                fig.update_layout(
                                    mapbox=dict(
                                        style="open-street-map",
                                        center=dict(lat=branch_lat, lon=branch_lon),
                                        zoom=12
                                    ),
                                    height=400,
                                    margin={"r":0,"t":0,"l":0,"b":0}
                                )

                                st.plotly_chart(fig, width='stretch')
                                st.markdown("---")
                else:
                    st.info("No route optimizations available to display.")

                # Export options
                st.markdown("---")
                st.markdown("### 📥 Export Results")

                col1, col2, col3 = st.columns(3)

                # Export metrics as JSON (only if routing was done)
                with col1:
                    if optimize_routes and metrics:
                        metrics_json = pd.io.json.dumps(metrics, indent=2)
                        st.download_button(
                            label="📄 Download Metrics (JSON)",
                            data=metrics_json,
                            file_name=f"{city_name.lower().replace(' ', '_')}_metrics.json",
                            mime="application/json"
                        )
                    else:
                        # Export clustering metrics only
                        clustering_json = pd.io.json.dumps(clustering_system.optimization_metrics, indent=2)
                        st.download_button(
                            label="📄 Download Clustering (JSON)",
                            data=clustering_json,
                            file_name=f"{city_name.lower().replace(' ', '_')}_clustering.json",
                            mime="application/json"
                        )

                # Export optimized assignments
                with col2:
                    assignments_df = clustering_system.courier_assignments
                    csv_buffer = BytesIO()
                    assignments_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📊 Download Assignments (CSV)",
                        data=csv_buffer.getvalue(),
                        file_name=f"{city_name.lower().replace(' ', '_')}_assignments.csv",
                        mime="text/csv"
                    )

                # Export cluster summary
                with col3:
                    clusters_df = clustering_system.new_pods
                    csv_buffer2 = BytesIO()
                    clusters_df.to_csv(csv_buffer2, index=False)
                    st.download_button(
                        label="🗂️ Download Clusters (CSV)",
                        data=csv_buffer2.getvalue(),
                        file_name=f"{city_name.lower().replace(' ', '_')}_clusters.csv",
                        mime="text/csv"
                    )

                st.success("✅ Optimization completed successfully! Download the results above.")

            except Exception as e:
                st.error(f"❌ Error during optimization: {str(e)}")
                st.exception(e)

            finally:
                progress_bar.empty()
                status_text.empty()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🚚 Route Optimization System | Built with ❤️ using Streamlit</p>
    <p>Powered by Google OR-Tools, scikit-learn, and geopy</p>
</div>
""", unsafe_allow_html=True)
