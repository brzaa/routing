"""
POD CLUSTERING & COURIER ASSIGNMENT SYSTEM
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class BranchManager:
    """
    Manages branch locations and information
    """
    def __init__(self):
        self.branches = {}

    def detect_branch_from_data(self, df: pd.DataFrame, city_name: str) -> Dict:
        """
        Detect branch location from data columns
        Returns branch info including coordinates
        """
        branch_info = {}

        # Extract branch location from data columns
        if 'BRANCH_LATITUDE' in df.columns and 'BRANCH_LONGITUDE' in df.columns:
            # Get the first non-null branch location (should be same for all rows)
            branch_lat = df['BRANCH_LATITUDE'].dropna().iloc[0] if not df['BRANCH_LATITUDE'].dropna().empty else None
            branch_lon = df['BRANCH_LONGITUDE'].dropna().iloc[0] if not df['BRANCH_LONGITUDE'].dropna().empty else None

            if branch_lat is not None and branch_lon is not None:
                branch_info = {
                    'name': city_name,
                    'latitude': float(branch_lat),
                    'longitude': float(branch_lon),
                    'type': 'actual'
                }

                # Add additional branch information if available
                if 'BRANCH_NAME' in df.columns:
                    branch_info['branch_name'] = df['BRANCH_NAME'].dropna().iloc[0] if not df['BRANCH_NAME'].dropna().empty else city_name
                if 'BRANCH_CODE' in df.columns:
                    branch_info['branch_code'] = df['BRANCH_CODE'].dropna().iloc[0] if not df['BRANCH_CODE'].dropna().empty else None
                if 'BRANCH_TYPE_NAME' in df.columns:
                    branch_info['branch_type'] = df['BRANCH_TYPE_NAME'].dropna().iloc[0] if not df['BRANCH_TYPE_NAME'].dropna().empty else None
            else:
                # Fallback to centroid if branch columns exist but are empty
                branch_info = self._fallback_to_centroid(df, city_name)
        else:
            # Fallback to centroid if branch columns don't exist
            branch_info = self._fallback_to_centroid(df, city_name)

        return branch_info

    def _fallback_to_centroid(self, df: pd.DataFrame, city_name: str) -> Dict:
        """
        Fallback method using centroid of deliveries
        """
        center_lat = df['SELECTED_LATITUDE'].median()
        center_lon = df['SELECTED_LONGITUDE'].median()

        return {
            'name': city_name,
            'latitude': center_lat,
            'longitude': center_lon,
            'type': 'estimated',
            'branch_name': city_name,
            'branch_code': None,
            'branch_type': None
        }

    def set_branch_location(self, city_name: str, lat: float, lon: float):
        """
        Manually set branch location
        """
        self.branches[city_name] = {
            'name': city_name,
            'latitude': lat,
            'longitude': lon,
            'type': 'manual'
        }

    def get_branch(self, city_name: str) -> Optional[Dict]:
        """
        Get branch information
        """
        return self.branches.get(city_name)

class PODClusteringSystem:
    """
    End-to-end system for POD clustering and courier assignment with branch visualization
    """

    def __init__(self, df: pd.DataFrame, city_name: str = "Unknown City",
                 branch_location: Optional[Dict] = None):
        self.df = df.copy()
        self.city_name = city_name
        self.branch_location = branch_location

        # Validate required columns
        self._validate_data()

        # Add unique ID for each delivery
        self.df['delivery_id'] = range(len(self.df))

        # Detect or set branch location
        if not self.branch_location:
            branch_manager = BranchManager()
            self.branch_location = branch_manager.detect_branch_from_data(self.df, city_name)

        # Add delivery categorization
        self._categorize_deliveries()

        # Storage for results
        self.current_state = None
        self.new_pods = None
        self.courier_assignments = None
        self.optimization_metrics = None

    def _validate_data(self):
        """
        Validate that required columns exist in the dataframe
        """
        required_columns = [
            'AWB_NUMBER', 'SELECTED_LATITUDE', 'SELECTED_LONGITUDE',
            'DO_POD_DELIVER_CODE', 'EMPLOYEE_ID', 'NICKNAME', 'BERATASLI'
        ]

        missing_columns = [col for col in required_columns if col not in self.df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Check for null values in critical columns
        critical_columns = ['SELECTED_LATITUDE', 'SELECTED_LONGITUDE', 'DO_POD_DELIVER_CODE']
        for col in critical_columns:
            null_count = self.df[col].isnull().sum()
            if null_count > 0:
                print(f"⚠️ Warning: {null_count} null values in {col}, removing these rows")
                self.df = self.df.dropna(subset=critical_columns)

    def _categorize_deliveries(self):
        """
        Categorize deliveries based on weight and calculate distance from branch.
        """
        print("\n📊 Calculating distances and categorizing by vehicle type...")

        # Calculate distance from branch to each delivery
        if self.branch_location:
            self.df['distance_from_branch_km'] = self._calculate_distance(
                self.branch_location['latitude'],
                self.branch_location['longitude'],
                self.df['SELECTED_LATITUDE'].values,
                self.df['SELECTED_LONGITUDE'].values
            ) / 1000  # Convert to km
        else:
            # If no branch location, use median as estimate
            median_lat = self.df['SELECTED_LATITUDE'].median()
            median_lon = self.df['SELECTED_LONGITUDE'].median()
            self.df['distance_from_branch_km'] = self._calculate_distance(
                median_lat, median_lon,
                self.df['SELECTED_LATITUDE'].values,
                self.df['SELECTED_LONGITUDE'].values
            ) / 1000

        # Vehicle assignment based on weight
        self.df['vehicle_type'] = self.df['BERATASLI'].apply(
            lambda x: 'SIGESIT_CAR' if x >= 7 else 'MOTORCYCLE'
        )

        # Print categorization summary
        print("\n🚗 Vehicle Assignment Summary:")
        vehicle_summary = self.df['vehicle_type'].value_counts()
        for vehicle, count in vehicle_summary.items():
            weight_range = "≥7kg" if vehicle == 'SIGESIT_CAR' else "<7kg"
            print(f"  • {vehicle}: {count} packages ({weight_range})")

    def cluster_delivery_points(self, method='kmeans', target_pods_per_courier=4,
                                max_packages_per_pod=40,
                                min_packages_per_pod=10,
                                separate_heavy_packages=True,
                                max_total_pods=None):
        """
        Step 2: Create new POD clusters based on geography and vehicle type.
        """
        print("\n" + "="*80)
        print("STEP 2: CLUSTERING DELIVERY POINTS INTO NEW PODS")
        print("="*80)

        # Calculate optimal POD count
        n_couriers = self.current_state['metrics']['n_couriers']
        current_pods = self.current_state['metrics']['n_pods']

        # Target fewer PODs for operational efficiency
        target_n_pods = n_couriers * target_pods_per_courier

        # Apply maximum cap if specified
        if max_total_pods:
            target_n_pods = min(target_n_pods, max_total_pods)

        # Suggest optimal range
        suggested_min = int(current_pods * 0.8)
        suggested_max = int(current_pods * 1.2)

        print(f"\n📊 POD Planning:")
        print(f"  • Current PODs: {current_pods}")
        print(f"  • Suggested range: {suggested_min} - {suggested_max} PODs")
        print(f"  • Initial target: {target_n_pods} PODs")

        # Adjust if target is too different from current
        if target_n_pods > suggested_max:
            target_n_pods = suggested_max
            print(f"  • Adjusted to: {target_n_pods} PODs (capped at suggested max)")
        elif target_n_pods < suggested_min:
            target_n_pods = suggested_min
            print(f"  • Adjusted to: {target_n_pods} PODs (raised to suggested min)")

        all_pods = []

        if separate_heavy_packages:
            print("\n Separating clusters by vehicle type...")
            motorcycle_df = self.df[self.df['vehicle_type'] == 'MOTORCYCLE'].copy()
            car_df = self.df[self.df['vehicle_type'] == 'SIGESIT_CAR'].copy()

            if not motorcycle_df.empty:
                print(f"  Clustering {len(motorcycle_df)} motorcycle packages...")
                motorcycle_pods = self._cluster_subset(
                    motorcycle_df, 'MOTORCYCLE', method,
                    target_pods=max(1, int(target_n_pods * len(motorcycle_df) / len(self.df))),
                    max_packages_per_pod=max_packages_per_pod,
                    min_packages_per_pod=min_packages_per_pod
                )
                all_pods.append(motorcycle_pods)

            if not car_df.empty:
                print(f"  Clustering {len(car_df)} car packages...")
                car_pods = self._cluster_subset(
                    car_df, 'SIGESIT_CAR', method,
                    target_pods=max(1, int(target_n_pods * len(car_df) / len(self.df))),
                    max_packages_per_pod=max_packages_per_pod,
                    min_packages_per_pod=min_packages_per_pod
                )
                all_pods.append(car_pods)
        else:
            print("\n Clustering all packages together...")
            all_pods.append(self._cluster_subset(
                self.df, 'MIXED', method,
                target_pods=target_n_pods,
                max_packages_per_pod=max_packages_per_pod,
                min_packages_per_pod=min_packages_per_pod
            ))

        # Combine all PODs
        if all_pods:
            self.new_pods = pd.concat(all_pods, ignore_index=True)

            # Update cluster assignments in main dataframe
            for _, pod in self.new_pods.iterrows():
                for delivery_id in pod['delivery_ids']:
                    self.df.loc[self.df['delivery_id'] == delivery_id, 'new_pod_cluster'] = pod['pod_id']
        else:
            self.new_pods = pd.DataFrame()

        print(f"\n✅ Created {len(self.new_pods)} new PODs")
        print(f"  • Change from current: {len(self.new_pods) - current_pods:+d} PODs")
        print(f"  • Avg packages/POD: {self.new_pods['n_packages'].mean():.1f}")
        print(f"  • Min packages: {self.new_pods['n_packages'].min()}")
        print(f"  • Max packages: {self.new_pods['n_packages'].max()}")
        print(f"  • Avg spread: {self.new_pods['geographic_spread'].mean():.1f}m")

        return self.new_pods

    def _cluster_subset(self, df_subset, vehicle_type, method,
                        target_pods=None,
                        max_packages_per_pod=40, min_packages_per_pod=10):
        """
        Cluster a subset of deliveries with specific target POD count
        """
        if len(df_subset) == 0:
            return pd.DataFrame()

        # Prepare coordinates
        coords = df_subset[['SELECTED_LATITUDE', 'SELECTED_LONGITUDE']].values

        # Determine target PODs
        target_n_pods = target_pods if target_pods is not None else max(1, len(df_subset) // 20)
        target_n_pods = max(1, min(target_n_pods, len(df_subset)))

        # Validate the target makes sense
        if target_n_pods > 0:
            avg_packages = len(df_subset) / target_n_pods
            if avg_packages > max_packages_per_pod:
                target_n_pods = int(len(df_subset) / max_packages_per_pod * 1.1)
            elif avg_packages < min_packages_per_pod:
                target_n_pods = max(1, int(len(df_subset) / min_packages_per_pod * 0.9))

        # Perform clustering
        if len(df_subset) < 2 or target_n_pods <= 1:
            df_subset['new_pod_cluster'] = 0
        elif method == 'kmeans':
            clusterer = KMeans(n_clusters=target_n_pods, random_state=42, n_init=10)
            df_subset['new_pod_cluster'] = clusterer.fit_predict(coords)
        elif method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=target_n_pods)
            df_subset['new_pod_cluster'] = clusterer.fit_predict(coords)

        # Create POD structure
        pods = df_subset.groupby('new_pod_cluster').agg({
            'AWB_NUMBER': 'count',
            'BERATASLI': 'sum',
            'SELECTED_LATITUDE': ['mean', 'std', list],
            'SELECTED_LONGITUDE': ['mean', 'std', list],
            'delivery_id': list,
            'distance_from_branch_km': 'mean'
        }).reset_index()

        pods.columns = ['pod_id', 'n_packages', 'total_weight',
                        'lat_center', 'lat_std', 'lat_points',
                        'lon_center', 'lon_std', 'lon_points',
                        'delivery_ids', 'avg_distance_km']

        pods['vehicle_type'] = vehicle_type
        pods['geographic_spread'] = np.sqrt(pods['lat_std']**2 + pods['lon_std']**2) * 111000

        if self.branch_location:
            pods['distance_from_branch'] = self._calculate_distance(
                self.branch_location['latitude'], self.branch_location['longitude'],
                pods['lat_center'].values, pods['lon_center'].values
            )

        pods['pod_code'] = f'POD_{vehicle_type}_' + pods['pod_id'].astype(str).str.zfill(4)

        return pods

    def analyze_current_state(self) -> Dict:
        """
        Step 1: Analyze current POD and courier distribution
        """
        print("\n" + "="*80)
        print("STEP 1: ANALYZING CURRENT STATE")
        print("="*80)

        # Current POD analysis
        current_pods = self.df.groupby('DO_POD_DELIVER_CODE').agg({
            'AWB_NUMBER': 'count',
            'BERATASLI': 'sum',
            'EMPLOYEE_ID': 'first',
            'NICKNAME': 'first',
            'SELECTED_LATITUDE': ['mean', 'std'],
            'SELECTED_LONGITUDE': ['mean', 'std'],
            'distance_from_branch_km': 'mean'
        }).reset_index()

        current_pods.columns = ['pod_code', 'n_packages', 'total_weight',
                                'courier_id', 'courier_name',
                                'lat_mean', 'lat_std', 'lon_mean', 'lon_std',
                                'avg_distance_km']

        # Calculate spread and distance from branch
        current_pods['geographic_spread'] = np.sqrt(
            current_pods['lat_std']**2 + current_pods['lon_std']**2
        ) * 111000  # Convert to meters

        # Distance from branch
        if self.branch_location:
            current_pods['distance_from_branch'] = self._calculate_distance(
                self.branch_location['latitude'], self.branch_location['longitude'],
                current_pods['lat_mean'].values, current_pods['lon_mean'].values
            )

        # Current courier workload
        current_couriers = self.df.groupby(['EMPLOYEE_ID', 'NICKNAME']).agg({
            'AWB_NUMBER': 'count',
            'DO_POD_DELIVER_CODE': 'nunique',
            'BERATASLI': 'sum'
        }).reset_index()

        current_couriers.columns = ['courier_id', 'courier_name',
                                    'total_packages', 'pod_count', 'total_weight']

        # Calculate imbalance metrics
        cv_packages = (current_couriers['total_packages'].std() /
                       current_couriers['total_packages'].mean()) * 100

        max_min_ratio = (current_couriers['total_packages'].max() /
                         current_couriers['total_packages'].min())

        self.current_state = {
            'pods': current_pods,
            'couriers': current_couriers,
            'branch': self.branch_location,
            'metrics': {
                'n_pods': len(current_pods),
                'n_couriers': len(current_couriers),
                'total_packages': len(self.df),
                'avg_packages_per_pod': current_pods['n_packages'].mean(),
                'avg_packages_per_courier': current_couriers['total_packages'].mean(),
                'cv_packages': cv_packages,
                'max_min_ratio': max_min_ratio,
                'avg_spread': current_pods['geographic_spread'].mean(),
                'avg_distance_from_branch': current_pods.get('distance_from_branch', pd.Series()).mean()
            }
        }

        # Display branch information
        branch_name = self.branch_location.get('branch_name', self.city_name)
        branch_code = self.branch_location.get('branch_code', 'N/A')
        branch_type = self.branch_location.get('branch_type', 'N/A')

        print(f"📊 Current State for {self.city_name}:")
        print(f"  • Branch: {branch_name} (Code: {branch_code})")
        print(f"  • Branch Type: {branch_type}")
        print(f"  • Branch Location: {self.branch_location['type']} "
              f"({self.branch_location['latitude']:.4f}, {self.branch_location['longitude']:.4f})")
        print(f"  • PODs: {self.current_state['metrics']['n_pods']}")
        print(f"  • Couriers: {self.current_state['metrics']['n_couriers']}")
        print(f"  • Total Packages: {self.current_state['metrics']['total_packages']}")
        print(f"  • Avg Packages/POD: {self.current_state['metrics']['avg_packages_per_pod']:.1f}")
        print(f"  • Workload CV: {cv_packages:.1f}%")
        print(f"  • Max/Min Ratio: {max_min_ratio:.2f}x")
        if 'distance_from_branch' in current_pods.columns:
            print(f"  • Avg Distance from Branch: {current_pods['distance_from_branch'].mean():.0f}m")

        return self.current_state

    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate distance between coordinates (in meters)
        """
        R = 6371000  # Earth radius in meters
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    def assign_pods_to_couriers(self, method='balanced'):
        """
        Step 3: Assign new PODs to couriers optimally
        """
        print("\n" + "="*80)
        print("STEP 3: ASSIGNING PODS TO COURIERS")
        print("="*80)

        courier_ids = self.current_state['couriers']['courier_id'].values
        courier_names = self.current_state['couriers']['courier_name'].values

        if method == 'balanced':
            pod_assignments, courier_workloads = self._balanced_assignment(
                courier_ids, courier_names
            )
        else:
            raise ValueError(f"Unknown assignment method: {method}")

        self.new_pods['assigned_courier_id'] = self.new_pods['pod_id'].map(pod_assignments)

        # Add courier names to new_pods
        courier_map = dict(zip(courier_ids, courier_names))
        self.new_pods['assigned_courier_name'] = self.new_pods['assigned_courier_id'].map(courier_map)

        # Create assignment summary
        self.courier_assignments = pd.DataFrame.from_dict(courier_workloads, orient='index')
        self.courier_assignments['pod_count'] = self.courier_assignments['pods'].apply(len)
        self.courier_assignments.reset_index(inplace=True)
        self.courier_assignments.rename(columns={'index': 'courier_id'}, inplace=True)

        print(f"✅ PODs assigned to {len(self.courier_assignments)} couriers")
        print(f"  • Avg PODs/courier: {self.courier_assignments['pod_count'].mean():.1f}")
        print(f"  • Avg packages/courier: {self.courier_assignments['packages'].mean():.1f}")

        return self.courier_assignments

    def _balanced_assignment(self, courier_ids, courier_names):
        """Greedy balanced assignment algorithm"""
        print("  Using Balanced Assignment Algorithm...")

        self.new_pods['workload_score'] = (
            self.new_pods['n_packages'] * 0.7 +
            self.new_pods['total_weight'] * 0.3
        )
        pods_sorted = self.new_pods.sort_values('workload_score', ascending=False)

        courier_workloads = {cid: {
            'packages': 0,
            'weight': 0,
            'pods': [],
            'workload': 0,
            'name': cname
        } for cid, cname in zip(courier_ids, courier_names)}

        pod_assignments = {}

        for _, pod in pods_sorted.iterrows():
            min_courier = min(courier_workloads.keys(),
                              key=lambda x: courier_workloads[x]['workload'])

            pod_assignments[pod['pod_id']] = min_courier
            courier_workloads[min_courier]['packages'] += pod['n_packages']
            courier_workloads[min_courier]['weight'] += pod['total_weight']
            courier_workloads[min_courier]['pods'].append(pod['pod_id'])
            courier_workloads[min_courier]['workload'] += pod['workload_score']

        return pod_assignments, courier_workloads

    def evaluate_optimization(self):
        """
        Step 4: Compare before vs after metrics.
        """
        print("\n" + "="*80)
        print("STEP 4: EVALUATING OPTIMIZATION RESULTS")
        print("="*80)

        # After metrics
        cv_after = (self.courier_assignments['packages'].std() /
                    self.courier_assignments['packages'].mean()) * 100

        max_min_after = (self.courier_assignments['packages'].max() /
                         self.courier_assignments['packages'].min())

        avg_spread_after = self.new_pods['geographic_spread'].mean()

        # Improvements
        cv_improvement = ((self.current_state['metrics']['cv_packages'] - cv_after) /
                          self.current_state['metrics']['cv_packages']) * 100

        ratio_improvement = ((self.current_state['metrics']['max_min_ratio'] - max_min_after) /
                             self.current_state['metrics']['max_min_ratio']) * 100

        pod_reduction = ((self.current_state['metrics']['n_pods'] - len(self.new_pods)) /
                         self.current_state['metrics']['n_pods']) * 100

        spread_improvement = ((self.current_state['metrics']['avg_spread'] - avg_spread_after) /
                              self.current_state['metrics']['avg_spread']) * 100

        self.optimization_metrics = {
            'before': {
                'n_pods': self.current_state['metrics']['n_pods'],
                'cv': self.current_state['metrics']['cv_packages'],
                'max_min_ratio': self.current_state['metrics']['max_min_ratio'],
                'avg_spread': self.current_state['metrics']['avg_spread']
            },
            'after': {
                'n_pods': len(self.new_pods),
                'cv': cv_after,
                'max_min_ratio': max_min_after,
                'avg_spread': avg_spread_after
            },
            'improvements': {
                'cv_improvement': cv_improvement,
                'ratio_improvement': ratio_improvement,
                'pod_reduction': pod_reduction,
                'spread_improvement': spread_improvement
            }
        }

        print("\n📊 OPTIMIZATION RESULTS:")
        print(f"  PODs: {self.optimization_metrics['before']['n_pods']} → "
              f"{self.optimization_metrics['after']['n_pods']} "
              f"({pod_reduction:+.1f}%)")
        print(f"  Workload CV: {self.optimization_metrics['before']['cv']:.1f}% → "
              f"{self.optimization_metrics['after']['cv']:.1f}% "
              f"({cv_improvement:+.1f}% improvement)")
        print(f"  Max/Min Ratio: {self.optimization_metrics['before']['max_min_ratio']:.2f}x → "
              f"{self.optimization_metrics['after']['max_min_ratio']:.2f}x "
              f"({ratio_improvement:+.1f}% improvement)")
        print(f"  Avg Spread: {self.optimization_metrics['before']['avg_spread']:.1f}m → "
              f"{self.optimization_metrics['after']['avg_spread']:.1f}m "
              f"({spread_improvement:+.1f}% improvement)")

        return self.optimization_metrics

    def create_comparison_maps(self, output_dir='maps'):
        """
        Create before/after comparison maps.
        """
        print("\n🗺️ Creating comparison maps...")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Calculate center (use branch location if available)
        if self.branch_location:
            center_lat = self.branch_location['latitude']
            center_lon = self.branch_location['longitude']
        else:
            center_lat = self.df['SELECTED_LATITUDE'].mean()
            center_lon = self.df['SELECTED_LONGITUDE'].mean()

        # Create combined HTML
        combined_html = self._create_combined_html_map(center_lat, center_lon)

        # Save the combined HTML
        output_file = os.path.join(output_dir, f'{self.city_name.lower().replace(" ", "_")}_comparison.html')
        with open(output_file, 'w') as f:
            f.write(combined_html)

        print(f"✅ Comparison map created: {output_file}")

        return output_file

    def _create_combined_html_map(self, center_lat, center_lon):
        """Create combined HTML map"""

        # Extended color palette for better differentiation
        extended_colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#FF8C00', '#32CD32', '#BA55D3', '#20B2AA',
            '#FF1493', '#00CED1', '#FFD700', '#8A2BE2', '#00FA9A',
            '#DC143C', '#00BFFF', '#FF69B4', '#1E90FF', '#FA8072',
            '#40E0D0', '#EE82EE', '#FFA500', '#9370DB', '#3CB371',
            '#FF4500', '#2E8B57', '#DA70D6', '#6495ED', '#F08080'
        ]

        # Get branch details for display
        branch_name = self.branch_location.get('branch_name', self.city_name)
        branch_code = self.branch_location.get('branch_code', '')
        branch_type = self.branch_location.get('branch_type', '')

        combined_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>POD Clustering Comparison - {self.city_name}</title>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css"/>
            <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    font-family: Arial, sans-serif;
                }}
                .header {{
                    background: linear-gradient(90deg, #ff6b6b 50%, #4ecdc4 50%);
                    color: white;
                    padding: 20px;
                    text-align: center;
                }}
                .container {{
                    display: flex;
                    height: calc(100vh - 200px);
                    position: relative;
                }}
                .map-container {{
                    flex: 1;
                    position: relative;
                }}
                #map-before, #map-after {{
                    height: 100%;
                    width: 100%;
                }}
                .divider {{
                    width: 2px;
                    background-color: #333;
                }}
                .map-title {{
                    position: absolute;
                    top: 10px;
                    left: 10px;
                    z-index: 1000;
                    background: white;
                    padding: 10px;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.3);
                }}
                .legend {{
                    position: absolute;
                    bottom: 30px;
                    right: 10px;
                    background: white;
                    padding: 10px;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.3);
                    z-index: 1000;
                    font-size: 12px;
                    max-width: 200px;
                }}
                .stats {{
                    background: white;
                    padding: 15px;
                    text-align: center;
                    border-top: 2px solid #eee;
                }}
                .branch-icon {{
                    background-color: #FFD700;
                    width: 35px;
                    height: 35px;
                    border-radius: 50%;
                    border: 4px solid #FF4500;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: bold;
                    font-size: 18px;
                    color: #FF4500;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>POD Clustering Optimization - {self.city_name}</h2>
                <small>Branch: {branch_name} {f'(Code: {branch_code})' if branch_code else ''} | Type: {self.branch_location['type'].upper()}</small>
            </div>
            <div class="container">
                <div class="map-container">
                    <div class="map-title">
                        <h4>🔴 BEFORE: {self.current_state['metrics']['n_pods']} PODs</h4>
                        <small>Current courier assignments</small>
                    </div>
                    <div id="map-before"></div>
                </div>
                <div class="divider"></div>
                <div class="map-container">
                    <div class="map-title">
                        <h4>🟢 AFTER: {len(self.new_pods)} PODs</h4>
                        <small>Optimized by geographic clusters</small>
                    </div>
                    <div id="map-after"></div>
                </div>
            </div>
            <div class="stats">
                <b>Optimization Results:</b>
                PODs: {self.optimization_metrics['before']['n_pods']} → {self.optimization_metrics['after']['n_pods']} |
                Workload Balance: {self.optimization_metrics['improvements']['cv_improvement']:.1f}% improvement |
                Geographic Efficiency: {self.optimization_metrics['improvements']['spread_improvement']:.1f}% improvement
            </div>

            <script>
                // Color palette for PODs
                var podColors = {extended_colors};

                // Initialize BEFORE map
                var mapBefore = L.map('map-before').setView([{center_lat}, {center_lon}], 13);
                L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                    attribution: '© OpenStreetMap contributors'
                }}).addTo(mapBefore);

                // Initialize AFTER map
                var mapAfter = L.map('map-after').setView([{center_lat}, {center_lon}], 13);
                L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                    attribution: '© OpenStreetMap contributors'
                }}).addTo(mapAfter);

                // Add BRANCH marker to both maps
                var branchIcon = L.divIcon({{
                    html: '<div class="branch-icon">HUB</div>',
                    iconSize: [35, 35],
                    className: 'branch-marker'
                }});

                L.marker([{self.branch_location['latitude']}, {self.branch_location['longitude']}], {{
                    icon: branchIcon
                }}).addTo(mapBefore).bindPopup('<b>BRANCH HUB: {branch_name}</b><br>Code: {branch_code if branch_code else "N/A"}<br>Type: {branch_type if branch_type else "N/A"}');

                L.marker([{self.branch_location['latitude']}, {self.branch_location['longitude']}], {{
                    icon: branchIcon
                }}).addTo(mapAfter).bindPopup('<b>BRANCH HUB: {branch_name}</b><br>Code: {branch_code if branch_code else "N/A"}<br>Type: {branch_type if branch_type else "N/A"}');

                // Add BEFORE markers
        """

        # Add JavaScript for BEFORE map markers
        pod_index = 0
        for idx, (pod_code, pod_group) in enumerate(self.df.groupby('DO_POD_DELIVER_CODE')):
            pod_color = extended_colors[pod_index % len(extended_colors)]
            pod_index += 1

            courier_name = pod_group['NICKNAME'].iloc[0]

            for _, row in pod_group.iterrows():
                distance_km = row.get('distance_from_branch_km', 0)
                combined_html += f"""
                L.circleMarker([{row['SELECTED_LATITUDE']}, {row['SELECTED_LONGITUDE']}], {{
                    radius: 4,
                    fillColor: '{pod_color}',
                    color: 'white',
                    weight: 1,
                    opacity: 1,
                    fillOpacity: 0.8
                }}).addTo(mapBefore).bindPopup(
                    '<b>POD: {pod_code[:20]}</b><br>' +
                    'Courier: {courier_name}<br>' +
                    'Weight: {row['BERATASLI']:.1f}kg<br>' +
                    'Distance: {distance_km:.1f}km'
                );
                """

        # Add JavaScript for AFTER map markers
        for idx, pod in self.new_pods.iterrows():
            pod_color = extended_colors[idx % len(extended_colors)]

            # Draw all delivery points in this POD
            for lat, lon in zip(pod['lat_points'], pod['lon_points']):
                combined_html += f"""
                L.circleMarker([{lat}, {lon}], {{
                    radius: 4,
                    fillColor: '{pod_color}',
                    color: 'white',
                    weight: 1,
                    opacity: 1,
                    fillOpacity: 0.8
                }}).addTo(mapAfter).bindPopup(
                    '<b>{pod['pod_code']}</b><br>' +
                    'Assigned to: {pod['assigned_courier_name']}<br>' +
                    'Packages: {pod['n_packages']}'
                );
                """

            # Add center marker for each new POD
            marker_size = 25 if pod['n_packages'] >= 20 else 20
            combined_html += f"""
            L.marker([{pod['lat_center']}, {pod['lon_center']}], {{
                icon: L.divIcon({{
                    html: '<div style="background-color: {pod_color}; width: {marker_size}px; height: {marker_size}px; border-radius: 50%; border: 3px solid white; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 10px; color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.3);">{idx+1}</div>',
                    iconSize: [{marker_size}, {marker_size}],
                    className: 'pod-center-marker'
                }})
            }}).addTo(mapAfter).bindPopup(
                '<b>{pod['pod_code']}</b><br>' +
                '<b>POD #{idx+1}</b><br>' +
                'Packages: {pod['n_packages']}<br>' +
                'Weight: {pod['total_weight']:.1f}kg<br>' +
                'Avg Distance: {pod.get('avg_distance_km', 0):.1f}km'
            );
            """

        combined_html += """
            </script>
        </body>
        </html>
        """

        return combined_html

    def export_assignments(self, output_dir='exports'):
        """Export POD assignments to CSV for implementation"""
        os.makedirs(output_dir, exist_ok=True)

        filename_prefix = os.path.join(output_dir, f'{self.city_name.lower().replace(" ", "_")}_optimized')

        # Export new POD definitions
        export_columns = [
            'pod_code', 'assigned_courier_id', 'assigned_courier_name',
            'n_packages', 'total_weight', 'geographic_spread', 'vehicle_type',
            'avg_distance_km'
        ]
        pod_export = self.new_pods[[col for col in export_columns if col in self.new_pods.columns]]
        pod_export.to_csv(f'{filename_prefix}_pods.csv', index=False)

        # Export detailed assignments
        assignment_details = []
        for _, pod in self.new_pods.iterrows():
            for delivery_id in pod['delivery_ids']:
                delivery = self.df[self.df['delivery_id'] == delivery_id].iloc[0]
                assignment_details.append({
                    'awb_number': delivery['AWB_NUMBER'],
                    'weight_kg': delivery['BERATASLI'],
                    'vehicle_type': delivery.get('vehicle_type', 'UNKNOWN'),
                    'distance_from_branch_km': delivery.get('distance_from_branch_km', 0),
                    'new_pod_code': pod['pod_code'],
                    'assigned_courier_id': pod['assigned_courier_id'],
                    'assigned_courier_name': pod['assigned_courier_name'],
                    'old_pod_code': delivery['DO_POD_DELIVER_CODE'],
                    'old_courier': delivery['NICKNAME']
                })

        assignments_df = pd.DataFrame(assignment_details)
        assignments_df.to_csv(f'{filename_prefix}_details.csv', index=False)

        # Export courier summary
        courier_summary_df = self.courier_assignments[['courier_id', 'name', 'packages', 'weight', 'pod_count']]
        courier_summary_df.columns = ['courier_id', 'courier_name', 'total_packages', 'total_weight_kg', 'pod_count']
        courier_summary_df.to_csv(f'{filename_prefix}_courier_summary.csv', index=False)


        print(f"\n📁 Assignments exported to {output_dir}:")
        print(f"  • {filename_prefix}_pods.csv - New POD definitions")
        print(f"  • {filename_prefix}_details.csv - Detailed AWB-level assignments")
        print(f"  • {filename_prefix}_courier_summary.csv - Courier workload summary")

        return assignments_df

class MultiFileProcessor:
    """
    Process multiple files with POD clustering optimization
    """

    def __init__(self, input_dir: str, output_dir: str = 'output'):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.results = {}
        self.branch_manager = BranchManager()

        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'maps').mkdir(exist_ok=True)
        (self.output_dir / 'exports').mkdir(exist_ok=True)
        (self.output_dir / 'reports').mkdir(exist_ok=True)

    def process_all_files(self,
                          file_pattern: str = '*.csv',
                          clustering_method: str = 'kmeans',
                          assignment_method: str = 'balanced',
                          target_pods_per_courier: int = 4,
                          max_packages_per_pod: int = 40,
                          min_packages_per_pod: int = 10,
                          branch_locations: Optional[Dict[str, Tuple[float, float]]] = None,
                          separate_heavy_packages: bool = True):
        """
        Process all matching files in the input directory
        """

        # Find all matching files
        files = list(self.input_dir.glob(file_pattern))

        if not files:
            print(f"⚠️ No files found matching pattern: {file_pattern} in {self.input_dir}")
            return

        print(f"\n{'='*80}")
        print(f"PROCESSING {len(files)} FILES")
        print(f"Parameters: {target_pods_per_courier} PODs/courier, "
              f"{max_packages_per_pod} max packages/POD, "
              f"{min_packages_per_pod} min packages/POD")
        print(f"{'='*80}")

        # Process each file
        for file_path in files:
            city_name = file_path.stem.replace('_', ' ').title()

            print(f"\n{'='*80}")
            print(f"Processing: {file_path.name}")
            print(f"City: {city_name}")
            print(f"{'='*80}")

            try:
                # Load data
                df = pd.read_csv(file_path)

                # Set branch location if provided
                branch_location = None
                if branch_locations and city_name in branch_locations:
                    lat, lon = branch_locations[city_name]
                    branch_location = {
                        'name': city_name,
                        'latitude': lat,
                        'longitude': lon,
                        'type': 'manual'
                    }

                # Run optimization
                optimizer = self._run_optimization_for_file(
                    df, city_name, branch_location,
                    clustering_method, assignment_method,
                    target_pods_per_courier, max_packages_per_pod,
                    min_packages_per_pod, separate_heavy_packages
                )

                # Store results
                self.results[city_name] = {
                    'file': file_path.name,
                    'optimizer': optimizer,
                    'metrics': optimizer.optimization_metrics
                }

                print(f"✅ Successfully processed {file_path.name}")

            except Exception as e:
                print(f"❌ Error processing {file_path.name}: {str(e)}")
                self.results[city_name] = {'error': str(e)}

        # Generate summary report
        self._generate_summary_report()

        return self.results

    def _run_optimization_for_file(self, df, city_name, branch_location,
                                  clustering_method, assignment_method,
                                  target_pods_per_courier, max_packages_per_pod,
                                  min_packages_per_pod, separate_heavy_packages):
        """Run optimization for a single file"""

        # Initialize optimizer
        optimizer = PODClusteringSystem(df, city_name, branch_location)

        # Run optimization pipeline
        optimizer.analyze_current_state()

        # Cap max pods to prevent excessive splitting
        current_pods = optimizer.current_state['metrics']['n_pods']
        max_total_pods = int(current_pods * 1.5)

        optimizer.cluster_delivery_points(
            method=clustering_method,
            target_pods_per_courier=target_pods_per_courier,
            max_packages_per_pod=max_packages_per_pod,
            min_packages_per_pod=min_packages_per_pod,
            separate_heavy_packages=separate_heavy_packages,
            max_total_pods=max_total_pods
        )
        optimizer.assign_pods_to_couriers(method=assignment_method)
        optimizer.evaluate_optimization()

        # Create outputs
        optimizer.create_comparison_maps(self.output_dir / 'maps')
        optimizer.export_assignments(self.output_dir / 'exports')

        return optimizer

    def _generate_summary_report(self):
        """Generate a summary report for all processed files"""

        report_path = self.output_dir / 'reports' / 'summary_report.txt'

        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("POD CLUSTERING OPTIMIZATION - SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")

            successful = 0
            failed = 0

            for city_name, result in self.results.items():
                f.write(f"\n{city_name}\n")
                f.write("-"*40 + "\n")

                if 'error' in result:
                    f.write(f"Status: FAILED\n")
                    f.write(f"Error: {result['error']}\n")
                    failed += 1
                else:
                    successful += 1
                    metrics = result['metrics']
                    optimizer = result['optimizer']

                    # Get branch info
                    branch_info = optimizer.branch_location

                    f.write(f"Status: SUCCESS\n")
                    f.write(f"File: {result['file']}\n")
                    f.write(f"Branch: {branch_info.get('branch_name', 'N/A')} (Code: {branch_info.get('branch_code', 'N/A')})\n")
                    f.write(f"Branch Type: {branch_info.get('branch_type', 'N/A')}\n")
                    f.write(f"Branch Location: ({branch_info['latitude']:.4f}, {branch_info['longitude']:.4f}) - {branch_info['type']}\n")
                    f.write(f"PODs: {metrics['before']['n_pods']} → {metrics['after']['n_pods']}\n")
                    f.write(f"CV Improvement: {metrics['improvements']['cv_improvement']:.1f}%\n")
                    f.write(f"Spread Improvement: {metrics['improvements']['spread_improvement']:.1f}%\n")

            f.write("\n" + "="*80 + "\n")
            f.write(f"SUMMARY: {successful} successful, {failed} failed\n")
            f.write("="*80 + "\n")

        print(f"\n📊 Summary report generated: {report_path}")

        # Also create CSV summary
        summary_data = []
        for city_name, result in self.results.items():
            if 'error' not in result:
                metrics = result['metrics']
                optimizer = result['optimizer']
                branch_info = optimizer.branch_location

                summary_data.append({
                    'city': city_name,
                    'file': result['file'],
                    'branch_name': branch_info.get('branch_name', 'N/A'),
                    'branch_code': branch_info.get('branch_code', 'N/A'),
                    'branch_type': branch_info.get('branch_type', 'N/A'),
                    'branch_lat': branch_info['latitude'],
                    'branch_lon': branch_info['longitude'],
                    'branch_location_type': branch_info['type'],
                    'pods_before': metrics['before']['n_pods'],
                    'pods_after': metrics['after']['n_pods'],
                    'cv_before': metrics['before']['cv'],
                    'cv_after': metrics['after']['cv'],
                    'cv_improvement': metrics['improvements']['cv_improvement'],
                    'spread_improvement': metrics['improvements']['spread_improvement']
                })

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_csv = self.output_dir / 'reports' / 'summary_report.csv'
            summary_df.to_csv(summary_csv, index=False)
            print(f"📊 CSV summary generated: {summary_csv}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def process_delivery_files(input_dir: str = '.', output_dir: str = 'optimization_results',
                           separate_heavy_packages: bool = False):
    """
    Process all delivery files with geographic-based clustering.

    Parameters:
    -----------
    input_dir : str
        Directory containing the CSV files.
    output_dir : str
        Directory for output results.
    separate_heavy_packages : bool
        If True, separates packages >= 7kg into their own clusters for car delivery.
    """
    # Initialize processor
    processor = MultiFileProcessor(input_dir, output_dir)

    # Process all files
    results = processor.process_all_files(
        file_pattern='*.csv',
        clustering_method='hierarchical',
        assignment_method='balanced',
        target_pods_per_courier=4,
        max_packages_per_pod=80,
        min_packages_per_pod=25,
        branch_locations=None,  # Auto-detect from data
        separate_heavy_packages=separate_heavy_packages
    )

    return results


if __name__ == "__main__":
    print("\n" + "="*80)
    print("GEOGRAPHIC POD CLUSTERING & COURIER ASSIGNMENT")
    print("="*80)
    print("\n📋 System Features:")
    print("  ✅ Clusters delivery points based on geographic proximity.")
    print("  ✅ Balances workload evenly among available couriers.")
    print("  ✅ Optionally separates heavy packages (>=7kg) for car delivery.")
    print("  ✅ Processes all CSV files in a specified folder automatically.")
    print("  ✅ Generates interactive before/after maps and detailed CSV reports.")
    print("="*80)

    # Process ALL CSV files in the folder
    # Make sure to change 'input_dir' to the folder containing your CSV files.
    results = process_delivery_files(
        input_dir='/content/sample_data/hub_sample',  # Use '.' for the current directory, or provide a path e.g., '/path/to/your/data'
        output_dir='optimization_results',
        separate_heavy_packages=False # Set to True if you want to separate car/motorcycle deliveries
    )

    # Print summary
    print("\n" + "="*80)
    print("PROCESSING COMPLETE - SUMMARY")
    print("="*80)

    if results:
        print(f"\n✅ Processed {len(results)} files:")
        for city_name, result in results.items():
            if 'error' not in result:
                optimizer = result['optimizer']
                branch = optimizer.branch_location
                metrics = result['metrics']
                pod_change = metrics['after']['n_pods'] - metrics['before']['n_pods']
                pod_change_percent = (pod_change / metrics['before']['n_pods'] * 100) if metrics['before']['n_pods'] > 0 else 0

                print(f"\n  📍 {city_name}:")
                print(f"    • Branch: {branch.get('branch_name', 'N/A')} ({branch['type']})")
                print(f"    • PODs: {metrics['before']['n_pods']} → {metrics['after']['n_pods']} ({pod_change:+d}, {pod_change_percent:+.0f}%)")
                print(f"    • Workload Balance: {metrics['improvements']['cv_improvement']:.1f}% improvement")
            else:
                print(f"\n  ❌ {city_name}: {result['error']}")

    print("\n📁 Output Directory Structure:")
    print("  optimization_results/")
    print("    ├── maps/              # Interactive comparison maps for each city")
    print("    ├── exports/           # CSV files with optimized assignments")
    print("    │   ├── *_pods.csv     # New POD definitions")
    print("    │   ├── *_details.csv  # Detailed AWB assignments")
    print("    │   └── *_courier_summary.csv # Courier workload summary")
    print("    └── reports/           # Summary reports across all cities")

    print("\n" + "="*80)
    print("✅ ALL PROCESSING COMPLETE!")
    print("="*80)