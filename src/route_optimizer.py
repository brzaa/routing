"""
Route Optimization Module

Integrates TSP/VRP solving using Google OR-Tools for delivery route optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from geopy.distance import geodesic
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import requests
from pathlib import Path
import sys

# Import LKH solver if available
try:
    from src.lkh_solver import LKHSolver
    LKH_AVAILABLE = True
except ImportError:
    LKH_AVAILABLE = False

warnings.filterwarnings('ignore')


class RouteOptimizer:
    """
    Route optimizer that wraps OR-Tools TSP/VRP solver.

    Uses the clustering results from PODClusteringSystem to optimize
    delivery routes within each cluster using:
    - VRP (Vehicle Routing Problem) for clusters with multiple couriers
    - TSP (Traveling Salesman Problem) for single courier routes
    """

    def __init__(self, clustering_system,
                 use_ensemble=False,
                 road_distance_factor=1.35,
                 use_osrm=False,
                 osrm_server='http://router.project-osrm.org',
                 solver_type='ortools',
                 lkh_path='LKH'):
        """
        Initialize RouteOptimizer with clustering results.

        Args:
            clustering_system: PODClusteringSystem instance with clusters already created
            use_ensemble: If True, uses multiple solvers in parallel and picks best solution
            road_distance_factor: Multiplier for geodesic distance to approximate road distance (1.35 = 35% longer)
            use_osrm: If True, uses OSRM API for real road network distances
            osrm_server: OSRM server URL (default: public server, consider self-hosting for production)
            solver_type: TSP solver to use ('ortools', 'lkh', or 'both' for comparison)
            lkh_path: Path to LKH executable (default: 'LKH' if in PATH)
        """
        self.clustering_system = clustering_system
        self.df = clustering_system.df
        self.branch_location = clustering_system.branch_location
        self.new_pods = clustering_system.new_pods
        self.courier_assignments = clustering_system.courier_assignments

        # Configuration
        self.use_ensemble = use_ensemble
        self.road_distance_factor = road_distance_factor
        self.use_osrm = use_osrm
        self.osrm_server = osrm_server
        self.solver_type = solver_type

        # Initialize LKH solver if requested
        self.lkh_solver = None
        if solver_type in ['lkh', 'both']:
            if LKH_AVAILABLE:
                self.lkh_solver = LKHSolver(lkh_path)
                print(f"✓ LKH solver initialized")
            else:
                print(f"⚠️  LKH solver requested but not available")
                print(f"   Install from: http://webhotel4.ruc.dk/~keld/research/LKH/")
                if solver_type == 'lkh':
                    print(f"   Falling back to OR-Tools")
                    self.solver_type = 'ortools'

        # Storage for optimization results
        self.routes = {}  # Optimized routes per cluster
        self.metrics = {}  # Distance metrics per cluster
        self.optimization_status = {}  # Status of each optimization

    def solve_all_clusters(self, time_limit_seconds: int = 30) -> Dict:
        """
        Run VRP/TSP optimization for all clusters.

        Args:
            time_limit_seconds: Maximum time allowed for each cluster optimization

        Returns:
            Dictionary containing optimization results for all clusters
        """
        print("\n" + "="*80)
        print("ROUTE OPTIMIZATION - TSP/VRP SOLVING")
        print("="*80)

        if self.new_pods is None or len(self.new_pods) == 0:
            print("⚠️  No clusters to optimize. Run clustering first.")
            return {}

        # Group PODs by assigned courier to identify clusters
        courier_pods = self.new_pods.groupby('assigned_courier_id')['pod_id'].apply(list).to_dict()

        print(f"\n📊 Optimization Overview:")
        print(f"  • Total clusters to optimize: {len(courier_pods)}")
        print(f"  • Time limit per cluster: {time_limit_seconds}s")

        cluster_id = 0
        for courier_id, pod_ids in courier_pods.items():
            print(f"\n{'='*80}")
            print(f"Cluster {cluster_id}: Courier {courier_id}")
            print(f"{'='*80}")

            # Get all deliveries for this courier's PODs
            cluster_deliveries = self.df[self.df['new_pod_cluster'].isin(pod_ids)].copy()

            if len(cluster_deliveries) == 0:
                print(f"  ⚠️  No deliveries found for cluster {cluster_id}")
                cluster_id += 1
                continue

            print(f"  • Deliveries: {len(cluster_deliveries)}")
            print(f"  • PODs: {len(pod_ids)}")
            print(f"  • Total weight: {cluster_deliveries['BERATASLI'].sum():.1f} kg")

            # Solve based on cluster characteristics
            if len(cluster_deliveries) == 1:
                print(f"  ℹ️  Single delivery - no optimization needed")
                self._handle_single_delivery(cluster_id, courier_id, cluster_deliveries)
            elif len(cluster_deliveries) <= 50:
                # Solve as TSP (single courier, optimized route)
                self._solve_single_courier_tsp(cluster_id, courier_id, cluster_deliveries, time_limit_seconds)
            else:
                # For large clusters, solve as TSP with limited search
                print(f"  ℹ️  Large cluster - using heuristic solution")
                self._solve_single_courier_tsp(cluster_id, courier_id, cluster_deliveries, time_limit_seconds // 2)

            cluster_id += 1

        self._summarize_optimization()

        return {
            'routes': self.routes,
            'metrics': self.metrics,
            'status': self.optimization_status
        }

    def _handle_single_delivery(self, cluster_id: int, courier_id: str, deliveries: pd.DataFrame) -> None:
        """Handle single delivery case (no optimization needed)."""
        delivery = deliveries.iloc[0]
        branch_coords = (self.branch_location['latitude'], self.branch_location['longitude'])
        delivery_coords = (delivery['SELECTED_LATITUDE'], delivery['SELECTED_LONGITUDE'])

        # Simple route: branch -> delivery -> branch
        distance_meters = geodesic(branch_coords, delivery_coords).meters * 2  # Round trip

        self.routes[cluster_id] = {
            'courier_id': courier_id,
            'route': [branch_coords, delivery_coords, branch_coords],
            'delivery_sequence': [delivery['AWB_NUMBER']],
            'total_distance_meters': distance_meters
        }

        self.metrics[cluster_id] = {
            'optimized_distance_meters': distance_meters,
            'num_deliveries': 1
        }

        self.optimization_status[cluster_id] = 'single_delivery'

        print(f"  ✓ Route: Branch → Delivery → Branch ({distance_meters:.0f}m)")

    def _solve_single_courier_tsp(self, cluster_id: int, courier_id: str,
                                   deliveries: pd.DataFrame, time_limit: int) -> None:
        """
        Solve TSP for a single courier's deliveries.

        Args:
            cluster_id: Cluster identifier
            courier_id: Courier identifier
            deliveries: DataFrame of deliveries for this cluster
            time_limit: Time limit in seconds
        """
        print(f"  🔄 Solving TSP for {len(deliveries)} deliveries...")

        # Prepare locations
        branch_coords = (self.branch_location['latitude'], self.branch_location['longitude'])
        delivery_coords = deliveries[['SELECTED_LATITUDE', 'SELECTED_LONGITUDE']].values.tolist()
        delivery_coords = [(lat, lon) for lat, lon in delivery_coords]

        # All locations: depot (branch) + deliveries
        all_locations = [branch_coords] + delivery_coords

        # Create distance matrix
        distance_matrix = self._create_distance_matrix(all_locations)

        # Solve TSP
        try:
            solution = None
            solver_used = "OR-Tools"

            # Choose solver
            if self.solver_type == 'lkh' and self.lkh_solver:
                print(f"  🎯 Using LKH solver...")
                solution = self.lkh_solver.solve_tsp(distance_matrix, time_limit, runs=3)
                solver_used = "LKH"
            elif self.solver_type == 'both' and self.lkh_solver:
                # Run both solvers and compare
                print(f"  ⚔️  Running OR-Tools vs LKH comparison...")

                # OR-Tools solution
                if self.use_ensemble:
                    ortools_solution = self._solve_tsp_ensemble(distance_matrix, time_limit // 2)
                else:
                    ortools_solution = self._solve_tsp(distance_matrix, time_limit // 2)

                # LKH solution
                lkh_solution = self.lkh_solver.solve_tsp(distance_matrix, time_limit // 2, runs=2)

                # Pick best
                if ortools_solution and lkh_solution:
                    if lkh_solution[1] < ortools_solution[1]:
                        solution = lkh_solution
                        solver_used = "LKH"
                        improvement = ((ortools_solution[1] - lkh_solution[1]) / ortools_solution[1]) * 100
                        print(f"  🏆 LKH wins! {improvement:.1f}% better than OR-Tools")
                    else:
                        solution = ortools_solution
                        solver_used = "OR-Tools"
                        improvement = ((lkh_solution[1] - ortools_solution[1]) / lkh_solution[1]) * 100
                        print(f"  🏆 OR-Tools wins! {improvement:.1f}% better than LKH")
                elif ortools_solution:
                    solution = ortools_solution
                    solver_used = "OR-Tools (LKH failed)"
                elif lkh_solution:
                    solution = lkh_solution
                    solver_used = "LKH (OR-Tools failed)"
            else:
                # Default: OR-Tools
                if self.use_ensemble:
                    print(f"  🔀 Using ensemble solving (parallel strategies)...")
                    solution = self._solve_tsp_ensemble(distance_matrix, time_limit)
                else:
                    solution = self._solve_tsp(distance_matrix, time_limit)

            if solution:
                route_indices, total_distance = solution

                # Extract delivery sequence (excluding depot) - use AWB_NUMBER for tracking
                delivery_sequence = [deliveries.iloc[idx-1]['AWB_NUMBER'] for idx in route_indices if idx > 0]

                # Fetch road geometry if using OSRM
                route_geometry = None
                if self.use_osrm:
                    route_geometry = self._fetch_route_geometry([all_locations[idx] for idx in route_indices])

                # Store route
                self.routes[cluster_id] = {
                    'courier_id': courier_id,
                    'route': [all_locations[idx] for idx in route_indices],
                    'delivery_sequence': delivery_sequence,
                    'total_distance_meters': total_distance,
                    'route_geometry': route_geometry,  # Actual road path coordinates
                    'solver_used': solver_used
                }

                self.metrics[cluster_id] = {
                    'optimized_distance_meters': total_distance,
                    'num_deliveries': len(delivery_coords)
                }

                self.optimization_status[cluster_id] = 'success'

                ensemble_label = " (best of ensemble)" if self.use_ensemble and solver_used == "OR-Tools" else ""
                print(f"  ✓ TSP solved with {solver_used}{ensemble_label}: {total_distance:.0f}m total distance")
            else:
                print(f"  ⚠️  TSP failed - using fallback route")
                self._create_fallback_route(cluster_id, courier_id, deliveries, all_locations)

        except Exception as e:
            print(f"  ❌ TSP error: {str(e)}")
            self._create_fallback_route(cluster_id, courier_id, deliveries, all_locations)

    def _solve_tsp(self, distance_matrix: List[List[int]], time_limit: int,
                   first_solution_strategy=None, local_search=None) -> Optional[Tuple[List[int], float]]:
        """
        Solve TSP using OR-Tools.

        Args:
            distance_matrix: Distance matrix in meters (integers)
            time_limit: Time limit in seconds
            first_solution_strategy: OR-Tools first solution strategy (optional)
            local_search: OR-Tools local search metaheuristic (optional)

        Returns:
            Tuple of (route_indices, total_distance) or None if no solution
        """
        num_locations = len(distance_matrix)

        # Create routing index manager
        manager = pywrapcp.RoutingIndexManager(num_locations, 1, 0)  # 1 vehicle, depot at 0

        # Create routing model
        routing = pywrapcp.RoutingModel(manager)

        # Create distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            first_solution_strategy or routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            local_search or routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.FromSeconds(time_limit)

        # Solve
        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            # Extract route
            route_indices = []
            index = routing.Start(0)
            while not routing.IsEnd(index):
                route_indices.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            route_indices.append(manager.IndexToNode(index))  # Add final depot

            total_distance = solution.ObjectiveValue()

            return route_indices, total_distance

        return None

    def _solve_tsp_ensemble(self, distance_matrix: List[List[int]], time_limit: int) -> Optional[Tuple[List[int], float]]:
        """
        Solve TSP using ensemble of multiple strategies in parallel.

        Args:
            distance_matrix: Distance matrix in meters (integers)
            time_limit: Time limit in seconds (each solver runs for full time in parallel)

        Returns:
            Best solution from all strategies (route_indices, total_distance) or None
        """
        strategies = [
            {
                'name': 'PATH_CHEAPEST_ARC + GUIDED_LOCAL_SEARCH',
                'first_solution': routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
                'local_search': routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
            },
            {
                'name': 'GLOBAL_CHEAPEST_ARC + SIMULATED_ANNEALING',
                'first_solution': routing_enums_pb2.FirstSolutionStrategy.GLOBAL_CHEAPEST_ARC,
                'local_search': routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING
            },
            {
                'name': 'LOCAL_CHEAPEST_INSERTION + TABU_SEARCH',
                'first_solution': routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION,
                'local_search': routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH
            }
        ]

        solutions = []

        # Run all strategies in parallel
        with ThreadPoolExecutor(max_workers=len(strategies)) as executor:
            futures = {}
            for strategy in strategies:
                future = executor.submit(
                    self._solve_tsp,
                    distance_matrix,
                    time_limit,
                    strategy['first_solution'],
                    strategy['local_search']
                )
                futures[future] = strategy['name']

            # Collect results as they complete
            for future in as_completed(futures):
                strategy_name = futures[future]
                try:
                    result = future.result()
                    if result:
                        solutions.append((result, strategy_name))
                except Exception as e:
                    print(f"    ⚠️ Strategy {strategy_name} failed: {str(e)}")

        # Return best solution
        if solutions:
            best_solution, best_strategy = min(solutions, key=lambda x: x[0][1])
            print(f"    ✓ Best strategy: {best_strategy}")
            return best_solution

        return None

    def _create_distance_matrix(self, locations: List[Tuple[float, float]]) -> List[List[int]]:
        """
        Create distance matrix from list of coordinate tuples.

        Args:
            locations: List of (latitude, longitude) tuples

        Returns:
            Distance matrix in meters (integers for OR-Tools)
        """
        if self.use_osrm:
            return self._create_distance_matrix_osrm(locations)
        else:
            return self._create_distance_matrix_geodesic(locations)

    def _create_distance_matrix_geodesic(self, locations: List[Tuple[float, float]]) -> List[List[int]]:
        """
        Create distance matrix using geodesic distance with road correction factor.

        Args:
            locations: List of (latitude, longitude) tuples

        Returns:
            Distance matrix in meters (integers for OR-Tools)
        """
        num_locations = len(locations)
        distance_matrix = [[0] * num_locations for _ in range(num_locations)]

        for i in range(num_locations):
            for j in range(num_locations):
                if i != j:
                    straight_line_distance = geodesic(locations[i], locations[j]).meters
                    # Apply road distance correction factor
                    road_distance = straight_line_distance * self.road_distance_factor
                    distance_matrix[i][j] = int(road_distance)

        return distance_matrix

    def _create_distance_matrix_osrm(self, locations: List[Tuple[float, float]]) -> List[List[int]]:
        """
        Create distance matrix using OSRM (Open Source Routing Machine) for real road network distances.

        Args:
            locations: List of (latitude, longitude) tuples

        Returns:
            Distance matrix in meters (integers for OR-Tools)
        """
        num_locations = len(locations)

        # OSRM expects lon,lat format (reversed from our lat,lon)
        coords_str = ';'.join([f"{lon},{lat}" for lat, lon in locations])

        try:
            url = f"{self.osrm_server}/table/v1/driving/{coords_str}"
            params = {'annotations': 'distance'}

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if data.get('code') == 'Ok':
                # OSRM returns distance matrix directly
                osrm_distances = data['distances']

                # Convert to integer matrix
                distance_matrix = [[int(osrm_distances[i][j]) for j in range(num_locations)]
                                   for i in range(num_locations)]

                return distance_matrix
            else:
                print(f"  ⚠️ OSRM API error: {data.get('message', 'Unknown error')}, falling back to geodesic")
                return self._create_distance_matrix_geodesic(locations)

        except requests.exceptions.RequestException as e:
            print(f"  ⚠️ OSRM request failed: {str(e)}, falling back to geodesic")
            return self._create_distance_matrix_geodesic(locations)
        except Exception as e:
            print(f"  ⚠️ OSRM error: {str(e)}, falling back to geodesic")
            return self._create_distance_matrix_geodesic(locations)

    def _fetch_route_geometry(self, route_locations: List[Tuple[float, float]]) -> Optional[List[Tuple[float, float]]]:
        """
        Fetch actual road geometry from OSRM for visualization.

        Args:
            route_locations: Ordered list of (latitude, longitude) tuples for the route

        Returns:
            List of (latitude, longitude) tuples representing the actual road path, or None if failed
        """
        if len(route_locations) < 2:
            return None

        try:
            # OSRM expects lon,lat format
            coords_str = ';'.join([f"{lon},{lat}" for lat, lon in route_locations])

            url = f"{self.osrm_server}/route/v1/driving/{coords_str}"
            params = {
                'overview': 'full',  # Get full geometry
                'geometries': 'geojson'  # Use GeoJSON format (easier to parse)
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if data.get('code') == 'Ok' and 'routes' in data:
                # Extract geometry coordinates
                geometry = data['routes'][0]['geometry']['coordinates']
                # Convert from [lon, lat] to (lat, lon)
                road_path = [(lat, lon) for lon, lat in geometry]
                return road_path
            else:
                print(f"  ⚠️ Could not fetch route geometry: {data.get('message', 'Unknown error')}")
                return None

        except Exception as e:
            print(f"  ⚠️ Failed to fetch route geometry: {str(e)}")
            return None

    def _create_fallback_route(self, cluster_id: int, courier_id: str,
                                deliveries: pd.DataFrame, all_locations: List) -> None:
        """Create a simple fallback route when optimization fails."""
        # Simple route: visit deliveries in order, return to branch
        branch_coords = all_locations[0]
        total_distance = 0

        # Calculate distance for sequential visits
        current_location = branch_coords
        for delivery_coords in all_locations[1:]:
            total_distance += geodesic(current_location, delivery_coords).meters
            current_location = delivery_coords

        # Return to branch
        total_distance += geodesic(current_location, branch_coords).meters

        self.routes[cluster_id] = {
            'courier_id': courier_id,
            'route': all_locations + [branch_coords],
            'delivery_sequence': deliveries['AWB_NUMBER'].tolist(),
            'total_distance_meters': total_distance
        }

        self.metrics[cluster_id] = {
            'optimized_distance_meters': total_distance,
            'num_deliveries': len(deliveries)
        }

        self.optimization_status[cluster_id] = 'fallback'

        print(f"  ℹ️  Using fallback route: {total_distance:.0f}m")

    def _calculate_baseline_distance(self) -> float:
        """
        Calculate baseline distance (current state).

        Assumes couriers return to branch after each delivery:
        branch -> delivery1 -> branch -> delivery2 -> branch, etc.

        Returns:
            Total baseline distance in meters
        """
        branch_coords = (self.branch_location['latitude'], self.branch_location['longitude'])
        total_distance = 0

        for _, delivery in self.df.iterrows():
            delivery_coords = (delivery['SELECTED_LATITUDE'], delivery['SELECTED_LONGITUDE'])
            # Round trip: branch -> delivery -> branch
            total_distance += geodesic(branch_coords, delivery_coords).meters * 2

        return total_distance

    def get_total_distance_savings(self) -> Dict:
        """
        Calculate overall distance improvement.

        Returns:
            Dictionary with baseline, optimized, and savings metrics
        """
        baseline_distance_meters = self._calculate_baseline_distance()
        optimized_distance_meters = sum(
            m['optimized_distance_meters'] for m in self.metrics.values()
        )

        savings_meters = baseline_distance_meters - optimized_distance_meters
        savings_percent = (savings_meters / baseline_distance_meters * 100) if baseline_distance_meters > 0 else 0

        return {
            'baseline_distance_km': baseline_distance_meters / 1000,
            'optimized_distance_km': optimized_distance_meters / 1000,
            'savings_km': savings_meters / 1000,
            'savings_percent': savings_percent,
            'baseline_distance_meters': baseline_distance_meters,
            'optimized_distance_meters': optimized_distance_meters,
            'savings_meters': savings_meters
        }

    def _summarize_optimization(self) -> None:
        """Print summary of optimization results."""
        print("\n" + "="*80)
        print("OPTIMIZATION SUMMARY")
        print("="*80)

        total_clusters = len(self.routes)
        successful = sum(1 for s in self.optimization_status.values() if s == 'success')
        fallback = sum(1 for s in self.optimization_status.values() if s == 'fallback')
        single = sum(1 for s in self.optimization_status.values() if s == 'single_delivery')

        print(f"  • Total clusters optimized: {total_clusters}")
        print(f"  • Successful TSP solutions: {successful}")
        print(f"  • Fallback routes: {fallback}")
        print(f"  • Single deliveries: {single}")

        if self.metrics:
            total_optimized_distance = sum(m['optimized_distance_meters'] for m in self.metrics.values())
            avg_distance = total_optimized_distance / total_clusters if total_clusters > 0 else 0
            print(f"  • Total optimized distance: {total_optimized_distance/1000:.1f} km")
            print(f"  • Average distance per cluster: {avg_distance/1000:.1f} km")

        savings = self.get_total_distance_savings()
        print(f"\n  📊 Distance Comparison:")
        print(f"  • Baseline (current): {savings['baseline_distance_km']:.1f} km")
        print(f"  • Optimized: {savings['optimized_distance_km']:.1f} km")
        print(f"  • Savings: {savings['savings_km']:.1f} km ({savings['savings_percent']:.1f}%)")

    def export_routes_csv(self, output_dir: str) -> None:
        """
        Export optimized routes to CSV files.

        Args:
            output_dir: Directory to save route CSV files
        """
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create detailed routes export
        route_details = []

        for cluster_id, route_data in self.routes.items():
            courier_id = route_data['courier_id']
            delivery_sequence = route_data['delivery_sequence']
            total_distance = route_data['total_distance_meters']

            for seq_num, awb_number in enumerate(delivery_sequence, 1):
                delivery = self.df[self.df['AWB_NUMBER'] == awb_number].iloc[0]

                route_details.append({
                    'cluster_id': cluster_id,
                    'courier_id': courier_id,
                    'sequence_number': seq_num,
                    'awb_number': awb_number,
                    'latitude': delivery['SELECTED_LATITUDE'],
                    'longitude': delivery['SELECTED_LONGITUDE'],
                    'weight_kg': delivery['BERATASLI'],
                    'total_cluster_distance_km': total_distance / 1000
                })

        routes_df = pd.DataFrame(route_details)
        output_file = output_path / 'optimized_routes.csv'
        routes_df.to_csv(output_file, index=False)

        print(f"\n💾 Optimized routes exported to: {output_file}")

        # Create cluster summary
        cluster_summary = []
        for cluster_id, route_data in self.routes.items():
            cluster_summary.append({
                'cluster_id': cluster_id,
                'courier_id': route_data['courier_id'],
                'num_deliveries': len(route_data['delivery_sequence']),
                'total_distance_km': route_data['total_distance_meters'] / 1000,
                'optimization_status': self.optimization_status.get(cluster_id, 'unknown')
            })

        summary_df = pd.DataFrame(cluster_summary)
        summary_file = output_path / 'cluster_summary.csv'
        summary_df.to_csv(summary_file, index=False)

        print(f"💾 Cluster summary exported to: {summary_file}")


if __name__ == "__main__":
    print("RouteOptimizer module - use via main.py pipeline")
