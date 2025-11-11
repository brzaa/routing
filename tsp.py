from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from geopy.distance import geodesic

# Get branch location (assuming only one branch in this data)
branch_lat = df['BRANCH_LATITUDE'].iloc[0]
branch_lon = df['BRANCH_LONGITUDE'].iloc[0]
branch_location = (branch_lat, branch_lon)

# Identify unique clusters (excluding noise)
unique_clusters = df['cluster_label'].unique()
relevant_clusters = unique_clusters[unique_clusters != -1]

# Define a simplified distance callback (Euclidean distance for demonstration)
def distance_callback(from_index, to_index, manager, data):
    # Convert from routing variable index to data index
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    # Check if nodes are within bounds of data['locations']
    if from_node < len(data['locations']) and to_node < len(data['locations']):
        return int(geodesic(data['locations'][from_node], data['locations'][to_node]).meters) # Use meters for integer distance
    return 0 # Return 0 distance for invalid indices

# Define a demand callback (package weight)
def demand_callback(from_index, manager, data):
    # Convert from routing variable index to data index
    from_node = manager.IndexToNode(from_index)
     # Check if node is within bounds of data['demands']
    if from_node < len(data['demands']):
        return data['demands'][from_node]
    return 0 # Return 0 demand for invalid index

# Function to get the route for a vehicle
def get_route(data, manager, routing, solution, vehicle_id):
    route = []
    index = routing.Start(vehicle_id)
    while not routing.IsEnd(index):
        node_index = manager.IndexToNode(index)
        route.append(data['locations'][node_index])
        index = solution.Value(routing.NextVar(index))
    node_index = manager.IndexToNode(index)
    route.append(data['locations'][node_index])
    return route

# Store solutions and routes
optimization_results = {}
optimized_routes = {}

# Iterate through each relevant cluster (VRP or TSP)
# Re-identifying unique clusters and outliers for solving
unique_clusters = df['cluster_label'].unique()
relevant_clusters = unique_clusters[unique_clusters != -1]
outlier_df = df[df['cluster_label'] == -1].copy()
outlier_couriers = outlier_df['EMPLOYEE_ID'].unique()

# Solve for each relevant cluster (VRP) - Skip Cluster 0 due to size
for cluster_label in relevant_clusters:
    if cluster_label == 0:
        print(f"\n--- Skipping Cluster {cluster_label} (too large for direct VRP solution) ---")
        optimization_results[f'Cluster_{cluster_label}_VRP'] = {
            'status': 'skipped',
            'message': 'Cluster too large for direct VRP solution within time limit',
            'num_vehicles': len(df[df['cluster_label'] == cluster_label]['EMPLOYEE_ID'].unique()),
            'num_locations': len(df[df['cluster_label'] == cluster_label])
        }
        continue # Skip to the next cluster

    print(f"\n--- Solving Cluster {cluster_label} ---")

    cluster_df = df[df['cluster_label'] == cluster_label].copy()

    # Extract delivery locations and demands (weights)
    delivery_locations = cluster_df[['SELECTED_LATITUDE', 'SELECTED_LONGITUDE']].values.tolist()
    demands = cluster_df['BERATASLI'].values.tolist()

    # Add branch location as the depot (index 0)
    all_locations = [branch_location] + delivery_locations
    all_demands = [0] + demands # Depot has no demand

    # Determine couriers assigned to this cluster
    cluster_couriers = cluster_df['EMPLOYEE_ID'].unique()
    num_couriers = len(cluster_couriers)

    if num_couriers > 1 and len(delivery_locations) > 0:
        print("Solving Vehicle Routing Problem (VRP)")

        # VRP Data Model
        data = {}
        data['locations'] = all_locations
        data['demands'] = all_demands
        data['num_vehicles'] = num_couriers
        data['depot'] = 0 # Depot is the branch location

        # Vehicle Capacities (using the heuristic from formulation)
        max_total_weight = cluster_df['BERATASLI'].sum()
        avg_weight_per_vehicle = max_total_weight / num_couriers if num_couriers > 0 else max_total_weight
        data['vehicle_capacities'] = [int(avg_weight_per_vehicle * 1.5) + 1] * data['num_vehicles']

        # Create the routing index manager
        manager = pywrapcp.RoutingIndexManager(len(data['locations']),
                                               data['num_vehicles'], data['depot'])

        # Create Routing Model
        routing = pywrapcp.RoutingModel(manager)

        # Create and register callbacks
        transit_callback_index = routing.RegisterTransitCallback(
            lambda from_index, to_index: distance_callback(from_index, to_index, manager, data))
        demand_callback_index = routing.RegisterUnaryTransitCallback(
            lambda from_index: demand_callback(from_index, manager, data))


        # Define cost of each arc
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Capacity constraint
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            data['vehicle_capacities'],  # vehicle capacities
            True,  # start cumul to zero
            'Capacity')

        # Setting search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.FromSeconds(10) # Set a time limit for solving


        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)

        # Print solution and store results
        if solution:
            print("Solution found.")
            optimization_results[f'Cluster_{cluster_label}_VRP'] = {
                'status': 'success',
                'objective_value_meters': solution.ObjectiveValue(),
                'num_vehicles': data['num_vehicles'],
                'num_locations': len(data['locations']) -1 # Excluding depot
            }
            # Store routes
            cluster_routes = {}
            for vehicle_id in range(data['num_vehicles']):
                 cluster_routes[f'vehicle_{vehicle_id}'] = get_route(data, manager, routing, solution, vehicle_id)
            optimized_routes[f'Cluster_{cluster_label}_VRP'] = cluster_routes

            print(f"Total optimized distance for Cluster {cluster_label}: {solution.ObjectiveValue()} meters")
        else:
            print("No solution found.")
            optimization_results[f'Cluster_{cluster_label}_VRP'] = {
                'status': 'failure',
                'message': 'No solution found',
                'num_vehicles': data['num_vehicles'],
                 'num_locations': len(data['locations']) -1 # Excluding depot
            }
    else:
        print("Skipping solving for this cluster (no deliveries or only one courier identified).")

# Solve for outlier deliveries (TSP per courier)
if not outlier_df.empty:
    print("\n--- Solving Outlier Deliveries ---")

    for courier_id in outlier_couriers:
        courier_outliers_df = outlier_df[outlier_df['EMPLOYEE_ID'] == courier_id].copy()

        if not courier_outliers_df.empty:
            print(f"\nSolving problem for Outliers assigned to Courier {courier_id}")

            outlier_delivery_locations = courier_outliers_df[['SELECTED_LATITUDE', 'SELECTED_LONGITUDE']].values.tolist()

            # Add branch location as the starting/ending point (depot)
            outlier_all_locations = [branch_location] + outlier_delivery_locations

            if len(outlier_delivery_locations) > 0:
                print(f"Solving Traveling Salesperson Problem (TSP) for Courier {courier_id}'s outliers")

                # TSP Data Model
                data = {}
                data['locations'] = outlier_all_locations
                data['num_vehicles'] = 1 # Single courier
                data['depot'] = 0 # Depot is the branch location

                # Generate distance matrix
                num_locations = len(data['locations'])
                distance_matrix = [[0] * num_locations for _ in range(num_locations)]
                for i in range(num_locations):
                    for j in range(num_locations):
                        if i != j:
                            distance_matrix[i][j] = int(geodesic(data['locations'][i], data['locations'][j]).meters)

                data['distance_matrix'] = distance_matrix

                # Create the routing index manager
                manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                                       data['num_vehicles'], data['depot'])

                # Create Routing Model
                routing = pywrapcp.RoutingModel(manager)

                # Create and register a transit callback
                def distance_matrix_callback(from_index, to_index):
                    # Convert from routing variable index to distance matrix index
                    from_node = manager.IndexToNode(from_index)
                    to_node = manager.IndexToNode(to_index)
                    # Check if nodes are within bounds of data['distance_matrix']
                    if from_node < len(data['distance_matrix']) and to_node < len(data['distance_matrix'][0]):
                         return data['distance_matrix'][from_node][to_node]
                    return 0 # Return 0 distance for invalid indices


                transit_callback_index = routing.RegisterTransitCallback(distance_matrix_callback)

                # Define cost of each arc
                routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

                 # Setting search parameters
                search_parameters = pywrapcp.DefaultRoutingSearchParameters()
                search_parameters.first_solution_strategy = (
                    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
                search_parameters.local_search_metaheuristic = (
                    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
                search_parameters.time_limit.FromSeconds(10) # Set a time limit for solving


                # Solve the problem.
                solution = routing.SolveWithParameters(search_parameters)

                # Print solution and store results
                if solution:
                    print("Solution found.")
                    optimization_results[f'Outliers_Courier_{courier_id}_TSP'] = {
                        'status': 'success',
                        'objective_value_meters': solution.ObjectiveValue(),
                        'num_locations': len(data['locations']) -1 # Excluding depot
                    }
                     # Store route
                    optimized_routes[f'Outliers_Courier_{courier_id}_TSP_route'] = get_route(data, manager, routing, solution, 0) # TSP has only one vehicle (0)

                    print(f"Total optimized distance for Outliers assigned to Courier {courier_id}: {solution.ObjectiveValue()} meters")
                else:
                    print("No solution found.")
                    optimization_results[f'Outliers_Courier_{courier_id}_TSP'] = {
                        'status': 'failure',
                        'message': 'No solution found',
                         'num_locations': len(data['locations']) -1 # Excluding depot
                    }
            else:
                print(f"No outlier deliveries for Courier {courier_id} to formulate a problem.")

# Print overall summary of optimization results
print("\n--- Overall Optimization Results Summary ---")
for problem_id, result in optimization_results.items():
    print(f"\nProblem: {problem_id}")
    print(f"Status: {result['status']}")
    if result['status'] == 'success':
        print(f"Optimized Total Distance: {result['objective_value_meters']} meters")
        if 'num_vehicles' in result:
             print(f"Number of Vehicles: {result['num_vehicles']}")
        print(f"Number of Delivery Locations (excl. depot): {result['num_locations']}")
    elif result['status'] == 'skipped':
        print(f"Message: {result['message']}")
        if 'num_vehicles' in result:
             print(f"Number of Vehicles: {result['num_vehicles']}")
        print(f"Number of Delivery Locations: {result['num_locations']}")
    else:
        print(f"Message: {result['message']}")
        if 'num_vehicles' in result:
             print(f"Number of Vehicles: {result['num_vehicles']}")
        print(f"Number of Delivery Locations (excl. depot): {result['num_locations']}")

# Initialize a dictionary to store comparison results
comparison_results = {}

# Iterate through the optimization results
for problem_id, result in optimization_results.items():
    if result['status'] == 'success':
        print(f"\n--- Comparing Current and Optimized Routes for {problem_id} ---")

        # Determine which deliveries are part of this problem instance
        if 'Cluster_' in problem_id:
            # This is a cluster VRP
            cluster_label = int(problem_id.split('_')[1])
            current_deliveries_df = df[df['cluster_label'] == cluster_label].copy()
        elif 'Outliers_Courier_' in problem_id:
            # This is an outlier TSP for a specific courier
            courier_id = problem_id.split('_')[2]
            current_deliveries_df = df[(df['cluster_label'] == -1) & (df['EMPLOYEE_ID'] == courier_id)].copy()
        else:
            print(f"Unknown problem type: {problem_id}. Skipping comparison.")
            continue

        # 1. Calculate the total distance of the current routes
        # Approximate current total distance by summing the distance_km for deliveries
        # This is an approximation as the current routes are not explicitly defined as tours
        current_total_distance_km = current_deliveries_df['distance_km'].sum()
        current_total_distance_meters = current_total_distance_km * 1000 # Convert to meters

        print(f"Approximate Current Total Distance: {current_total_distance_km:.2f} km ({current_total_distance_meters:.2f} meters)")

        # 2. Retrieve the optimized total distance in meters
        optimized_total_distance_meters = result['objective_value_meters']
        print(f"Optimized Total Distance: {optimized_total_distance_meters:.2f} meters")

        # 4. Compare and 5. Calculate percentage improvement
        distance_difference = current_total_distance_meters - optimized_total_distance_meters
        percentage_improvement = (distance_difference / current_total_distance_meters) * 100 if current_total_distance_meters > 0 else 0

        print(f"Distance Difference (Current - Optimized): {distance_difference:.2f} meters")
        print(f"Percentage Improvement: {percentage_improvement:.2f}%")

        # Store results
        comparison_results[problem_id] = {
            'current_distance_meters': current_total_distance_meters,
            'optimized_distance_meters': optimized_total_distance_meters,
            'distance_difference_meters': distance_difference,
            'percentage_improvement': percentage_improvement,
            'num_locations': result['num_locations']
        }
    elif result['status'] == 'skipped':
         print(f"\n--- Skipping Comparison for {problem_id} (Problem skipped) ---")
         comparison_results[problem_id] = {
            'status': 'skipped',
            'message': result['message'],
            'num_locations': result['num_locations']
        }
    else:
        print(f"\n--- Skipping Comparison for {problem_id} (Problem failed to solve) ---")
        comparison_results[problem_id] = {
            'status': 'failure',
            'message': result['message'],
            'num_locations': result['num_locations']
        }


# 6. Summarize the findings
print("\n\n--- Summary of Optimization Impact Analysis ---")
print("Comparison of Approximate Current Route Distances vs. Optimized Route Distances:")

successful_optimizations = {k: v for k, v in comparison_results.items() if v.get('status') != 'skipped' and v.get('status') != 'failure'}

if successful_optimizations:
    for problem_id, data in successful_optimizations.items():
        print(f"\nProblem: {problem_id} ({data['num_locations']} deliveries)")
        print(f"  Approximate Current Distance: {data['current_distance_meters']:.2f} meters")
        print(f"  Optimized Distance: {data['optimized_distance_meters']:.2f} meters")
        print(f"  Distance Improvement: {data['distance_difference_meters']:.2f} meters ({data['percentage_improvement']:.2f}%)")

    # Highlight which types of delivery sets showed the most significant potential
    most_improved_problem = max(successful_optimizations, key=lambda k: successful_optimizations[k]['percentage_improvement'])
    print(f"\nProblem with the highest percentage improvement: {most_improved_problem} ({successful_optimizations[most_improved_problem]['percentage_improvement']:.2f}%)")

else:
    print("\nNo successful optimizations to compare.")


# 7. Discuss limitations
print("\n--- Limitations of this Comparison ---")
print("- The 'current route distance' is approximated by summing straight-line distances from the branch to each delivery point.")
print("  Actual current routes are likely more complex, involving sequences of deliveries, and their true total distance is unknown without actual route data.")
print("- This comparison focuses solely on distance.")
print("  Optimization of real-world delivery routes also considers time, traffic, vehicle capacity constraints, time windows, and courier availability, which were simplified or not included in the current optimization models.")
print("- The largest cluster (Cluster 0) could not be solved due to its size and computational complexity within the given constraints.")
print("  Therefore, the potential optimization impact on the majority of deliveries could not be assessed in this analysis.")
print("- The capacity constraint for the VRP was based on a heuristic (1.5 * average weight per vehicle) rather than actual vehicle capacities, which might not reflect real-world constraints accurately.")
print("- The TSP/VRP models assume a simple structure (start and end at the depot) and do not account for potential complex operational aspects like mid-route pickups or multiple depots.")

import plotly.graph_objects as go

# Get branch location for centering the map (assuming only one branch in this data)
branch_lat = df['BRANCH_LATITUDE'].iloc[0]
branch_lon = df['BRANCH_LONGITUDE'].iloc[0]
branch_name = df['GERAI'].iloc[0]

# Iterate through the optimized routes and visualize each one
for problem_id, routes in optimized_routes.items():
    print(f"\nVisualizing optimized routes for: {problem_id}")

    fig = go.Figure()

    # Add branch location as a distinct marker
    fig.add_trace(go.Scattermapbox(
        lat=[branch_lat], lon=[branch_lon],
        mode='markers',
        marker=dict(size=15, color='red'),
        name='Branch Location'
    ))

    # Handle both dictionary of routes (VRP) and single list route (TSP)
    if isinstance(routes, dict):
        # This is a VRP with multiple vehicles
        for vehicle_id, route in routes.items():
            if len(route) > 1: # Only plot routes with more than one location
                route_lats = [loc[0] for loc in route]
                route_lons = [loc[1] for loc in route]

                fig.add_trace(go.Scattermapbox(
                    lat=route_lats,
                    lon=route_lons,
                    mode='lines+markers',
                    marker=go.scattermapbox.Marker(size=8),
                    line=dict(width=2),
                    name=f'{problem_id} - {vehicle_id}'
                ))
    elif isinstance(routes, list):
        # This is a TSP with a single route
        route = routes
        if len(route) > 1: # Only plot routes with more than one location
            route_lats = [loc[0] for loc in route]
            route_lons = [loc[1] for loc in route]

            fig.add_trace(go.Scattermapbox(
                lat=route_lats,
                lon=route_lons,
                mode='lines+markers',
                marker=go.scattermapbox.Marker(size=8),
                line=dict(width=2),
                name=f'{problem_id}' # Use problem_id as name for TSP
            ))


    # Add original delivery locations for context (optional, but helpful)
    # Filter the original data based on the problem_id to show relevant delivery points
    if 'Cluster_' in problem_id:
        cluster_label = int(problem_id.split('_')[1])
        delivery_points_df = df[df['cluster_label'] == cluster_label].copy()
        title_text = f'Optimized Routes for Cluster {cluster_label}'
    elif 'Outliers_Courier_' in problem_id:
        # For outliers, the problem_id might end with '_TSP_route'
        # Need to get the courier ID from the original problem_id format
        parts = problem_id.split('_')
        if len(parts) >= 3:
             courier_id = parts[2]
             delivery_points_df = df[(df['cluster_label'] == -1) & (df['EMPLOYEE_ID'] == courier_id)].copy()
             title_text = f'Optimized Routes for Outliers (Courier {courier_id})'
        else:
            delivery_points_df = pd.DataFrame() # Empty DataFrame if problem_id format is unexpected
            title_text = f'Optimized Routes for {problem_id}'
    else:
        delivery_points_df = pd.DataFrame() # Empty DataFrame if problem_id is not recognized
        title_text = f'Optimized Routes for {problem_id}'


    if not delivery_points_df.empty:
         fig.add_trace(go.Scattermapbox(
            lat=delivery_points_df['SELECTED_LATITUDE'],
            lon=delivery_points_df['SELECTED_LONGITUDE'],
            mode='markers',
            marker=go.scattermapbox.Marker(size=5, color='blue', opacity=0.6),
            name='Delivery Locations'
        ))


    # Customize map appearance
    fig.update_layout(
        title=title_text,
        mapbox_style="open-street-map",
        mapbox_center_lon=branch_lon,
        mapbox_center_lat=branch_lat,
        mapbox_zoom=10,
        margin={"r":0,"t":50,"l":0,"b":0}
    )

    fig.show()