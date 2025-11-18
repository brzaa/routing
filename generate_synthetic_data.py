#!/usr/bin/env python3
"""
Synthetic Delivery Data Generator

Generates realistic delivery CSV data for testing route optimization.
Supports multiple cities and configurable delivery volumes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import argparse


# City configurations with realistic coordinates
CITIES = {
    'jakarta': {
        'name': 'Jakarta',
        'branch_lat': -6.2088,
        'branch_lon': 106.8456,
        'lat_range': (-6.35, -6.05),  # ~30km radius
        'lon_range': (106.70, 107.00),
        'branch_code': 'JKT01'
    },
    'semarang': {
        'name': 'Semarang',
        'branch_lat': -7.0051,
        'branch_lon': 110.4381,
        'lat_range': (-7.15, -6.85),
        'lon_range': (110.30, 110.60),
        'branch_code': 'SMG01'
    },
    'surabaya': {
        'name': 'Surabaya',
        'branch_lat': -7.2575,
        'branch_lon': 112.7521,
        'lat_range': (-7.40, -7.10),
        'lon_range': (112.60, 112.90),
        'branch_code': 'SBY01'
    },
    'bandung': {
        'name': 'Bandung',
        'branch_lat': -6.9175,
        'branch_lon': 107.6191,
        'lat_range': (-7.05, -6.80),
        'lon_range': (107.50, 107.75),
        'branch_code': 'BDG01'
    },
    'yogyakarta': {
        'name': 'Yogyakarta',
        'branch_lat': -7.7956,
        'branch_lon': 110.3695,
        'lat_range': (-7.90, -7.70),
        'lon_range': (110.25, 110.50),
        'branch_code': 'YOG01'
    },
    'grobogan': {
        'name': 'Grobogan',
        'branch_lat': -7.0789624,
        'branch_lon': 110.8940848,
        'lat_range': (-7.30, -6.95),
        'lon_range': (110.70, 111.10),
        'branch_code': 'GRB01'
    }
}

# Indonesian names for couriers
COURIER_NAMES = [
    'Budi Santoso', 'Ahmad Hidayat', 'Siti Nurhaliza', 'Dewi Lestari',
    'Agus Susanto', 'Rina Wijaya', 'Bambang Prasetyo', 'Indah Permata',
    'Rudi Hartono', 'Sri Mulyani', 'Eko Wibowo', 'Fitri Handayani',
    'Joko Widodo', 'Maya Sari', 'Hendra Gunawan', 'Nur Azizah',
    'Dedi Kurniawan', 'Lina Marlina', 'Arif Rahman', 'Yanti Kusuma'
]


def generate_clustered_deliveries(city_config, num_deliveries, num_clusters=5):
    """
    Generate deliveries clustered in geographic hotspots (realistic).

    Args:
        city_config: City configuration dict
        num_deliveries: Number of deliveries to generate
        num_clusters: Number of geographic clusters/hotspots

    Returns:
        List of (lat, lon) tuples
    """
    lat_min, lat_max = city_config['lat_range']
    lon_min, lon_max = city_config['lon_range']

    # Generate cluster centers
    cluster_centers = []
    for _ in range(num_clusters):
        center_lat = np.random.uniform(lat_min, lat_max)
        center_lon = np.random.uniform(lon_min, lon_max)
        cluster_centers.append((center_lat, center_lon))

    # Assign deliveries to clusters
    deliveries = []
    for _ in range(num_deliveries):
        # Pick a random cluster
        center_lat, center_lon = random.choice(cluster_centers)

        # Add some spread around the cluster center (realistic delivery density)
        spread = 0.02  # ~2km radius
        delivery_lat = center_lat + np.random.normal(0, spread)
        delivery_lon = center_lon + np.random.normal(0, spread)

        # Clamp to city bounds
        delivery_lat = np.clip(delivery_lat, lat_min, lat_max)
        delivery_lon = np.clip(delivery_lon, lon_min, lon_max)

        deliveries.append((delivery_lat, delivery_lon))

    return deliveries


def generate_synthetic_data(city='jakarta', num_deliveries=500, num_couriers=10,
                           date=None, output_file=None):
    """
    Generate synthetic delivery data CSV.

    Args:
        city: City name (jakarta, semarang, surabaya, bandung, yogyakarta, grobogan)
        num_deliveries: Number of deliveries to generate
        num_couriers: Number of couriers
        date: Delivery date (defaults to today)
        output_file: Output CSV filename (if None, returns DataFrame without saving)

    Returns:
        DataFrame with synthetic delivery data
    """
    if city not in CITIES:
        raise ValueError(f"Unknown city: {city}. Available: {list(CITIES.keys())}")

    city_config = CITIES[city]

    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')

    # Only print when saving to file (command-line usage)
    if output_file:
        print(f"Generating {num_deliveries} deliveries for {city_config['name']}...")

    # Generate delivery locations (clustered for realism)
    delivery_locations = generate_clustered_deliveries(city_config, num_deliveries)

    # Generate courier assignments
    courier_ids = [f"EMP{i+1:03d}" for i in range(num_couriers)]
    courier_names = random.sample(COURIER_NAMES, min(num_couriers, len(COURIER_NAMES)))
    if num_couriers > len(COURIER_NAMES):
        # Repeat names if we need more couriers
        courier_names.extend([f"{name} {i}" for i, name in enumerate(COURIER_NAMES[:num_couriers - len(COURIER_NAMES)])])

    # Build dataframe
    data = []
    for i, (lat, lon) in enumerate(delivery_locations):
        # Assign to random courier
        courier_idx = random.randint(0, num_couriers - 1)

        # Generate random POD code (current assignment before optimization)
        pod_code = f"POD{random.randint(1, num_couriers * 3):03d}"

        # Generate package weight (0.5 - 25 kg, realistic distribution)
        weight = round(np.random.lognormal(1.2, 0.8), 2)
        weight = np.clip(weight, 0.5, 25.0)

        row = {
            'AWB_NUMBER': f'AWB{date.replace("-", "")}{i+1:05d}',
            'EMPLOYEE_ID': courier_ids[courier_idx],
            'NICKNAME': courier_names[courier_idx],
            'DO_POD_DELIVER_CODE': pod_code,
            'BERATASLI': weight,
            'SELECTED_LATITUDE': round(lat, 7),
            'SELECTED_LONGITUDE': round(lon, 7),
            'BRANCH_LATITUDE': city_config['branch_lat'],
            'BRANCH_LONGITUDE': city_config['branch_lon'],
            'GERAI': city_config['name'],
            'BRANCH_NAME': f"{city_config['name']} Branch",
            'KODE_GERAI': city_config['branch_code'],
            'BRANCH_CODE': city_config['branch_code'],
            'DO_POD_DELIVER_DATE': date,
            'DELIVERY_DATE': date
        }
        data.append(row)

    df = pd.DataFrame(data)

    # Save to CSV if output_file specified
    if output_file:
        df.to_csv(output_file, index=False)

        print(f"✅ Generated {len(df)} deliveries")
        print(f"📊 Statistics:")
        print(f"   - Couriers: {num_couriers}")
        print(f"   - Current PODs: {df['DO_POD_DELIVER_CODE'].nunique()}")
        print(f"   - Total Weight: {df['BERATASLI'].sum():.1f} kg")
        print(f"   - Avg Weight: {df['BERATASLI'].mean():.2f} kg")
        print(f"   - Date: {date}")
        print(f"📁 Saved to: {output_file}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic delivery data for route optimization testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Generate 500 deliveries for Jakarta
  python generate_synthetic_data.py --city jakarta --deliveries 500

  # Generate 1000 deliveries for Grobogan with 15 couriers
  python generate_synthetic_data.py --city grobogan --deliveries 1000 --couriers 15

  # Generate data for specific date
  python generate_synthetic_data.py --city semarang --deliveries 300 --date 2025-01-15

Available cities: {', '.join(CITIES.keys())}
        """
    )

    parser.add_argument('--city', '-c',
                       default='jakarta',
                       choices=list(CITIES.keys()),
                       help='City to generate data for (default: jakarta)')

    parser.add_argument('--deliveries', '-d',
                       type=int,
                       default=500,
                       help='Number of deliveries to generate (default: 500)')

    parser.add_argument('--couriers', '-n',
                       type=int,
                       default=10,
                       help='Number of couriers (default: 10)')

    parser.add_argument('--date',
                       type=str,
                       default=None,
                       help='Delivery date in YYYY-MM-DD format (default: today)')

    parser.add_argument('--output', '-o',
                       type=str,
                       default=None,
                       help='Output CSV filename (default: auto-generated)')

    args = parser.parse_args()

    try:
        generate_synthetic_data(
            city=args.city,
            num_deliveries=args.deliveries,
            num_couriers=args.couriers,
            date=args.date,
            output_file=args.output
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
