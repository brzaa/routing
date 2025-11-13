#!/usr/bin/env python3
"""Test OSRM connectivity and debug issues."""

import requests
import sys

def test_osrm_connection(osrm_server="http://router.project-osrm.org"):
    """Test basic OSRM connectivity."""
    # Remove trailing slash to avoid double slashes in URLs
    osrm_server = osrm_server.rstrip('/')
    print(f"Testing OSRM server: {osrm_server}")
    print("=" * 60)

    # Test coordinates (Jakarta area)
    test_coords = [
        (-6.2088, 106.8456),  # lat, lon
        (-6.2044, 106.8478)
    ]

    # Convert to OSRM format (lon,lat)
    coords_str = ';'.join([f"{lon},{lat}" for lat, lon in test_coords])

    # Test 1: Table API (for distance matrix)
    print("\n1. Testing Table API (distance matrix)...")
    try:
        url = f"{osrm_server}/table/v1/driving/{coords_str}"
        params = {'annotations': 'distance'}

        print(f"   URL: {url}")
        print(f"   Params: {params}")

        response = requests.get(url, params=params, timeout=30)
        print(f"   Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            if data.get('code') == 'Ok':
                print("   ✅ Table API working!")
                print(f"   Distance matrix: {data['distances']}")
            else:
                print(f"   ❌ OSRM API error: {data.get('message', 'Unknown')}")
                print(f"   Full response: {data}")
        else:
            print(f"   ❌ HTTP error: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
    except Exception as e:
        print(f"   ❌ Connection failed: {str(e)}")

    # Test 2: Route API (for geometry)
    print("\n2. Testing Route API (route geometry)...")
    try:
        url = f"{osrm_server}/route/v1/driving/{coords_str}"
        params = {'overview': 'full', 'geometries': 'geojson'}

        print(f"   URL: {url}")
        print(f"   Params: {params}")

        response = requests.get(url, params=params, timeout=30)
        print(f"   Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            if data.get('code') == 'Ok':
                print("   ✅ Route API working!")
                route = data['routes'][0]
                print(f"   Distance: {route['distance']} meters")
                print(f"   Duration: {route['duration']} seconds")
                geom_points = len(route['geometry']['coordinates'])
                print(f"   Geometry points: {geom_points}")
            else:
                print(f"   ❌ OSRM API error: {data.get('message', 'Unknown')}")
                print(f"   Full response: {data}")
        else:
            print(f"   ❌ HTTP error: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
    except Exception as e:
        print(f"   ❌ Connection failed: {str(e)}")

    print("\n" + "=" * 60)
    print("\nCommon issues and solutions:")
    print("1. Connection refused → OSRM server not running")
    print("2. Timeout → Check firewall/network")
    print("3. 'NoRoute' error → Coordinates not on road network")
    print("4. 'TooBig' error → Too many coordinates (max ~100)")
    print("\nIn Streamlit app:")
    print("- Make sure 'Use OSRM (real roads)' is CHECKED in sidebar")
    print("- Verify OSRM Server URL is correct")
    print("- Check the '⚠️ OSRM Issues Detected' section for errors")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_osrm_connection(sys.argv[1])
    else:
        test_osrm_connection()
