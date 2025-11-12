"""
Configuration Management for Route Optimization System

Supports different environments:
- development: Fast settings for local development and testing
- staging: Production-like settings for final testing before deployment
- production: Optimized settings for live deployment
"""

import os
from typing import Dict, Any


class Config:
    """Base configuration"""

    # Default settings
    CLUSTERING_METHOD = 'hierarchical'
    TARGET_PODS_PER_COURIER = 4
    MAX_PACKAGES_PER_POD = 40
    MIN_PACKAGES_PER_POD = 10
    TIME_LIMIT = 30

    # Route optimization enhancements (NEW)
    USE_ENSEMBLE = False
    ROAD_DISTANCE_FACTOR = 1.35  # Roads are typically 35% longer than straight-line
    USE_OSRM = False
    OSRM_SERVER = 'http://router.project-osrm.org'


class DevelopmentConfig(Config):
    """
    Development configuration for fast iteration and testing.

    Use this for:
    - Local development
    - Quick testing with small datasets
    - Debugging
    """
    # Faster settings for development
    TIME_LIMIT = 10  # Quick optimization
    USE_ENSEMBLE = False  # Single solver for speed
    ROAD_DISTANCE_FACTOR = 1.35  # Use default correction factor
    USE_OSRM = False  # No external API calls

    # For easy identification
    ENV = 'development'


class StagingConfig(Config):
    """
    Staging configuration for pre-production testing.

    Use this for:
    - Testing new features before production
    - Validating improvements with real data
    - A/B testing different configurations
    """
    # More thorough optimization
    TIME_LIMIT = 30
    USE_ENSEMBLE = True  # Test ensemble solving
    ROAD_DISTANCE_FACTOR = 1.35  # Can be calibrated for your region
    USE_OSRM = False  # Could enable for testing real road distances

    # Optional: Use custom OSRM server if self-hosted
    # OSRM_SERVER = 'http://your-staging-osrm-server:5000'

    ENV = 'staging'


class ProductionConfig(Config):
    """
    Production configuration for optimal results.

    Use this for:
    - Live deployments
    - Customer-facing results
    - Maximum optimization quality
    """
    # Best quality optimization
    TIME_LIMIT = 30
    USE_ENSEMBLE = True  # Use ensemble for best results
    ROAD_DISTANCE_FACTOR = 1.35  # Calibrate based on your actual road network
    USE_OSRM = False  # Set to True if you have self-hosted OSRM server

    # Production OSRM server (self-hosted recommended for production)
    # OSRM_SERVER = 'http://your-production-osrm-server:5000'

    ENV = 'production'


# Environment mapping
CONFIG_MAP = {
    'development': DevelopmentConfig,
    'staging': StagingConfig,
    'production': ProductionConfig
}


def get_config(env: str = None) -> Config:
    """
    Get configuration for specified environment.

    Args:
        env: Environment name ('development', 'staging', 'production')
             If None, reads from ROUTE_OPTIMIZER_ENV environment variable
             Defaults to 'development' if not set

    Returns:
        Config instance for the specified environment

    Examples:
        # Use development config
        config = get_config('development')

        # Use staging config
        config = get_config('staging')

        # Use environment variable
        export ROUTE_OPTIMIZER_ENV=production
        config = get_config()  # Uses production config
    """
    if env is None:
        env = os.getenv('ROUTE_OPTIMIZER_ENV', 'development')

    config_class = CONFIG_MAP.get(env.lower(), DevelopmentConfig)
    return config_class()


def get_config_dict(env: str = None) -> Dict[str, Any]:
    """
    Get configuration as dictionary for easy parameter passing.

    Args:
        env: Environment name

    Returns:
        Dictionary of configuration parameters

    Example:
        config = get_config_dict('staging')
        route_optimizer = RouteOptimizer(
            clustering_system,
            use_ensemble=config['USE_ENSEMBLE'],
            road_distance_factor=config['ROAD_DISTANCE_FACTOR'],
            use_osrm=config['USE_OSRM'],
            osrm_server=config['OSRM_SERVER']
        )
    """
    config = get_config(env)
    return {
        'clustering_method': config.CLUSTERING_METHOD,
        'target_pods_per_courier': config.TARGET_PODS_PER_COURIER,
        'max_packages_per_pod': config.MAX_PACKAGES_PER_POD,
        'min_packages_per_pod': config.MIN_PACKAGES_PER_POD,
        'time_limit': config.TIME_LIMIT,
        'use_ensemble': config.USE_ENSEMBLE,
        'road_distance_factor': config.ROAD_DISTANCE_FACTOR,
        'use_osrm': config.USE_OSRM,
        'osrm_server': config.OSRM_SERVER,
        'env': config.ENV
    }


if __name__ == "__main__":
    """Test configuration system"""
    print("="*80)
    print("ROUTE OPTIMIZATION CONFIGURATION SYSTEM")
    print("="*80)

    for env_name in ['development', 'staging', 'production']:
        print(f"\n{env_name.upper()} Configuration:")
        print("-" * 40)
        config = get_config(env_name)
        config_dict = get_config_dict(env_name)

        for key, value in config_dict.items():
            print(f"  {key}: {value}")

    print("\n" + "="*80)
    print("Usage Examples:")
    print("="*80)
    print("\n1. Command line with environment:")
    print("   export ROUTE_OPTIMIZER_ENV=staging")
    print("   python main.py --input data.csv --city Jakarta")
    print("\n2. Command line with explicit flags:")
    print("   python main.py --input data.csv --city Jakarta --use-ensemble --road-factor 1.4")
    print("\n3. In Python code:")
    print("   from config import get_config_dict")
    print("   config = get_config_dict('production')")
    print("   optimizer = RouteOptimizer(clustering_system, **config)")
