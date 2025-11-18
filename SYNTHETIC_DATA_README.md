# Synthetic Delivery Data Generator

Generate realistic test data for route optimization system.

## Quick Start

```bash
# Generate 500 deliveries for Jakarta
python generate_synthetic_data.py --city jakarta --deliveries 500

# Generate 1000 deliveries for Grobogan with 15 couriers
python generate_synthetic_data.py --city grobogan --deliveries 1000 --couriers 15

# Show all options
python generate_synthetic_data.py --help
```

## Available Cities

- **jakarta** - Jakarta (capital)
- **semarang** - Semarang (Central Java)
- **surabaya** - Surabaya (East Java)
- **bandung** - Bandung (West Java)
- **yogyakarta** - Yogyakarta
- **grobogan** - Grobogan (your current location)

## Parameters

| Parameter | Short | Default | Description |
|-----------|-------|---------|-------------|
| `--city` | `-c` | jakarta | City to generate data for |
| `--deliveries` | `-d` | 500 | Number of deliveries |
| `--couriers` | `-n` | 10 | Number of couriers |
| `--date` | | today | Delivery date (YYYY-MM-DD) |
| `--output` | `-o` | auto | Output CSV filename |

## Examples

### Small Test Dataset (Fast)
```bash
python generate_synthetic_data.py --city jakarta --deliveries 100 --couriers 5
# Output: synthetic_jakarta_100del_20250113_143022.csv
```

### Medium Dataset (Realistic)
```bash
python generate_synthetic_data.py --city grobogan --deliveries 500 --couriers 10
```

### Large Dataset (Performance Testing)
```bash
python generate_synthetic_data.py --city semarang --deliveries 2000 --couriers 20
```

### Custom Filename
```bash
python generate_synthetic_data.py \
  --city bandung \
  --deliveries 750 \
  --couriers 12 \
  --date 2025-01-15 \
  --output bandung_test_data.csv
```

## Generated Data Format

The CSV includes all required columns:

### Essential Columns
- `AWB_NUMBER` - Unique delivery ID (AWB20250113XXXXX)
- `EMPLOYEE_ID` - Courier ID (EMP001, EMP002, ...)
- `NICKNAME` - Courier name (Indonesian names)
- `DO_POD_DELIVER_CODE` - Current POD assignment (POD001, POD002, ...)
- `BERATASLI` - Package weight in kg (0.5 - 25.0 kg, realistic distribution)
- `SELECTED_LATITUDE` - Delivery latitude
- `SELECTED_LONGITUDE` - Delivery longitude

### Branch Information
- `BRANCH_LATITUDE` - Branch/depot latitude
- `BRANCH_LONGITUDE` - Branch/depot longitude
- `GERAI` / `BRANCH_NAME` - Branch name
- `KODE_GERAI` / `BRANCH_CODE` - Branch code
- `DO_POD_DELIVER_DATE` / `DELIVERY_DATE` - Delivery date

## Realistic Features

✅ **Geographic Clustering** - Deliveries are clustered in hotspots (like real cities)
✅ **Realistic Weights** - Log-normal distribution (most packages 1-5kg, some heavier)
✅ **Random Assignment** - Couriers randomly assigned to deliveries
✅ **POD Diversity** - More PODs than couriers (simulates unoptimized state)
✅ **Indonesian Names** - Realistic courier names for the region

## Use Cases

### 1. Quick Testing
```bash
# Small dataset, runs fast
python generate_synthetic_data.py --city jakarta --deliveries 50
```

### 2. Algorithm Comparison
```bash
# Generate same dataset, test OR-Tools vs LKH
python generate_synthetic_data.py --city grobogan --deliveries 500 --output test_dataset.csv

# Then in Streamlit:
# - Upload test_dataset.csv
# - Run with solver='ortools', note distance
# - Run with solver='lkh', compare distance
```

### 3. Performance Benchmarking
```bash
# Large dataset to test optimization speed
python generate_synthetic_data.py --city surabaya --deliveries 2000 --couriers 25
```

### 4. OSRM Testing
```bash
# Generate data in region covered by your OSRM server
python generate_synthetic_data.py --city grobogan --deliveries 300
```

## Tips

💡 **Start Small** - Test with 100-200 deliveries first
💡 **Match Your OSRM** - Use cities your OSRM server has map data for
💡 **Courier Count** - Use ~1 courier per 30-50 deliveries for realism
💡 **Time Limits** - Larger datasets need higher time limits in Streamlit

## Sample Output

```
Generating 500 deliveries for Grobogan...
✅ Generated 500 deliveries
📊 Statistics:
   - Couriers: 10
   - Current PODs: 28
   - Total Weight: 1847.3 kg
   - Avg Weight: 3.69 kg
   - Date: 2025-01-13
📁 Saved to: synthetic_grobogan_500del_20250113_143522.csv
```

## Adding New Cities

Edit `generate_synthetic_data.py` and add to the `CITIES` dict:

```python
CITIES = {
    'mycity': {
        'name': 'My City',
        'branch_lat': -7.XXX,    # Branch latitude
        'branch_lon': 110.XXX,    # Branch longitude
        'lat_range': (-7.20, -7.00),  # City bounds
        'lon_range': (110.30, 110.50),
        'branch_code': 'MYC01'
    },
    # ...
}
```

Then run:
```bash
python generate_synthetic_data.py --city mycity --deliveries 500
```
