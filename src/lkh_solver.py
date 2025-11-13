"""
LKH (Lin-Kernighan-Helsgaun) TSP Solver Wrapper

This module provides an interface to the LKH TSP solver, which is one of the
best heuristic solvers for the Traveling Salesman Problem.

LKH must be installed separately. Download from:
http://webhotel4.ruc.dk/~keld/research/LKH/
"""

import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional
import math


class LKHSolver:
    """Wrapper for LKH TSP solver."""

    def __init__(self, lkh_path: str = "LKH"):
        """
        Initialize LKH solver.

        Args:
            lkh_path: Path to LKH executable (default: "LKH" if in PATH)
        """
        self.lkh_path = lkh_path
        self._check_lkh_installed()

    def _check_lkh_installed(self) -> bool:
        """Check if LKH is installed and accessible."""
        try:
            result = subprocess.run(
                [self.lkh_path],
                capture_output=True,
                timeout=2
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print(f"⚠️  LKH not found at '{self.lkh_path}'")
            print("   Download from: http://webhotel4.ruc.dk/~keld/research/LKH/")
            return False

    def solve_tsp(self, distance_matrix: List[List[int]],
                  time_limit: int = 30,
                  runs: int = 1) -> Optional[Tuple[List[int], float]]:
        """
        Solve TSP using LKH.

        Args:
            distance_matrix: Square matrix of distances (integers in meters)
            time_limit: Time limit in seconds
            runs: Number of independent runs (LKH will pick best)

        Returns:
            Tuple of (route_indices, total_distance) or None if failed
        """
        num_cities = len(distance_matrix)

        # Create temporary directory for LKH files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Write TSPLIB problem file
            tsp_file = tmpdir / "problem.tsp"
            self._write_tsplib_file(tsp_file, distance_matrix)

            # Write LKH parameter file
            par_file = tmpdir / "problem.par"
            tour_file = tmpdir / "problem.tour"
            self._write_parameter_file(par_file, tsp_file, tour_file, time_limit, runs)

            # Run LKH
            try:
                result = subprocess.run(
                    [self.lkh_path, str(par_file)],
                    capture_output=True,
                    text=True,
                    timeout=time_limit + 10  # Extra buffer
                )

                if result.returncode != 0:
                    print(f"LKH error: {result.stderr}")
                    return None

                # Parse tour file
                tour = self._read_tour_file(tour_file, num_cities)
                if tour is None:
                    return None

                # Calculate tour length
                total_distance = self._calculate_tour_length(tour, distance_matrix)

                return tour, total_distance

            except subprocess.TimeoutExpired:
                print(f"LKH timeout after {time_limit}s")
                return None
            except Exception as e:
                print(f"LKH error: {str(e)}")
                return None

    def _write_tsplib_file(self, filepath: Path, distance_matrix: List[List[int]]) -> None:
        """Write distance matrix in TSPLIB format."""
        num_cities = len(distance_matrix)

        with open(filepath, 'w') as f:
            # Header
            f.write(f"NAME: TSP_Problem\n")
            f.write(f"TYPE: TSP\n")
            f.write(f"DIMENSION: {num_cities}\n")
            f.write(f"EDGE_WEIGHT_TYPE: EXPLICIT\n")
            f.write(f"EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
            f.write(f"EDGE_WEIGHT_SECTION\n")

            # Distance matrix
            for row in distance_matrix:
                f.write(" ".join(str(d) for d in row) + "\n")

            f.write("EOF\n")

    def _write_parameter_file(self, par_file: Path, tsp_file: Path,
                              tour_file: Path, time_limit: int, runs: int) -> None:
        """Write LKH parameter file."""
        with open(par_file, 'w') as f:
            f.write(f"PROBLEM_FILE = {tsp_file}\n")
            f.write(f"OUTPUT_TOUR_FILE = {tour_file}\n")
            f.write(f"RUNS = {runs}\n")
            f.write(f"TIME_LIMIT = {time_limit}\n")
            # Use good default parameters for delivery routing
            f.write(f"MOVE_TYPE = 5\n")  # Lin-Kernighan with 5-opt
            f.write(f"PATCHING_C = 3\n")
            f.write(f"PATCHING_A = 2\n")
            f.write(f"MAX_TRIALS = 1000\n")

    def _read_tour_file(self, tour_file: Path, num_cities: int) -> Optional[List[int]]:
        """Parse LKH tour output file."""
        if not tour_file.exists():
            return None

        try:
            with open(tour_file, 'r') as f:
                lines = f.readlines()

            # Find TOUR_SECTION
            tour_start = None
            for i, line in enumerate(lines):
                if line.strip() == "TOUR_SECTION":
                    tour_start = i + 1
                    break

            if tour_start is None:
                return None

            # Read tour
            tour = []
            for i in range(tour_start, len(lines)):
                line = lines[i].strip()
                if line == "-1" or line == "EOF":
                    break
                try:
                    city = int(line) - 1  # Convert to 0-indexed
                    tour.append(city)
                except ValueError:
                    continue

            # LKH doesn't repeat depot at end, add it
            if len(tour) == num_cities and tour[0] == 0:
                tour.append(0)

            return tour

        except Exception as e:
            print(f"Error reading tour file: {str(e)}")
            return None

    def _calculate_tour_length(self, tour: List[int],
                               distance_matrix: List[List[int]]) -> float:
        """Calculate total length of tour."""
        total = 0
        for i in range(len(tour) - 1):
            total += distance_matrix[tour[i]][tour[i+1]]
        return total


def test_lkh():
    """Test LKH solver with a simple example."""
    # Simple 5-city example
    distance_matrix = [
        [0, 10, 15, 20, 25],
        [10, 0, 35, 25, 30],
        [15, 35, 0, 30, 20],
        [20, 25, 30, 0, 15],
        [25, 30, 20, 15, 0]
    ]

    solver = LKHSolver()
    result = solver.solve_tsp(distance_matrix, time_limit=5)

    if result:
        tour, distance = result
        print(f"✓ LKH Tour: {tour}")
        print(f"✓ Distance: {distance}")
    else:
        print("✗ LKH failed to solve")


if __name__ == "__main__":
    test_lkh()
