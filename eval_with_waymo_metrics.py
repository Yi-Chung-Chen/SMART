"""
Advanced evaluation script with full Waymo Sim Agents Challenge metrics.

This script demonstrates how to compute Waymo metrics locally if you have access
to the original scenario files (not just preprocessed .pkl files).

Requirements:
  - waymo-open-dataset-tf-2-12-0==1.6.4
  - Original Waymo scenario files in TFRecord format
  - tensorflow (for reading TFRecords)
"""

import os
import pickle
from argparse import ArgumentParser
from typing import Dict, List
import tensorflow as tf
import torch
from tqdm import tqdm

try:
    from waymo_open_dataset.wdl_limited.sim_agents_metrics import metric_features
    from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics as waymo_metrics
    from waymo_open_dataset.protos import scenario_pb2
    WAYMO_AVAILABLE = True
except ImportError:
    WAYMO_AVAILABLE = False
    print("ERROR: Waymo Open Dataset not installed!")
    print("Install with: pip install waymo-open-dataset-tf-2-12-0==1.6.4")
    exit(1)


def load_scenario_from_file(scenario_file: str, scenario_id: str) -> scenario_pb2.Scenario:
    """
    Load a specific scenario from a Waymo TFRecord file.

    Args:
        scenario_file: Path to the .tfrecord file
        scenario_id: ID of the scenario to load

    Returns:
        The scenario protobuf object
    """
    dataset = tf.data.TFRecordDataset(scenario_file, compression_type='')

    for data in dataset:
        proto = scenario_pb2.Scenario()
        proto.ParseFromString(data.numpy())
        if proto.scenario_id == scenario_id:
            return proto

    raise ValueError(f"Scenario {scenario_id} not found in {scenario_file}")


def compute_waymo_metrics(
    predictions_file: str,
    scenario_dir: str,
    challenge_config: str = '2024'
) -> Dict:
    """
    Compute Waymo Sim Agents Challenge metrics for saved predictions.

    Args:
        predictions_file: Path to saved predictions (.pkl file from eval.py)
        scenario_dir: Directory containing original Waymo scenario files (.tfrecord)
        challenge_config: Challenge version ('2023' or '2024')

    Returns:
        Dictionary containing all metrics
    """
    # Load predictions
    print(f"Loading predictions from: {predictions_file}")
    with open(predictions_file, 'rb') as f:
        predictions = pickle.load(f)

    print(f"Loaded {len(predictions)} scenario predictions")

    # Load metrics configuration
    print(f"Loading metrics config for challenge: {challenge_config}")
    config = waymo_metrics.load_metrics_config(challenge_config)

    # Find scenario files
    scenario_files = [
        os.path.join(scenario_dir, f)
        for f in os.listdir(scenario_dir)
        if f.endswith('.tfrecord') or f.endswith('.tfrecord-00000-of-00001')
    ]

    print(f"Found {len(scenario_files)} scenario files in {scenario_dir}")

    # Compute metrics for each scenario
    all_metrics = []
    print("\nComputing metrics for each scenario...")

    for scenario_id, pred_data in tqdm(predictions.items(), desc="Evaluating"):
        try:
            # Load the original scenario
            scenario = None
            for scenario_file in scenario_files:
                try:
                    scenario = load_scenario_from_file(scenario_file, scenario_id)
                    break
                except ValueError:
                    continue

            if scenario is None:
                print(f"Warning: Scenario {scenario_id} not found in any file, skipping...")
                continue

            # Compute metrics for this scenario
            scenario_metrics = waymo_metrics.compute_scenario_metrics_for_bundle(
                config,
                scenario,
                [pred_data['joint_scene']]
            )

            all_metrics.append(scenario_metrics)

        except Exception as e:
            print(f"Error processing scenario {scenario_id}: {e}")
            continue

    if len(all_metrics) == 0:
        raise RuntimeError("No scenarios were successfully evaluated!")

    # Aggregate metrics across all scenarios
    print(f"\nAggregating metrics across {len(all_metrics)} scenarios...")
    aggregated = waymo_metrics.aggregate_metrics(all_metrics)

    return aggregated


def print_metrics(metrics: Dict):
    """Pretty print the metrics results."""
    print("\n" + "="*80)
    print("WAYMO SIM AGENTS CHALLENGE EVALUATION RESULTS")
    print("="*80)

    # Meta metric (primary ranking metric)
    if hasattr(metrics, 'metametric'):
        print(f"\nREALISM META METRIC: {metrics.metametric:.4f}")
        print("  (Primary ranking metric - higher is better)")

    # Kinematic metrics
    print(f"\nKINEMATIC METRICS: {metrics.kinematic_metrics:.4f}")
    if hasattr(metrics, 'linear_speed'):
        print(f"  Linear Speed:         {metrics.linear_speed:.4f}")
    if hasattr(metrics, 'linear_acceleration'):
        print(f"  Linear Acceleration:  {metrics.linear_acceleration:.4f}")
    if hasattr(metrics, 'angular_speed'):
        print(f"  Angular Speed:        {metrics.angular_speed:.4f}")
    if hasattr(metrics, 'angular_acceleration'):
        print(f"  Angular Acceleration: {metrics.angular_acceleration:.4f}")

    # Interactive metrics
    print(f"\nINTERACTIVE METRICS: {metrics.interactive_metrics:.4f}")
    if hasattr(metrics, 'distance_to_nearest_object'):
        print(f"  Distance to Nearest:  {metrics.distance_to_nearest_object:.4f}")
    if hasattr(metrics, 'collision_indication'):
        print(f"  Collision Rate:       {metrics.collision_indication:.4f}")
    if hasattr(metrics, 'time_to_collision'):
        print(f"  Time to Collision:    {metrics.time_to_collision:.4f}")

    # Map-based metrics
    print(f"\nMAP-BASED METRICS: {metrics.map_based_metrics:.4f}")
    if hasattr(metrics, 'distance_to_road_edge'):
        print(f"  Distance to Road Edge: {metrics.distance_to_road_edge:.4f}")
    if hasattr(metrics, 'offroad_indication'):
        print(f"  Off-road Rate:         {metrics.offroad_indication:.4f}")

    # minADE (tie-breaker metric)
    if hasattr(metrics, 'average_displacement_error'):
        print(f"\nminADE: {metrics.average_displacement_error:.4f}")
        print("  (Tie-breaker metric - lower is better)")

    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    parser = ArgumentParser(description='Evaluate SMART predictions with Waymo metrics')
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to predictions.pkl file (from eval.py --save_predictions)')
    parser.add_argument('--scenario_dir', type=str, required=True,
                        help='Directory containing Waymo scenario .tfrecord files')
    parser.add_argument('--challenge', type=str, default='2024',
                        choices=['2023', '2024'],
                        help='Challenge version to use for metrics')
    parser.add_argument('--output', type=str, default=None,
                        help='Optional: Save results to this file')

    args = parser.parse_args()

    if not WAYMO_AVAILABLE:
        print("ERROR: This script requires waymo-open-dataset")
        exit(1)

    # Verify inputs exist
    if not os.path.exists(args.predictions):
        print(f"ERROR: Predictions file not found: {args.predictions}")
        exit(1)

    if not os.path.isdir(args.scenario_dir):
        print(f"ERROR: Scenario directory not found: {args.scenario_dir}")
        exit(1)

    # Compute metrics
    try:
        metrics = compute_waymo_metrics(
            args.predictions,
            args.scenario_dir,
            args.challenge
        )

        # Print results
        print_metrics(metrics)

        # Save if requested
        if args.output:
            with open(args.output, 'wb') as f:
                pickle.dump(metrics, f)
            print(f"Results saved to: {args.output}")

    except Exception as e:
        print(f"\nERROR: Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
