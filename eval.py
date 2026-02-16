"""
Evaluation script for SMART model to generate Waymo Sim Agents Challenge submissions.

Usage:
    # Full validation set with validation enabled
    python eval.py \
        --config configs/evaluation/eval_val.yaml \
        --pretrain_ckpt path/to/checkpoint.ckpt \
        --tfrecord_dir data/waymo/scenario/validation \
        --processed_dir data/waymo_processed/validation \
        --output_dir ./submission_output \
        --validate

    # Debug: Process first 10 scenarios only
    python eval.py \
        --config configs/evaluation/eval_val.yaml \
        --pretrain_ckpt path/to/checkpoint.ckpt \
        --tfrecord_dir data/waymo/scenario/validation \
        --processed_dir data/waymo_processed/validation \
        --output_dir ./debug_output \
        --num_scenarios 10

    # Test set submission (no validation)
    python eval.py \
        --config configs/evaluation/eval_test.yaml \
        --pretrain_ckpt path/to/checkpoint.ckpt \
        --tfrecord_dir data/waymo/scenario/testing \
        --processed_dir data/waymo_processed/testing \
        --output_dir ./submission_output

    # Adjust rollout chunk size for memory management
    python eval.py \
        --config configs/evaluation/eval_val.yaml \
        --pretrain_ckpt path/to/checkpoint.ckpt \
        --tfrecord_dir data/waymo/scenario/validation \
        --processed_dir data/waymo_processed/validation \
        --output_dir ./submission_output \
        --rollout_chunk_size 4
"""

import glob
import os
import pickle
import tarfile
from argparse import ArgumentParser
from collections import defaultdict

import pytorch_lightning as pl
import tensorflow as tf
import torch
from torch_geometric.data import Batch, HeteroData
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2, sim_agents_submission_pb2
from waymo_open_dataset.utils.sim_agents import submission_specs
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics

from smart.datasets.preprocess import TokenProcessor
from smart.model import SMART
from smart.model.smart import joint_scene_from_states
from smart.transforms import WaymoTargetBuilder
from smart.utils.config import load_config_act
from smart.utils.log import Logging

# Waymo Sim Agents Challenge constants
N_ROLLOUTS = 32  # Total rollouts per scenario
N_SIMULATION_STEPS = 80  # Future steps to predict
CURRENT_TIME_INDEX = 10  # Step 11 (0-indexed), last observed step
N_SHARDS = 150  # Number of submission shards


def create_scenario_rollouts(scenario_id, all_rollout_predictions):
    """
    Create ScenarioRollouts proto from predictions.

    Args:
        scenario_id: String ID of the scenario
        all_rollout_predictions: List of dicts containing predictions for each rollout

    Returns:
        sim_agents_submission_pb2.ScenarioRollouts proto
    """
    joint_scenes = []

    for rollout_pred in all_rollout_predictions:
        pred_traj = rollout_pred['pred_traj']  # (n_agents, 80, 2)
        pred_head = rollout_pred['pred_head']  # (n_agents, 80)
        valid_mask = rollout_pred['valid_mask']  # (n_agents,)
        agent_ids = rollout_pred['agent_ids']  # List of agent IDs
        agent_z = rollout_pred['agent_z']  # (n_agents,)

        # Filter to valid agents
        valid_indices = valid_mask.nonzero(as_tuple=True)[0]

        if len(valid_indices) == 0:
            # Create empty joint scene if no valid agents
            joint_scenes.append(sim_agents_submission_pb2.JointScene(simulated_trajectories=[]))
            continue

        # Build states tensor: (n_valid_agents, 80, 4) with [x, y, z, heading]
        n_valid = len(valid_indices)
        states = torch.zeros(n_valid, N_SIMULATION_STEPS, 4)
        valid_agent_ids = []

        for i, idx in enumerate(valid_indices):
            idx = idx.item()
            states[i, :, 0] = pred_traj[idx, :, 0]  # x
            states[i, :, 1] = pred_traj[idx, :, 1]  # y
            states[i, :, 2] = agent_z[idx]  # z (constant)
            states[i, :, 3] = pred_head[idx, :]  # heading
            valid_agent_ids.append(agent_ids[idx])

        # Convert agent IDs to tensor
        object_ids = torch.tensor(valid_agent_ids, dtype=torch.int64)

        # Create JointScene using existing helper
        joint_scene = joint_scene_from_states(states, object_ids)
        joint_scenes.append(joint_scene)

    return sim_agents_submission_pb2.ScenarioRollouts(
        scenario_id=scenario_id,
        joint_scenes=joint_scenes
    )


def create_submission_shards(scenario_rollouts_list, output_dir, n_shards=N_SHARDS):
    """
    Create sharded submission binproto files.

    Args:
        scenario_rollouts_list: List of ScenarioRollouts protos
        output_dir: Directory to write shard files
        n_shards: Number of shards to create
    """
    os.makedirs(output_dir, exist_ok=True)

    # Distribute scenarios across shards
    shards = defaultdict(list)
    for i, scenario_rollouts in enumerate(scenario_rollouts_list):
        shard_idx = i % n_shards
        shards[shard_idx].append(scenario_rollouts)

    # Write each shard
    shard_files = []
    for shard_idx in range(n_shards):
        shard_filename = f'submission.binproto-{shard_idx:05d}-of-{n_shards:05d}'
        shard_path = os.path.join(output_dir, shard_filename)

        # Create submission proto for this shard
        submission = sim_agents_submission_pb2.SimAgentsChallengeSubmission(
            scenario_rollouts=shards[shard_idx],
            submission_type=sim_agents_submission_pb2.SimAgentsChallengeSubmission.SIM_AGENTS_SUBMISSION,
            account_name='',  # Will be filled by submission system
            unique_method_name='SMART_from_scratch',
            authors=['Yi-Chung Chen'],
            affiliation='Purdue University',
            description='SMART: Scalable Multi-agent Real-time Motion Generation via Next-token Prediction',
            method_link='https://github.com/rainmaker22/SMART',
            # Required metadata fields
            uses_lidar_data=False,
            uses_camera_data=False,
            uses_public_model_pretraining=False,
            num_model_parameters='7M',
            acknowledge_complies_with_closed_loop_requirement=True,
        )

        with open(shard_path, 'wb') as f:
            f.write(submission.SerializeToString())
        shard_files.append(shard_filename)
        print(f'Written shard {shard_idx + 1}/{n_shards}: {shard_filename}')

    return shard_files


def create_tar_archive(output_dir, shard_files):
    """Create tar.gz archive of submission shards."""
    tar_path = os.path.join(output_dir, 'submission.tar.gz')
    with tarfile.open(tar_path, 'w:gz') as tar:
        for shard_file in shard_files:
            shard_path = os.path.join(output_dir, shard_file)
            tar.add(shard_path, arcname=shard_file)
    print(f'Created submission archive: {tar_path}')
    return tar_path


def iterate_tfrecord_scenarios(tfrecord_dir: str, num_scenarios: int = None):
    """Iterate through TFRecord files, yielding (scenario, scenario_id) pairs.

    Args:
        tfrecord_dir: Directory containing TFRecord files
        num_scenarios: Optional limit on number of scenarios to process

    Yields:
        Tuple of (scenario_pb2.Scenario, scenario_id string)
    """
    tfrecord_files = sorted(glob.glob(os.path.join(tfrecord_dir, "*.tfrecord*")))
    count = 0

    for filepath in tfrecord_files:
        dataset = tf.data.TFRecordDataset([filepath])
        for data in dataset:
            if num_scenarios and count >= num_scenarios:
                return
            scenario = scenario_pb2.Scenario.FromString(data.numpy())
            yield scenario, scenario.scenario_id
            count += 1


def load_pkl_by_scenario_id(scenario_id: str, processed_dir: str,
                            token_processor, transform) -> HeteroData:
    """Load preprocessed pkl file by scenario_id.

    Args:
        scenario_id: The scenario ID to load
        processed_dir: Directory containing processed pkl files
        token_processor: TokenProcessor instance for token preprocessing
        transform: WaymoTargetBuilder transform to apply

    Returns:
        Transformed HeteroData, or None if not found
    """
    pkl_path = os.path.join(processed_dir, f"{scenario_id}.pkl")
    if not os.path.exists(pkl_path):
        return None

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Apply token preprocessing (same as MultiDataset.get())
    data = token_processor.preprocess(data)

    # Apply transform (WaymoTargetBuilder)
    if transform:
        data = transform(data)

    return data


def replicate_for_rollouts(data: HeteroData, n_rollouts: int) -> Batch:
    """Replicate single scenario data for batched inference."""
    data_list = [data.clone() for _ in range(n_rollouts)]
    return Batch.from_data_list(data_list)


def run_rollouts_batched(model, data, n_rollouts, chunk_size=8):
    """Run batched rollouts with memory-aware chunking.

    Args:
        model: SMART model
        data: HeteroData for one scenario
        n_rollouts: Total number of rollouts to generate
        chunk_size: Number of rollouts to batch together

    Returns:
        List of prediction dicts, one per rollout
    """
    all_predictions = []
    num_agents = data['agent']['num_nodes']

    for chunk_start in range(0, n_rollouts, chunk_size):
        chunk_n = min(chunk_size, n_rollouts - chunk_start)
        batched_data = replicate_for_rollouts(data, chunk_n)

        pred = model.inference(batched_data)

        # Extract predictions per rollout
        for i in range(chunk_n):
            start_idx = i * num_agents
            end_idx = (i + 1) * num_agents
            all_predictions.append({
                'pred_traj': pred['pred_traj'][start_idx:end_idx].cpu(),
                'pred_head': pred['pred_head'][start_idx:end_idx].cpu(),
                'valid_mask': data['agent']['valid_mask'][:, model.num_historical_steps - 1].cpu(),
                'agent_ids': data['agent']['id'],
                'agent_z': data['agent']['position'][:, model.num_historical_steps - 1, 2].cpu(),
            })

    return all_predictions


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--pretrain_ckpt', type=str, required=True,
                        help='Path to pretrained checkpoint')
    parser.add_argument('--output_dir', type=str, default='./submission',
                        help='Output directory for submission files')
    parser.add_argument('--num_scenarios', type=int, default=None,
                        help='Process only first N scenarios (for debugging)')
    parser.add_argument('--n_rollouts', type=int, default=N_ROLLOUTS,
                        help=f'Number of rollouts per scenario (default: {N_ROLLOUTS})')
    parser.add_argument('--n_shards', type=int, default=N_SHARDS,
                        help=f'Number of submission shards (default: {N_SHARDS})')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--tfrecord_dir', type=str, required=True,
                        help='Directory with raw TFRecord files')
    parser.add_argument('--processed_dir', type=str, required=True,
                        help='Directory with processed pkl files')
    parser.add_argument('--validate', action='store_true',
                        help='Enable submission_specs validation')
    parser.add_argument('--rollout_chunk_size', type=int, default=8,
                        help='Batch size for parallel rollouts (GPU memory management)')
    args = parser.parse_args()

    # Set seed for reproducibility
    pl.seed_everything(2, workers=True)

    # Load config
    config = load_config_act(args.config)
    logger = Logging().log(level='DEBUG')

    # Create token processor and transform (same as MultiDataset uses)
    token_processor = TokenProcessor(2048)
    transform = WaymoTargetBuilder(
        config.Model.num_historical_steps,
        config.Model.decoder.num_future_steps,
        mode='val'
    )

    # Load model
    model = SMART(config.Model)
    model.load_params_from_file(filename=args.pretrain_ckpt, logger=logger)
    model.eval()
    model.noise = False  # Deterministic map encoding for inference

    # Move to device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Process scenarios using TFRecord-first approach
    scenario_rollouts_list = []
    skipped_count = 0

    with torch.no_grad():
        for scenario, scenario_id in tqdm(
            iterate_tfrecord_scenarios(args.tfrecord_dir, args.num_scenarios),
            desc='Processing scenarios'
        ):
            # Load corresponding pkl file
            data = load_pkl_by_scenario_id(
                scenario_id, args.processed_dir, token_processor, transform
            )
            if data is None:
                logger.warning(f'Pkl not found for {scenario_id}, skipping')
                skipped_count += 1
                continue

            # Move to device and prepare
            data = data.to(device)
            data = model.match_token_map(data)
            data = model.sample_pt_pred(data)  # Creates pt_valid_mask needed by map decoder

            # Run batched rollouts
            all_rollout_predictions = run_rollouts_batched(
                model, data, args.n_rollouts, args.rollout_chunk_size
            )

            # Create ScenarioRollouts proto
            scenario_rollouts = create_scenario_rollouts(scenario_id, all_rollout_predictions)

            # Validate if enabled
            if args.validate:
                submission_specs.validate_scenario_rollouts(scenario_rollouts, scenario)

                config = metrics.load_metrics_config()
                scenario_metrics = metrics.compute_scenario_metrics_for_bundle(
                    config, scenario, scenario_rollouts
                )
                print(scenario_metrics)

            scenario_rollouts_list.append(scenario_rollouts)

    if skipped_count > 0:
        logger.warning(f'Skipped {skipped_count} scenarios (pkl files not found)')

    logger.info(f'Processed {len(scenario_rollouts_list)} scenarios')

    # Adjust number of shards based on number of scenarios
    n_shards = min(args.n_shards, len(scenario_rollouts_list))
    if n_shards < args.n_shards:
        logger.info(f'Adjusted number of shards from {args.n_shards} to {n_shards} (fewer scenarios than shards)')

    # Create submission shards
    logger.info(f'Creating {n_shards} submission shards...')
    shard_files = create_submission_shards(scenario_rollouts_list, args.output_dir, n_shards)

    # Create tar archive
    tar_path = create_tar_archive(args.output_dir, shard_files)

    logger.info(f'Submission complete!')
    logger.info(f'  - Output directory: {args.output_dir}')
    logger.info(f'  - Number of scenarios: {len(scenario_rollouts_list)}')
    logger.info(f'  - Number of rollouts per scenario: {args.n_rollouts}')
    logger.info(f'  - Number of shards: {n_shards}')
    logger.info(f'  - Archive: {tar_path}')


if __name__ == '__main__':
    main()
