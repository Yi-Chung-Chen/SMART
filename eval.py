"""
Evaluation script for SMART model to generate Waymo Sim Agents Challenge submissions.

Usage:
    # Full test set submission
    python eval.py \
        --config configs/evaluation/eval_test.yaml \
        --pretrain_ckpt path/to/checkpoint.ckpt \
        --output_dir ./submission_output

    # Debug: Process first 10 scenarios only
    python eval.py \
        --config configs/evaluation/eval_val.yaml \
        --pretrain_ckpt path/to/checkpoint.ckpt \
        --output_dir ./debug_output \
        --num_scenarios 10

    # Debug: Process specific scenarios by ID
    python eval.py \
        --config configs/evaluation/eval_val.yaml \
        --pretrain_ckpt path/to/checkpoint.ckpt \
        --output_dir ./debug_output \
        --scenario_ids "scenario_id_1,scenario_id_2"
"""

import os
import tarfile
from argparse import ArgumentParser
from collections import defaultdict

import pytorch_lightning as pl
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from waymo_open_dataset.protos import sim_agents_submission_pb2

from smart.datasets.scalable_dataset import MultiDataset
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


class FilteredDataset(MultiDataset):
    """Dataset wrapper that filters scenarios based on provided criteria."""

    def __init__(self, base_dataset, num_scenarios=None, scenario_ids=None):
        """
        Args:
            base_dataset: The original MultiDataset instance
            num_scenarios: If provided, only use first N scenarios
            scenario_ids: If provided, only use scenarios with these IDs
        """
        # Copy attributes from base dataset
        self._raw_file_names = base_dataset._raw_file_names
        self._raw_paths = base_dataset._raw_paths
        self._raw_file_dataset = base_dataset._raw_file_dataset
        self.split = base_dataset.split
        self.training = base_dataset.training
        self.dim = base_dataset.dim
        self.num_historical_steps = base_dataset.num_historical_steps
        self.token_processor = base_dataset.token_processor
        self.transform = base_dataset.transform

        # Apply filtering
        if scenario_ids is not None:
            # Filter by specific scenario IDs
            scenario_id_set = set(scenario_ids)
            filtered_indices = []
            for idx in range(len(self._raw_paths)):
                # Extract scenario ID from filename (without .pkl extension)
                scenario_id = os.path.splitext(os.path.basename(self._raw_paths[idx]))[0]
                if scenario_id in scenario_id_set:
                    filtered_indices.append(idx)
            self._indices = filtered_indices
        elif num_scenarios is not None:
            # Filter by number of scenarios
            self._indices = list(range(min(num_scenarios, len(self._raw_paths))))
        else:
            # Use all scenarios
            self._indices = list(range(len(self._raw_paths)))

        self._num_samples = len(self._indices)

    def len(self) -> int:
        return self._num_samples

    def get(self, idx: int):
        real_idx = self._indices[idx]
        with open(self._raw_paths[real_idx], 'rb') as handle:
            import pickle
            data = pickle.load(handle)
        data = self.token_processor.preprocess(data)
        return data


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
        agent_ids = rollout_pred['agent_ids'][0]  # List of agent IDs
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
            unique_method_name='SMART',
            authors=[''],
            affiliation='',
            description='SMART: Scalable Multi-agent Real-time Motion Generation via Next-token Prediction',
            method_link=''
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


def run_rollouts(model, data, n_rollouts):
    """
    Run multiple rollouts for a single scenario.

    Args:
        model: SMART model
        data: HeteroData for one scenario
        n_rollouts: Number of rollouts to generate

    Returns:
        List of prediction dicts, one per rollout
    """
    all_predictions = []

    for _ in range(n_rollouts):
        # Run inference (stochastic sampling produces different results each time)
        pred = model.inference(data)

        all_predictions.append({
            'pred_traj': pred['pred_traj'].cpu(),
            'pred_head': pred['pred_head'].cpu(),
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
    parser.add_argument('--scenario_ids', type=str, default=None,
                        help='Comma-separated list of specific scenario IDs to process')
    parser.add_argument('--n_rollouts', type=int, default=N_ROLLOUTS,
                        help=f'Number of rollouts per scenario (default: {N_ROLLOUTS})')
    parser.add_argument('--n_shards', type=int, default=N_SHARDS,
                        help=f'Number of submission shards (default: {N_SHARDS})')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    args = parser.parse_args()

    # Parse scenario IDs if provided
    scenario_ids = None
    if args.scenario_ids:
        scenario_ids = [s.strip() for s in args.scenario_ids.split(',')]

    # Set seed for reproducibility
    pl.seed_everything(2, workers=True)

    # Load config
    config = load_config_act(args.config)
    logger = Logging().log(level='DEBUG')

    # Create dataset
    data_config = config.Dataset
    base_dataset = MultiDataset(
        root=data_config.root,
        split='val',  # Use 'val' split which maps to test for testing
        raw_dir=data_config.val_raw_dir,
        processed_dir=data_config.val_processed_dir,
        transform=WaymoTargetBuilder(
            config.Model.num_historical_steps,
            config.Model.decoder.num_future_steps,
            mode='val'
        )
    )

    # Apply filtering if needed
    if args.num_scenarios is not None or scenario_ids is not None:
        dataset = FilteredDataset(base_dataset, args.num_scenarios, scenario_ids)
        logger.info(f'Filtered dataset to {len(dataset)} scenarios')
    else:
        dataset = base_dataset

    logger.info(f'Dataset size: {len(dataset)} scenarios')

    # Create dataloader (batch_size=1 for scenarios)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=data_config.num_workers if hasattr(data_config, 'num_workers') else 4,
        pin_memory=True
    )

    # Load model
    model = SMART(config.Model)
    model.load_params_from_file(filename=args.pretrain_ckpt, logger=logger)
    model.eval()

    # Move to device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Process scenarios
    scenario_rollouts_list = []

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(dataloader, desc='Processing scenarios')):
            # Move data to device
            data = data.to(device)

            # Prepare data
            data = model.match_token_map(data)
            data = model.sample_pt_pred(data)

            # Get scenario ID
            scenario_id = data.get('scenario_id', f'scenario_{batch_idx}')
            if isinstance(scenario_id, list):
                scenario_id = scenario_id[0]

            # Run multiple rollouts
            all_rollout_predictions = run_rollouts(model, data, args.n_rollouts)

            # Create ScenarioRollouts proto
            scenario_rollouts = create_scenario_rollouts(scenario_id, all_rollout_predictions)
            scenario_rollouts_list.append(scenario_rollouts)

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
