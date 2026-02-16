"""
Update metadata in existing Waymo Sim Agents Challenge submission files.

This script updates the metadata fields in binproto shards and creates a new tar.gz.

Usage:
    # From a directory with binproto files (faster, no extraction needed)
    python update_submission_metadata.py \
        --input_dir debug_output \
        --output_tar debug_output/submission_updated.tar.gz

    # From an existing tar.gz file
    python update_submission_metadata.py \
        --input_tar debug_output/submission.tar.gz \
        --output_tar debug_output/submission_updated.tar.gz
"""

import argparse
import os
import tarfile
import tempfile
import shutil
from pathlib import Path

from waymo_open_dataset.protos import sim_agents_submission_pb2


def update_metadata(submission_proto):
    """Update metadata fields in a SimAgentsChallengeSubmission proto.

    Args:
        submission_proto: SimAgentsChallengeSubmission proto to update

    Returns:
        Updated proto with new metadata
    """
    # Update basic metadata
    submission_proto.unique_method_name = 'SMART_from_scratch'
    submission_proto.authors[:] = ['Yi-Chung Chen']
    submission_proto.affiliation = 'Purdue University'
    submission_proto.description = 'SMART: Scalable Multi-agent Real-time Motion Generation via Next-token Prediction'
    submission_proto.method_link = 'https://github.com/rainmaker22/SMART'

    # Add required metadata fields
    submission_proto.uses_lidar_data = False
    submission_proto.uses_camera_data = False
    submission_proto.uses_public_model_pretraining = False
    submission_proto.num_model_parameters = '7M'
    submission_proto.acknowledge_complies_with_closed_loop_requirement = True

    return submission_proto


def process_submission(input_path, output_tar_path, is_directory=False):
    """Process submission files and update metadata in all shards.

    Args:
        input_path: Path to input tar.gz or directory with binproto files
        output_tar_path: Path to output submission.tar.gz with updated metadata
        is_directory: If True, input_path is a directory; if False, it's a tar.gz
    """
    input_path = Path(input_path)
    output_tar_path = Path(output_tar_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    # Create output directory if needed
    output_tar_path.parent.mkdir(parents=True, exist_ok=True)

    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        updated_dir = temp_path / "updated"
        updated_dir.mkdir()

        # Get binproto files based on input type
        if is_directory:
            print(f"Reading binproto files from directory: {input_path}")
            binproto_files = sorted(input_path.glob("*.binproto*"))
            if not binproto_files:
                raise ValueError(f"No binproto files found in directory: {input_path}")
        else:
            print(f"Extracting {input_path}...")
            extract_dir = temp_path / "extracted"
            extract_dir.mkdir()
            # Extract the tar.gz
            with tarfile.open(input_path, 'r:gz') as tar:
                tar.extractall(extract_dir)
            binproto_files = sorted(extract_dir.glob("*.binproto*"))
            if not binproto_files:
                raise ValueError(f"No binproto files found in {input_path}")

        print(f"Found {len(binproto_files)} binproto shard(s)")

        # Process each binproto file
        for binproto_path in binproto_files:
            print(f"Processing {binproto_path.name}...")

            # Read the submission proto
            with open(binproto_path, 'rb') as f:
                submission = sim_agents_submission_pb2.SimAgentsChallengeSubmission.FromString(
                    f.read()
                )

            print(f"  - Found {len(submission.scenario_rollouts)} scenarios")

            # Update metadata
            submission = update_metadata(submission)

            # Write updated proto
            updated_path = updated_dir / binproto_path.name
            with open(updated_path, 'wb') as f:
                f.write(submission.SerializeToString())

            print(f"  - Updated metadata written")

        # Create new tar.gz with updated files
        print(f"\nCreating updated tar.gz at {output_tar_path}...")
        with tarfile.open(output_tar_path, 'w:gz') as tar:
            for updated_file in sorted(updated_dir.glob("*.binproto*")):
                tar.add(updated_file, arcname=updated_file.name)

        print(f"Done! Updated submission saved to {output_tar_path}")

        # Print summary of updated metadata
        print("\n" + "="*60)
        print("Updated metadata:")
        print("="*60)
        # Read one shard to show the metadata
        with open(next(updated_dir.glob("*.binproto*")), 'rb') as f:
            sample = sim_agents_submission_pb2.SimAgentsChallengeSubmission.FromString(
                f.read()
            )
        print(f"  unique_method_name: {sample.unique_method_name}")
        print(f"  authors: {list(sample.authors)}")
        print(f"  affiliation: {sample.affiliation}")
        print(f"  description: {sample.description}")
        print(f"  method_link: {sample.method_link}")
        print(f"  uses_lidar_data: {sample.uses_lidar_data}")
        print(f"  uses_camera_data: {sample.uses_camera_data}")
        print(f"  uses_public_model_pretraining: {sample.uses_public_model_pretraining}")
        print(f"  num_model_parameters: {sample.num_model_parameters}")
        print(f"  acknowledge_complies_with_closed_loop_requirement: {sample.acknowledge_complies_with_closed_loop_requirement}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Update metadata in existing Waymo submission files'
    )

    # Input options (one of these is required)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input_dir',
        type=str,
        help='Path to directory containing binproto files (faster, no extraction)'
    )
    input_group.add_argument(
        '--input_tar',
        type=str,
        help='Path to input submission.tar.gz'
    )

    parser.add_argument(
        '--output_tar',
        type=str,
        required=True,
        help='Path to output submission.tar.gz with updated metadata'
    )

    args = parser.parse_args()

    try:
        if args.input_dir:
            process_submission(args.input_dir, args.output_tar, is_directory=True)
        else:
            process_submission(args.input_tar, args.output_tar, is_directory=False)
    except Exception as e:
        print(f"\nError: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
