"""
Local evaluation script for SMART model using Waymo Sim Agents Challenge metrics.
This script generates predictions on the test set and computes metrics locally using Waymo's API.
"""

from argparse import ArgumentParser
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from smart.datasets.scalable_dataset import MultiDataset
from smart.model import SMART
from smart.transforms import WaymoTargetBuilder
from smart.utils.config import load_config_act
from smart.utils.log import Logging

if __name__ == '__main__':
    pl.seed_everything(2, workers=True)
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/validation/validation_scalable.yaml",
                        help='Path to the evaluation config file')
    parser.add_argument('--pretrain_ckpt', type=str, required=True,
                        help='Path to the pretrained checkpoint')
    parser.add_argument('--output_dir', type=str, default='./eval_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save predictions in Waymo submission format')
    args = parser.parse_args()

    config = load_config_act(args.config)

    # Load test dataset
    data_config = config.Dataset
    test_dataset = {
        "scalable": MultiDataset,
    }[data_config.dataset](
        root=data_config.root,
        split='test',
        raw_dir=data_config.get('test_raw_dir', data_config.val_raw_dir),
        processed_dir=data_config.get('test_processed_dir', data_config.val_processed_dir),
        transform=WaymoTargetBuilder(
            config.Model.num_historical_steps,
            config.Model.decoder.num_future_steps
        )
    )

    dataloader = DataLoader(
        test_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        persistent_workers=True if data_config.num_workers > 0 else False
    )

    # Load model
    logger = Logging().log(level='INFO')
    logger.info(f'Loading checkpoint from: {args.pretrain_ckpt}')

    Predictor = SMART
    model = Predictor(config.Model)
    model.load_params_from_file(filename=args.pretrain_ckpt, logger=logger)

    # Enable inference mode and set output directory
    model.inference_token = True
    model.save_predictions_to_file = args.save_predictions
    model.output_dir = args.output_dir

    # Create trainer for testing
    trainer_config = config.Trainer
    trainer = pl.Trainer(
        accelerator=trainer_config.accelerator,
        devices=1,  # Use single GPU for evaluation
        strategy='auto',  # No DDP needed for evaluation
        num_sanity_val_steps=0
    )

    logger.info('Starting evaluation...')
    logger.info(f'Test dataset size: {len(test_dataset)}')

    # Run testing
    results = trainer.test(model, dataloader)

    logger.info('Evaluation complete!')
    logger.info(f'Results: {results}')

    if args.save_predictions:
        logger.info(f'Predictions saved to: {args.output_dir}')
