mkdir -p data/waymo_processed/training_test
find data/waymo_processed/training -maxdepth 1 -name "*.pkl" | head -50 | xargs -I {} cp {} data/waymo_processed/training_test/