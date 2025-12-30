"""
Script to verify if the preprocessed data includes all agents required by Waymo Challenge.

This checks:
1. Does category 3 (tracks_to_predict) include the AV/ego vehicle?
2. Does category 3 include all agents valid at last history step?
3. What agents are being predicted vs. what Waymo requires?
"""

import pickle
import sys
from pathlib import Path

def check_scenario(pkl_file):
    """Check a single scenario for Waymo compliance."""
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    scenario_id = data.get('scenario_id', 'unknown')

    # Get agent information
    num_agents = len(data['agent']['id'])
    categories = data['agent']['category']
    valid_mask = data['agent']['valid_mask']
    predict_mask = data['agent']['predict_mask'] if 'predict_mask' in data['agent'] else None
    av_index = data['agent']['av_index']

    # Last history step is index 10 (0-10 = 11 steps)
    last_history_step = 10

    # Agents valid at last history step
    valid_at_last = valid_mask[:, last_history_step]

    # Count by category
    cat_0 = (categories == 0).sum()
    cat_1 = (categories == 1).sum()
    cat_2 = (categories == 2).sum()  # AV/ego
    cat_3 = (categories == 3).sum()  # tracks_to_predict

    # Check if AV is in category 3
    av_is_cat3 = categories[av_index] == 3
    av_is_cat2 = categories[av_index] == 2

    # Check valid agents at last step
    valid_agents = valid_at_last.sum()
    valid_in_cat3 = (valid_at_last & (categories == 3)).sum()
    valid_not_in_cat3 = (valid_at_last & (categories != 3)).sum()

    print(f"\n{'='*80}")
    print(f"Scenario: {scenario_id}")
    print(f"{'='*80}")
    print(f"\nAgent Breakdown:")
    print(f"  Total agents: {num_agents}")
    print(f"  Category 0 (background): {cat_0}")
    print(f"  Category 1 (unscored): {cat_1}")
    print(f"  Category 2 (AV/ego): {cat_2}")
    print(f"  Category 3 (tracks_to_predict): {cat_3}")

    print(f"\nAV/Ego Vehicle:")
    print(f"  AV index: {av_index}")
    print(f"  AV ID: {data['agent']['id'][av_index]}")
    print(f"  AV category: {categories[av_index]}")
    print(f"  AV is in category 2: {av_is_cat2}")
    print(f"  AV is in category 3: {av_is_cat3} {'✓ (will be predicted)' if av_is_cat3 else '✗ (NOT predicted!)'}")

    print(f"\nValid Agents at Last History Step (t={last_history_step}):")
    print(f"  Total valid: {valid_agents}")
    print(f"  Valid in category 3: {valid_in_cat3}")
    print(f"  Valid NOT in category 3: {valid_not_in_cat3}")

    if predict_mask is not None:
        should_predict = predict_mask[:, last_history_step + 1].sum()
        print(f"  predict_mask says should predict: {should_predict}")

    print(f"\nWaymo Compliance Check:")
    if av_is_cat3:
        print(f"  ✓ AV is included in predictions (category 3)")
    else:
        print(f"  ✗ WARNING: AV is NOT in category 3 (may not be predicted!)")

    if valid_not_in_cat3 > 0:
        print(f"  ⚠ Warning: {valid_not_in_cat3} valid agents are NOT in category 3")
        print(f"    These agents are valid but won't be predicted by the model!")
    else:
        print(f"  ✓ All valid agents are in category 3")

    # Show some examples
    print(f"\nExample Agents:")
    print(f"  {'ID':<12} {'Category':<10} {'Valid@t=10':<12} {'Will Predict?'}")
    print(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*12}")
    for i in range(min(10, num_agents)):
        agent_id = data['agent']['id'][i]
        cat = categories[i]
        valid = valid_at_last[i]
        will_predict = cat == 3
        marker = "✓" if will_predict else "✗"
        av_marker = " (AV)" if i == av_index else ""
        print(f"  {agent_id:<12.1f} {cat:<10} {str(valid):<12} {marker:<12}{av_marker}")

    return {
        'scenario_id': scenario_id,
        'av_in_cat3': av_is_cat3,
        'valid_not_in_cat3': valid_not_in_cat3,
        'total_valid': valid_agents,
        'cat3_count': cat_3
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_waymo_compliance.py <path_to_pkl_file_or_directory>")
        print("\nExamples:")
        print("  python check_waymo_compliance.py data/valid_demo/1a146468873b7871.pkl")
        print("  python check_waymo_compliance.py data/waymo_processed/validation/")
        sys.exit(1)

    path = Path(sys.argv[1])

    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = list(path.glob('*.pkl'))[:5]  # Check first 5 files
        print(f"Found {len(list(path.glob('*.pkl')))} .pkl files, checking first 5...")
    else:
        print(f"Error: {path} not found!")
        sys.exit(1)

    results = []
    for pkl_file in files:
        try:
            result = check_scenario(pkl_file)
            results.append(result)
        except Exception as e:
            print(f"\nError processing {pkl_file}: {e}")

    # Summary
    if len(results) > 1:
        print(f"\n{'='*80}")
        print(f"SUMMARY ({len(results)} scenarios)")
        print(f"{'='*80}")

        av_in_cat3_count = sum(1 for r in results if r['av_in_cat3'])
        has_missing = sum(1 for r in results if r['valid_not_in_cat3'] > 0)

        print(f"\nAV/Ego Vehicle:")
        print(f"  AV in category 3: {av_in_cat3_count}/{len(results)} scenarios")

        print(f"\nValid Agents Coverage:")
        print(f"  Scenarios with valid agents NOT in category 3: {has_missing}/{len(results)}")

        if av_in_cat3_count == len(results):
            print(f"\n✓ All scenarios include AV in predictions")
        else:
            print(f"\n✗ WARNING: Some scenarios missing AV in predictions!")

        if has_missing == 0:
            print(f"✓ All valid agents are in category 3 (tracks_to_predict)")
        else:
            print(f"⚠ Some scenarios have valid agents not in category 3")


if __name__ == '__main__':
    main()
