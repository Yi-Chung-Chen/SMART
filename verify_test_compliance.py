"""
Verify if TEST set tracks_to_predict includes all required agents for Waymo compliance.

Run this AFTER preprocessing the test set:
    python data_preprocess.py --input_dir ./data/waymo/scenario/testing --output_dir ./data/waymo_processed/testing
    python verify_test_compliance.py ./data/waymo_processed/testing/
"""

import sys
import pickle
from pathlib import Path
from collections import defaultdict

def check_test_scenario(pkl_file):
    """Check if a test scenario's category 3 includes all valid agents."""
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    scenario_id = data.get('scenario_id', 'unknown')
    categories = data['agent']['category']
    valid_mask = data['agent']['valid_mask']

    last_history_step = 10
    valid_at_last = valid_mask[:, last_history_step]

    total_valid = valid_at_last.sum().item()
    valid_in_cat3 = (valid_at_last & (categories == 3)).sum().item()
    valid_not_in_cat3 = (valid_at_last & (categories != 3)).sum().item()

    av_index = data['agent']['av_index']
    av_category = categories[av_index]

    return {
        'scenario_id': scenario_id,
        'total_valid': total_valid,
        'cat3_count': valid_in_cat3,
        'missing_count': valid_not_in_cat3,
        'av_in_cat3': av_category == 3,
        'coverage': valid_in_cat3 / total_valid if total_valid > 0 else 0
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_test_compliance.py <test_pkl_directory>")
        print("Example: python verify_test_compliance.py ./data/waymo_processed/testing/")
        sys.exit(1)

    test_dir = Path(sys.argv[1])
    if not test_dir.is_dir():
        print(f"Error: {test_dir} is not a directory!")
        sys.exit(1)

    pkl_files = list(test_dir.glob('*.pkl'))
    if len(pkl_files) == 0:
        print(f"Error: No .pkl files found in {test_dir}")
        sys.exit(1)

    print(f"Checking {len(pkl_files)} test scenarios...")
    print(f"{'='*80}\n")

    results = []
    sample_size = min(100, len(pkl_files))  # Check first 100 scenarios

    for pkl_file in pkl_files[:sample_size]:
        try:
            result = check_test_scenario(pkl_file)
            results.append(result)
        except Exception as e:
            print(f"Error processing {pkl_file.name}: {e}")

    # Analyze results
    full_coverage = sum(1 for r in results if r['coverage'] == 1.0)
    av_included = sum(1 for r in results if r['av_in_cat3'])
    has_missing = sum(1 for r in results if r['missing_count'] > 0)

    avg_coverage = sum(r['coverage'] for r in results) / len(results)
    avg_valid = sum(r['total_valid'] for r in results) / len(results)
    avg_cat3 = sum(r['cat3_count'] for r in results) / len(results)

    print(f"ANALYSIS OF {sample_size} TEST SCENARIOS")
    print(f"{'='*80}")
    print(f"\nCoverage Statistics:")
    print(f"  Average valid agents per scenario: {avg_valid:.1f}")
    print(f"  Average category 3 agents: {avg_cat3:.1f}")
    print(f"  Average coverage: {avg_coverage*100:.1f}%")
    print(f"\nScenario Breakdown:")
    print(f"  Full coverage (100%): {full_coverage}/{sample_size} ({full_coverage/sample_size*100:.1f}%)")
    print(f"  AV in category 3: {av_included}/{sample_size} ({av_included/sample_size*100:.1f}%)")
    print(f"  Has missing agents: {has_missing}/{sample_size} ({has_missing/sample_size*100:.1f}%)")

    print(f"\n{'='*80}")
    print(f"VERDICT:")
    print(f"{'='*80}\n")

    if full_coverage == sample_size and av_included == sample_size:
        print("✅ TEST SET IS COMPLIANT!")
        print("   All scenarios have 100% coverage in category 3 (tracks_to_predict)")
        print("   AV is included in all scenarios")
        print("   SMART's implementation is CORRECT for test set")
        print("\n   The filter 'agent_category != 3' is appropriate.")
    elif avg_coverage >= 0.95:
        print("⚠️  MOSTLY COMPLIANT")
        print(f"   {avg_coverage*100:.1f}% average coverage")
        print("   Most required agents are in category 3")
        print("   Small discrepancies may be acceptable")
    else:
        print("❌ TEST SET MAY NOT BE FULLY COMPLIANT")
        print(f"   Only {avg_coverage*100:.1f}% coverage on average")
        print(f"   {has_missing} scenarios have missing agents")
        print("\n   You may need to modify the model to predict ALL valid agents,")
        print("   not just category 3.")
        print("\n   Suggested fix in smart/modules/agent_decoder.py:")
        print("   Change line ~495 from:")
        print("       agent_valid_mask[agent_category != 3] = False")
        print("   To:")
        print("       # Predict all agents valid at last history step")
        print("       current_valid = data['agent']['valid_mask'][:, self.num_historical_steps - 1]")
        print("       agent_valid_mask = current_valid[:, None].expand_as(agent_valid_mask)")

    # Show some examples
    if has_missing > 0:
        print(f"\n\nExample scenarios WITH missing agents:")
        count = 0
        for r in results:
            if r['missing_count'] > 0 and count < 5:
                print(f"  {r['scenario_id']}: {r['cat3_count']}/{r['total_valid']} agents ({r['coverage']*100:.0f}% coverage)")
                count += 1

if __name__ == '__main__':
    main()
