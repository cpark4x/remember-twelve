#!/usr/bin/env python3
"""
Regenerate month distribution for existing twelve_2023_balanced.json
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, 'src')

from twelve_curator import TwelveCurator
from twelve_curator.data_classes import CurationConfig

def main():
    # Load existing data
    json_file = Path('twelve_2023_balanced.json')
    with open(json_file, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data['photos'])} photos from {json_file}")

    # Create curator
    curator = TwelveCurator(CurationConfig(strategy=data.get('strategy', 'balanced')))

    # Generate flexible distribution
    month_distribution = curator.distribute_to_twelve_months(
        data['photos'],
        flexible=True
    )

    # Update data with month_distribution
    data['month_distribution'] = month_distribution

    # Count filled months
    filled_months = sum(1 for photo in month_distribution.values() if photo is not None)
    print(f"✓ Distributed photos across {filled_months} months")

    # Save updated data to ui/
    output_file = Path('ui/twelve_2023_balanced.json')
    output_file.write_text(json.dumps(data, indent=2))
    print(f"✓ Saved to {output_file}")

if __name__ == '__main__':
    main()
