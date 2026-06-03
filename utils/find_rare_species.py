#!/usr/bin/env python3
"""
BirdCLEF 2026 — Rare Species Analysis
Prints statistics about species coverage across train.csv,
taxonomy.csv and train_soundscapes_labels.csv
"""

import pandas as pd
from pathlib import Path
from collections import Counter

# =========================
# PATHS
# =========================
BASE_DIR        = Path('/Users/ganapathychidambaram/Desktop/birds/birdclef-2026')
TRAIN_CSV       = BASE_DIR / 'train.csv'
TAXONOMY_CSV    = BASE_DIR / 'taxonomy.csv'
SOUNDSCAPES_CSV = BASE_DIR / 'train_soundscapes_labels.csv'

# =========================
# LOAD DATA
# =========================
train_df    = pd.read_csv(TRAIN_CSV)
taxonomy_df = pd.read_csv(TAXONOMY_CSV)
sound_df    = pd.read_csv(SOUNDSCAPES_CSV)

# Build lookup dicts from taxonomy
tax_class  = dict(zip(taxonomy_df['primary_label'].astype(str), taxonomy_df['class_name']))
tax_name   = dict(zip(taxonomy_df['primary_label'].astype(str), taxonomy_df['common_name']))

# =========================
# 1. OVERALL STATS
# =========================
print('=' * 70)
print('OVERALL DATASET STATS')
print('=' * 70)
print(f'Total rows in train.csv              : {len(train_df):,}')
print(f'Total unique species in train.csv    : {train_df["primary_label"].nunique()}')
print(f'Total species in taxonomy.csv        : {len(taxonomy_df)}')
print(f'Total rows in soundscape labels      : {len(sound_df)}')
print(f'Unique soundscape files              : {sound_df["filename"].nunique()}')

# =========================
# 2. SPECIES MISSING FROM TRAIN.CSV
# =========================
train_species    = set(train_df['primary_label'].astype(str).unique())
taxonomy_species = set(taxonomy_df['primary_label'].astype(str).unique())
missing_species  = taxonomy_species - train_species

print()
print('=' * 70)
print(f'SPECIES IN TAXONOMY BUT NOT IN TRAIN.CSV ({len(missing_species)} species)')
print('=' * 70)
print(f'{"Species":<15} {"Common Name":<45} {"Class":<12} {"SS Count"}')
print('-' * 70)

# Count occurrences in soundscape labels
soundscape_text = sound_df['primary_label'].fillna('').astype(str)
all_soundscape_labels = []
for label_str in soundscape_text:
    all_soundscape_labels.extend([l.strip() for l in label_str.split(';')])
soundscape_counts = Counter(all_soundscape_labels)

missing_rows = []
for species in sorted(missing_species):
    common = tax_name.get(species, 'Unknown')
    cls    = tax_class.get(species, 'Unknown')
    ss_cnt = soundscape_counts.get(species, 0)
    missing_rows.append((species, common, cls, ss_cnt))

missing_rows.sort(key=lambda x: x[3], reverse=True)
for species, common, cls, ss_cnt in missing_rows:
    print(f'{species:<15} {common:<45} {cls:<12} {ss_cnt}')

# =========================
# 3. RARE SPECIES IN TRAIN.CSV
# =========================
train_counts = train_df['primary_label'].astype(str).value_counts()

print()
print('=' * 70)
print('RARE SPECIES IN TRAIN.CSV (< 10 samples)')
print('=' * 70)
print(f'{"Species":<15} {"Common Name":<45} {"Class":<12} {"Count"}')
print('-' * 70)
rare_10 = train_counts[train_counts < 10]
for species, count in rare_10.sort_values().items():
    common = tax_name.get(species, 'Unknown')
    cls    = tax_class.get(species, 'Unknown')
    print(f'{species:<15} {common:<45} {cls:<12} {count}')

print()
print('=' * 70)
print('RARE SPECIES IN TRAIN.CSV (< 50 samples)')
print('=' * 70)
print(f'{"Species":<15} {"Common Name":<45} {"Class":<12} {"Count"}')
print('-' * 70)
rare_50 = train_counts[train_counts < 50]
for species, count in rare_50.sort_values().items():
    common = tax_name.get(species, 'Unknown')
    cls    = tax_class.get(species, 'Unknown')
    print(f'{species:<15} {common:<45} {cls:<12} {count}')

# =========================
# 4. SPECIES ONLY IN SOUNDSCAPES
# =========================
soundscape_only = set(soundscape_counts.keys()) - train_species - {'', 'nan'}
print()
print('=' * 70)
print(f'SPECIES IN SOUNDSCAPES BUT NOT IN TRAIN.CSV ({len(soundscape_only)} species)')
print('=' * 70)
print(f'{"Species":<15} {"Common Name":<45} {"Class":<12} {"SS Count"}')
print('-' * 70)
ss_only_rows = []
for species in soundscape_only:
    common = tax_name.get(species, 'Unknown')
    cls    = tax_class.get(species, 'Unknown')
    ss_cnt = soundscape_counts.get(species, 0)
    ss_only_rows.append((species, common, cls, ss_cnt))

ss_only_rows.sort(key=lambda x: x[3], reverse=True)
for species, common, cls, ss_cnt in ss_only_rows:
    print(f'{species:<15} {common:<45} {cls:<12} {ss_cnt}')

# =========================
# 5. SUMMARY BY CLASS
# =========================
print()
print('=' * 70)
print('SUMMARY BY CLASS')
print('=' * 70)
print(f'{"Class":<12} {"In Train":<12} {"Missing":<12} {"Total in Taxonomy"}')
print('-' * 70)
for cls in sorted(taxonomy_df['class_name'].unique()):
    cls_species  = set(taxonomy_df[taxonomy_df['class_name'] == cls]['primary_label'].astype(str))
    in_train     = len(cls_species & train_species)
    missing      = len(cls_species - train_species)
    total        = len(cls_species)
    print(f'{cls:<12} {in_train:<12} {missing:<12} {total}')

# =========================
# 6. BUCKET DISTRIBUTION
# =========================
print()
print('=' * 70)
print('TRAINING SAMPLE DISTRIBUTION')
print('=' * 70)
buckets = [(1, 1), (2, 5), (6, 10), (11, 50), (51, 100), (101, 500), (501, 99999)]
for lo, hi in buckets:
    count = ((train_counts >= lo) & (train_counts <= hi)).sum()
    label = f'{lo}-{hi}' if hi != 99999 else f'{lo}+'
    print(f'  {label:<10} samples : {count} species')

print()
print('Script complete.')
