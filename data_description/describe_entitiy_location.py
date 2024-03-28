import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

parser = argparse.ArgumentParser(
        description='This script is used to describe the relative positions of entities in the text.')

parser.add_argument('-i', '--input', type=str, required=True,
                    help='Choose the input CSV file.')

args = parser.parse_args()

if not args.input.endswith('.csv'):
    raise ValueError('Input file needs to be defined as a CSV-file')

df = pd.read_csv(args.input, sep='|', header=None, names=['Text', 'Annotations'])
df['Annotations'] = df['Annotations'].str.strip().str.split(' ')

def extract_entities(annotations):
    return [(i, tag) for i, tag in enumerate(annotations) if tag not in ['O']]

df['EntityPositions'] = df['Annotations'].apply(extract_entities)

def calculate_relative_positions(dataframe):
    relative_positions = []
    for _, row in dataframe.iterrows():
        annotations = row['Annotations']
        positions = [i for i, tag in enumerate(annotations) if tag not in ['O']]
        total_length = len(annotations)
        relative_positions.extend([pos / total_length for pos in positions])
    return relative_positions

relative_positions = calculate_relative_positions(df)

plt.figure(figsize=(12, 6))
sns.kdeplot(relative_positions, bw_adjust=0.5, fill=True)
plt.xlabel("Relative Word Position")
plt.ylabel("Density")
plt.title("Density Plot of Entities Across Relative Word Positions")
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.hist(relative_positions, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel("Relative Word Position")
plt.ylabel("Frequency")
plt.title("Distribution of Entities Across Relative Word Positions")
plt.grid(axis='y')
plt.show()

bin_edges = np.linspace(0, 1, 21)  # 20 bins from 0% to 100%
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Calculate bin centers
hist, _ = np.histogram(relative_positions, bins=bin_edges)

plt.figure(figsize=(12, 6))
plt.plot(bin_centers, hist, marker='o', linestyle='-', color='purple')
plt.xlabel("Relative Word Position")
plt.ylabel("Frequency")
plt.title("Distribution Graph of Entities Across Relative Word Positions")
plt.grid(True)
plt.show()

