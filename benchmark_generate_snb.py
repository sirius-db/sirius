import duckdb
import random

def generate_graph(num_persons, min_friends, max_friends, output_file):
    """Generate a synthetic social network graph with weighted edges"""
    conn = duckdb.connect(output_file)

    print(f"Generating graph with {num_persons:,} persons...")

    # Generate persons
    conn.execute(f"""
        CREATE TABLE Person AS
        SELECT i AS id,
               'Person_' || i::VARCHAR AS firstName,
               'Last_' || i::VARCHAR AS lastName
        FROM range({num_persons}) t(i)
    """)

    # Generate edges with weights
    print(f"Generating weighted edges (avg {(min_friends + max_friends) / 2} friends per person)...")
    edges = []
    for person in range(num_persons):
        num_friends = random.randint(min_friends, max_friends)
        num_friends = min(num_friends, num_persons - 1)
        friends = random.sample([p for p in range(num_persons) if p != person], num_friends)
        for friend in friends:
            # Add random weight between 1.0 and 10.0 (e.g., interaction strength)
            weight = round(random.uniform(1.0, 10.0), 2)
            edges.append((person, friend, weight))

    # Remove duplicates (keep first weight if duplicate edge)
    edges_dict = {}
    for src, dst, weight in edges:
        if (src, dst) not in edges_dict:
            edges_dict[(src, dst)] = weight

    edges = [(src, dst, weight) for (src, dst), weight in edges_dict.items()]

    conn.execute("""
                 CREATE TABLE Person_knows_Person (
                    source BIGINT,
                    destination BIGINT,
                    weight DOUBLE
                 )
                 """)
    conn.executemany("INSERT INTO Person_knows_Person VALUES (?, ?, ?)", edges)

    print(f"✓ Generated {len(edges):,} weighted edges")
    print(f"✓ Average degree: {len(edges) / num_persons:.1f} friends per person")
    print(f"✓ Saved to {output_file}\n")

    conn.close()
    return len(edges)

# num of persons, min num of friends, max num of friends, output filename
configs = [
    (1000, 3, 8, 'snb_1k.duckdb'),
    (10000, 3, 8, 'snb_10k.duckdb'),
    (50000, 3, 8, 'snb_50k.duckdb'),
    (100000, 3, 8, 'snb_100k.duckdb'),
    (500000, 3, 8, 'snb_500k.duckdb'),
]

print("=" * 60)
print("GENERATING BENCHMARK DATASETS")
print("=" * 60)

for num_persons, min_friends, max_friends, output_file in configs:
    generate_graph(num_persons, min_friends, max_friends, output_file)

print("✓ All test databases generated!")
