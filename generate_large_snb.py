import duckdb
import random

# ============================================
# PARAMETERS
# ============================================
NUM_PERSONS = 100000
MIN_FRIENDS = 3          # Minimum friends per person
MAX_FRIENDS = 8          # Maximum friends per person
OUTPUT_FILE = 'snb_large_100k.duckdb'
# ============================================

conn = duckdb.connect(OUTPUT_FILE)

print(f"Generating graph with {NUM_PERSONS} persons...")

# Generate persons
conn.execute(f"""
    CREATE TABLE Person AS
    SELECT i AS id,
           'Person_' || i::VARCHAR AS firstName,
           'Last_' || i::VARCHAR AS lastName
    FROM range({NUM_PERSONS}) t(i)
""")

# Generate edges (realistic social graph)
print(f"Generating edges (avg {(MIN_FRIENDS + MAX_FRIENDS) / 2} friends per person)...")
edges = []
for person in range(NUM_PERSONS):
    num_friends = random.randint(MIN_FRIENDS, MAX_FRIENDS)
    friends = random.sample(range(NUM_PERSONS), num_friends)
    for friend in friends:
        if person != friend:  # No self-loops
            edges.append((person, friend))

# Remove duplicates
edges = list(set(edges))

conn.execute("CREATE TABLE Person_knows_Person (source BIGINT, destination BIGINT)")
conn.executemany("INSERT INTO Person_knows_Person VALUES (?, ?)", edges)

print(f"✓ Generated {len(edges)} edges")
print(f"✓ Average degree: {len(edges) / NUM_PERSONS:.1f} friends per person")

# Verify
person_count = conn.execute("SELECT COUNT(*) FROM Person").fetchone()[0]
edge_count = conn.execute("SELECT COUNT(*) FROM Person_knows_Person").fetchone()[0]

print(f"✓ Final counts: {person_count} persons, {edge_count} edges")
print(f"✓ Saved to {OUTPUT_FILE}")