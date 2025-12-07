--------------------------------------------
-- DuckPGQ CPU Benchmark
-- (run on the instance, not on the docker image)
-- To install:
-- wget https://github.com/duckdb/duckdb/releases/download/v1.1.3/duckdb_cli-linux-amd64.zip
-- unzip duckdb_cli-linux-amd64.zip
-- sudo mv duckdb /usr/local/bin/
-- To test performance:
-- 1. copy xx.duckdb to instance root
-- 2. run duckdb on instance root
-- 3. run the script in duckdb using .read graph_test_snb_performance_duckpgq.sql
--    (adjust parameters as needed, e.g., file names)
--------------------------------------------


-- Setup
ATTACH 'snb_large_100k.duckdb' AS snb;
USE snb;

INSTALL duckpgq FROM community;
LOAD duckpgq;

CREATE OR REPLACE PROPERTY GRAPH snb
  VERTEX TABLES (Person)
  EDGE TABLES (
    Person_knows_Person SOURCE KEY (source) REFERENCES Person (id)
                        DESTINATION KEY (destination) REFERENCES Person (id)
                        LABEL knows
  );

-- Warmup
FROM GRAPH_TABLE (snb
    MATCH (p:person WHERE p.id = 14)-[k:knows]->(p2:person)
    COLUMNS (p2.id)
);

-- Query 1: Direct neighbors
.timer on
FROM GRAPH_TABLE (snb
    MATCH (p:person WHERE p.id = 14)-[k:knows]->(p2:person)
    COLUMNS (p2.id)
);
.timer off

-- Query 2: BFS zero-or-more (with upper bound)
.timer on
FROM GRAPH_TABLE (snb
  MATCH p = ANY SHORTEST (p1:person WHERE p1.id = 14)-[k:knows]->*(p2:person)
  COLUMNS (p2.id as other_person_id, path_length(p))
);
.timer off

-- Query 3: ANY SHORTEST (this works without bounds)
.timer on
FROM GRAPH_TABLE (snb
    MATCH p = ANY SHORTEST (p:person WHERE p.id = 14)-[k:knows]->*(p2:person)
    COLUMNS (p2.id, path_length(p))
);
.timer off

-- Query 4: Multi-source BFS with bound
-- .timer on
-- FROM GRAPH_TABLE (snb
--     MATCH p = ANY SHORTEST (p:person WHERE p.id IN (14, 25, 37))-[k:knows]->*(p2:person)
--     COLUMNS (p2.id, path_length(p))
-- );
-- .timer off

-- Query 5: BFS one-or-more with bound
-- .timer on
-- FROM GRAPH_TABLE (snb
--     MATCH (p:person WHERE p.id = 14)-[k:knows]->+(p2:person)
--     WHERE path_length(k) <= 10
--     COLUMNS (p2.id)
-- );
-- .timer off
