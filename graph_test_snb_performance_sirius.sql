-- To run performance test
-- ./build/release/duckdb xx.duckdb
-- .read graph_test_snb_performance_sirius.sql

-- Setup
CALL gpu_buffer_init('8GB', '8GB');
CALL gpu_processing("SELECT * FROM Person");
CALL gpu_processing("SELECT * FROM Person_knows_Person");

-- Warmup
CALL gpu_processing_graph("
    FROM GRAPH_TABLE (snb
        MATCH (p:person WHERE p.id = 14)-[k:Person_knows_Person]->(p2:person)
        COLUMNS (p2.id)
    )
");

-- Query 1: Direct neighbors
.timer on
CALL gpu_processing_graph("
    FROM GRAPH_TABLE (snb
        MATCH (p:person WHERE p.id = 14)-[k:Person_knows_Person]->(p2:person)
        COLUMNS (p2.id, distance, predecessor)
    )
");
.timer off

-- Query 2: BFS zero-or-more
.timer on
CALL gpu_processing_graph("
    FROM GRAPH_TABLE (snb
        MATCH (p:person WHERE p.id = 14)-[k:Person_knows_Person]->*(p2:person)
        COLUMNS (p2.id, distance, predecessor)
    )
");
.timer off

-- Query 3: ANY SHORTEST (find a hub vertex first with high degree)
.timer on
CALL gpu_processing_graph("
    FROM GRAPH_TABLE (snb
        MATCH p = ANY SHORTEST (p:person WHERE p.id = 14)-[k:Person_knows_Person]->*(p2:person)
        COLUMNS (p2.id, distance)
    )
");
.timer off

-- Query 4: Weighted query from a source
CALL gpu_processing_graph("
  SELECT * FROM GRAPH_TABLE (social
    MATCH SHORTEST (p:Person_w WHERE p.id=14)-[w:Person_knows_Person]->*(p2:Person_w)
    COLUMNS (p2.id, distance, predecessor)
  )
");


-- Query 4: Multi-source BFS
-- .timer on
-- CALL gpu_processing_graph("
--     FROM GRAPH_TABLE (snb
--         MATCH (p:person WHERE p.id IN (14, 25, 37))-[k:Person_knows_Person]->*(p2:person)
--         COLUMNS (p2.id, distance)
--     )
-- ");
-- .timer off

-- Query 5: BFS one-or-more (excludes source)
-- .timer on
-- CALL gpu_processing_graph("
--     FROM GRAPH_TABLE (snb
--         MATCH (p:person WHERE p.id = 14)-[k:Person_knows_Person]->+(p2:person)
--         COLUMNS (p2.id, distance)
--     )
-- ");
-- .timer off