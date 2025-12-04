CALL gpu_buffer_init('1GB', '1GB');

CALL gpu_processing("SELECT * FROM Person");
CALL gpu_processing("SELECT * FROM Person_knows_Person");

-- Modify column names to source and destination (since we hard coded that)
-- (only need to run once since duckdb is disk-based and the changes are persistent)

-- ALTER TABLE Person_knows_Person RENAME COLUMN Person1Id TO source;
-- ALTER TABLE Person_knows_Person RENAME COLUMN Person2Id TO destination;

-- ALTER TABLE Person DROP COLUMN creationDate;
-- ALTER TABLE Person_knows_Person DROP COLUMN creationDate;

-- ============================================
-- SECTION 1: EDGE TRAVERSAL OPERATORS (->)
-- Pattern: ()-[]->() - Direct neighbors only
-- ============================================

-- Test 1.1: Direct neighbors from vertex 14
-- Pattern: Simple right-directed edge traversal
-- Expected: All direct 1-hop neighbors
CALL gpu_processing_graph("
    FROM GRAPH_TABLE (snb
        MATCH (p1:person WHERE p1.id = 14)-[k:Person_knows_Person]->(p2:person)
        COLUMNS (p2.id, distance)
    )
");
-- Expected: 3 vertices with distance=1
-- ┌────────────────┬──────────┐
-- │       id       │ distance │
-- │     int64      │  int64   │
-- ├────────────────┼──────────┤
-- │ 10995116277782 │        1 │
-- │ 24189255811081 │        1 │
-- │ 26388279066668 │        1 │
-- └────────────────┴──────────┘
-- Compare with DuckPGQ:
-- FROM GRAPH_TABLE (snb
--     MATCH (p1:person WHERE p1.id = 14)-[k:knows]->(p2:person)
--     COLUMNS (p2.id)
-- );


-- Test 1.2: Direct neighbors to specific target
-- Pattern: Edge traversal with target filter
-- Expected: Single edge if direct connection exists
CALL gpu_processing_graph("
    FROM GRAPH_TABLE (snb
        MATCH (p1:person WHERE p1.id = 14)-[k:Person_knows_Person]->(p2:person WHERE p2.id = 10995116277782)
        COLUMNS (p2.id, distance, predecessor)
    )
");
-- Expected: 1 row if direct edge exists, 0 rows otherwise
-- ┌────────────────┬──────────┬─────────────┐
-- │       id       │ distance │ predecessor │
-- │     int64      │  int64   │    int64    │
-- ├────────────────┼──────────┼─────────────┤
-- │ 10995116277782 │    1     │     14      │
-- └────────────────┴──────────┴─────────────┘


-- Test 1.3: Direct neighbors from highly connected vertex
-- Pattern: Edge traversal from hub
-- Expected: All direct connections from a hub vertex
CALL gpu_processing_graph("
    FROM GRAPH_TABLE (snb
        MATCH (p1:person WHERE p1.id = 10995116277782)-[k:Person_knows_Person]->(p2:person)
        COLUMNS (p2.id, distance)
    )
");
-- Expected: Multiple direct neighbors (hub vertex)
-- ┌────────────────┬──────────┐
-- │       id       │ distance │
-- │     int64      │  int64   │
-- ├────────────────┼──────────┤
-- │ 26388279066641 │        1 │
-- │ 26388279066658 │        1 │
-- │ 28587302322180 │        1 │
-- │ 28587302322204 │        1 │
-- │ 35184372088856 │        1 │
-- └────────────────┴──────────┘


-- Test 1.4: Direct neighbors from leaf vertex
-- Pattern: Edge traversal from low-degree vertex
-- Expected: Few or no direct neighbors
CALL gpu_processing_graph("
    FROM GRAPH_TABLE (snb
        MATCH (p1:person WHERE p1.id = 35184372088856)-[k:Person_knows_Person]->(p2:person)
        COLUMNS (p2.id, distance)
    )
");
-- Expected: 0 or very few rows
-- ┌───────┬──────────┐
-- │  id   │ distance │
-- │ int64 │  int64   │
-- ├───────┴──────────┤
-- │      0 rows      │
-- └──────────────────┘


-- Test 1.5: Edge traversal with only ID column
-- Pattern: Minimal output (just IDs)
CALL gpu_processing_graph("
    FROM GRAPH_TABLE (snb
        MATCH (p1:person WHERE p1.id = 14)-[k:Person_knows_Person]->(p2:person)
        COLUMNS (p2.id)
    )
");
-- Expected: Just vertex IDs, no distance/predecessor
-- ┌────────────────┐
-- │       id       │
-- │     int64      │
-- ├────────────────┤
-- │ 10995116277782 │
-- │ 24189255811081 │
-- │ 26388279066668 │
-- └────────────────┘


-- ============================================
-- SECTION 2: BFS ZERO-OR-MORE (->*)
-- Pattern: ()->*() - Includes source vertex
-- ============================================

-- Test 2.1: BFS zero-or-more from vertex 14
-- Pattern: Full reachability including source
-- Expected: All reachable vertices + source (distance 0)
CALL gpu_processing_graph("
    FROM GRAPH_TABLE (snb
        MATCH (p1:person WHERE p1.id = 14)-[k:Person_knows_Person]->*(p2:person)
        COLUMNS (p2.id, distance, predecessor)
    )
");
-- Expected: 17 rows INCLUDING vertex 14 with distance=0
-- ┌────────────────┬──────────┬────────────────┐
-- │       id       │ distance │  predecessor   │
-- │     int64      │  int64   │     int64      │
-- ├────────────────┼──────────┼────────────────┤
-- │             14 │        0 │             -1 │
-- │ 10995116277782 │        1 │             14 │
-- │ 24189255811081 │        1 │             14 │
-- │ 24189255811109 │        2 │ 24189255811081 │
-- │ 26388279066641 │        2 │ 10995116277782 │
-- │ 26388279066658 │        2 │ 10995116277782 │
-- │ 26388279066668 │        1 │             14 │
-- │ 28587302322180 │        2 │ 10995116277782 │
-- │ 28587302322196 │        2 │ 24189255811081 │
-- │ 28587302322204 │        2 │ 10995116277782 │
-- │ 28587302322223 │        3 │ 28587302322180 │
-- │ 30786325577731 │        3 │ 28587302322180 │
-- │ 30786325577740 │        3 │ 26388279066658 │
-- │ 32985348833329 │        3 │ 28587302322180 │
-- │ 35184372088834 │        2 │ 24189255811081 │
-- │ 35184372088850 │        3 │ 26388279066658 │
-- │ 35184372088856 │        2 │ 10995116277782 │
-- ├────────────────┴──────────┴────────────────┤
-- │ 17 rows                          3 columns │
-- └────────────────────────────────────────────┘


-- Test 2.2: BFS zero-or-more from different source
-- Pattern: Full reachability from vertex 16
CALL gpu_processing_graph("
    FROM GRAPH_TABLE (snb
        MATCH (p1:person WHERE p1.id = 16)-[k:Person_knows_Person]->*(p2:person)
        COLUMNS (p2.id, distance)
    )
");
-- Expected: All vertices reachable from 16, including 16 itself
-- ┌────────────────┬──────────┐
-- │       id       │ distance │
-- │     int64      │  int64   │
-- ├────────────────┼──────────┤
-- │             16 │        0 │
-- │  2199023255594 │        1 │
-- │  8796093022244 │        2 │
-- │ 10995116277761 │        2 │
-- │ 13194139533342 │        2 │
-- │ 13194139533352 │        2 │
-- │ 13194139533355 │        3 │
-- │ 15393162788877 │        2 │
-- │ 17592186044443 │        3 │
-- │ 17592186044461 │        2 │
-- │ 19791209299987 │        3 │
-- │ 24189255811081 │        2 │
-- │ 24189255811109 │        3 │
-- │ 26388279066641 │        3 │
-- │ 26388279066655 │        1 │
-- │ 26388279066658 │        2 │
-- │ 26388279066668 │        2 │
-- │ 28587302322180 │        1 │
-- │ 28587302322196 │        2 │
-- │ 28587302322204 │        1 │
-- │ 28587302322223 │        2 │
-- │ 30786325577731 │        2 │
-- │ 30786325577740 │        2 │
-- │ 32985348833329 │        2 │
-- │ 35184372088834 │        3 │
-- │ 35184372088850 │        2 │
-- │ 35184372088856 │        3 │
-- ├────────────────┴──────────┤
-- │ 27 rows         2 columns │
-- └───────────────────────────┘


-- Test 2.3: BFS zero-or-more from hub vertex
-- Pattern: Explore from highly connected vertex
CALL gpu_processing_graph("
    FROM GRAPH_TABLE (snb
        MATCH (p1:person WHERE p1.id = 10995116277782)-[k:Person_knows_Person]->*(p2:person)
        COLUMNS (p2.id, distance)
    )
");
-- Expected: Large reachability set (hub has many connections)
-- ┌────────────────┬──────────┐
-- │       id       │ distance │
-- │     int64      │  int64   │
-- ├────────────────┼──────────┤
-- │ 10995116277782 │        0 │
-- │ 26388279066641 │        1 │
-- │ 26388279066658 │        1 │
-- │ 28587302322180 │        1 │
-- │ 28587302322196 │        2 │
-- │ 28587302322204 │        1 │
-- │ 28587302322223 │        2 │
-- │ 30786325577731 │        2 │
-- │ 30786325577740 │        2 │
-- │ 32985348833329 │        2 │
-- │ 35184372088850 │        2 │
-- │ 35184372088856 │        1 │
-- ├────────────────┴──────────┤
-- │ 12 rows         2 columns │
-- └───────────────────────────┘


-- Test 2.4: BFS zero-or-more with ID only
-- Pattern: Minimal output
CALL gpu_processing_graph("
    FROM GRAPH_TABLE (snb
        MATCH (p1:person WHERE p1.id = 14)-[k:Person_knows_Person]->*(p2:person)
        COLUMNS (p2.id)
    )
");
-- Expected: 17 vertex IDs
-- ┌────────────────┐
-- │       id       │
-- │     int64      │
-- ├────────────────┤
-- │             14 │
-- │ 10995116277782 │
-- │ 24189255811081 │
-- │ 24189255811109 │
-- │ 26388279066641 │
-- │ 26388279066658 │
-- │ 26388279066668 │
-- │ 28587302322180 │
-- │ 28587302322196 │
-- │ 28587302322204 │
-- │ 28587302322223 │
-- │ 30786325577731 │
-- │ 30786325577740 │
-- │ 32985348833329 │
-- │ 35184372088834 │
-- │ 35184372088850 │
-- │ 35184372088856 │
-- ├────────────────┤
-- │    17 rows     │
-- └────────────────┘


-- Test 2.5: BFS zero-or-more to specific target
-- Pattern: Check if path exists (including 0-length)
CALL gpu_processing_graph("
    FROM GRAPH_TABLE (snb
        MATCH (p1:person WHERE p1.id = 14)-[k:Person_knows_Person]->*(p2:person WHERE p2.id = 14)
        COLUMNS (p2.id, distance, predecessor)
    )
");
-- Expected: 1 row: 14, 0, -1 (source to itself)
-- ┌───────┬──────────┬─────────────┐
-- │  id   │ distance │ predecessor │
-- │ int64 │  int64   │    int64    │
-- ├───────┼──────────┼─────────────┤
-- │  14   │    0     │     -1      │
-- └───────┴──────────┴─────────────┘


-- ============================================
-- SECTION 3: BFS ONE-OR-MORE (->+)
-- Pattern: ()->+() - Excludes source vertex
-- ============================================

-- Test 3.1: BFS one-or-more from vertex 14
-- Pattern: Full reachability excluding source
-- Expected: All reachable vertices EXCEPT source
CALL gpu_processing_graph("
    FROM GRAPH_TABLE (snb
        MATCH (p1:person WHERE p1.id = 14)-[k:Person_knows_Person]->+(p2:person)
        COLUMNS (p2.id, distance, predecessor)
    )
");
-- Expected: 16 rows, NO vertex 14 (source excluded)
-- ┌────────────────┬──────────┬────────────────┐
-- │       id       │ distance │  predecessor   │
-- │     int64      │  int64   │     int64      │
-- ├────────────────┼──────────┼────────────────┤
-- │ 10995116277782 │        1 │             14 │
-- │ 24189255811081 │        1 │             14 │
-- │ 24189255811109 │        2 │ 24189255811081 │
-- │ 26388279066641 │        2 │ 10995116277782 │
-- │ 26388279066658 │        2 │ 10995116277782 │
-- │ 26388279066668 │        1 │             14 │
-- │ 28587302322180 │        2 │ 10995116277782 │
-- │ 28587302322196 │        2 │ 24189255811081 │
-- │ 28587302322204 │        2 │ 10995116277782 │
-- │ 28587302322223 │        3 │ 28587302322180 │
-- │ 30786325577731 │        3 │ 28587302322180 │
-- │ 30786325577740 │        3 │ 26388279066658 │
-- │ 32985348833329 │        3 │ 28587302322180 │
-- │ 35184372088834 │        2 │ 24189255811081 │
-- │ 35184372088850 │        3 │ 26388279066658 │
-- │ 35184372088856 │        2 │ 10995116277782 │
-- ├────────────────┴──────────┴────────────────┤
-- │ 16 rows                          3 columns │
-- └────────────────────────────────────────────┘


-- Test 3.2: BFS one-or-more from different source
-- Pattern: Reachability excluding source
CALL gpu_processing_graph("
    FROM GRAPH_TABLE (snb
        MATCH (p1:person WHERE p1.id = 16)-[k:Person_knows_Person]->+(p2:person)
        COLUMNS (p2.id, distance)
    )
");
-- Expected: All reachable from 16, but NOT 16 itself
-- ┌────────────────┬──────────┐
-- │       id       │ distance │
-- │     int64      │  int64   │
-- ├────────────────┼──────────┤
-- │  2199023255594 │        1 │
-- │  8796093022244 │        2 │
-- │ 10995116277761 │        2 │
-- │ 13194139533342 │        2 │
-- │ 13194139533352 │        2 │
-- │ 13194139533355 │        3 │
-- │ 15393162788877 │        2 │
-- │ 17592186044443 │        3 │
-- │ 17592186044461 │        2 │
-- │ 19791209299987 │        3 │
-- │ 24189255811081 │        2 │
-- │ 24189255811109 │        3 │
-- │ 26388279066641 │        3 │
-- │ 26388279066655 │        1 │
-- │ 26388279066658 │        2 │
-- │ 26388279066668 │        2 │
-- │ 28587302322180 │        1 │
-- │ 28587302322196 │        2 │
-- │ 28587302322204 │        1 │
-- │ 28587302322223 │        2 │
-- │ 30786325577731 │        2 │
-- │ 30786325577740 │        2 │
-- │ 32985348833329 │        2 │
-- │ 35184372088834 │        3 │
-- │ 35184372088850 │        2 │
-- │ 35184372088856 │        3 │
-- ├────────────────┴──────────┤
-- │ 26 rows         2 columns │
-- └───────────────────────────┘


-- Test 3.3: BFS one-or-more from isolated/leaf vertex
-- Pattern: Check behavior with no outgoing edges
CALL gpu_processing_graph("
    FROM GRAPH_TABLE (snb
        MATCH (p1:person WHERE p1.id = 2199023255557)-[k:Person_knows_Person]->+(p2:person)
        COLUMNS (p2.id, distance)
    )
");
-- Expected:
-- ┌────────────────┬──────────┐
-- │       id       │ distance │
-- │     int64      │  int64   │
-- ├────────────────┼──────────┤
-- │  8796093022237 │        1 │
-- │ 13194139533355 │        1 │
-- │ 24189255811081 │        1 │
-- │ 24189255811109 │        2 │
-- │ 26388279066658 │        2 │
-- │ 26388279066668 │        2 │
-- │ 28587302322180 │        2 │
-- │ 28587302322196 │        2 │
-- │ 28587302322204 │        3 │
-- │ 28587302322223 │        3 │
-- │ 30786325577731 │        3 │
-- │ 30786325577740 │        3 │
-- │ 32985348833329 │        2 │
-- │ 35184372088834 │        2 │
-- │ 35184372088850 │        3 │
-- │ 35184372088856 │        3 │
-- ├────────────────┴──────────┤
-- │ 16 rows         2 columns │
-- └───────────────────────────┘


-- Test 3.4: BFS one-or-more to specific target
-- Pattern: Path exists (excluding 0-length to self)
CALL gpu_processing_graph("
    FROM GRAPH_TABLE (snb
        MATCH (p1:person WHERE p1.id = 14)-[k:Person_knows_Person]->+(p2:person WHERE p2.id = 35184372088856)
        COLUMNS (p2.id, distance, predecessor)
    )
");
-- Expected: 1 row with path to target vertex
-- ┌────────────────┬──────────┬────────────────┐
-- │       id       │ distance │  predecessor   │
-- │     int64      │  int64   │     int64      │
-- ├────────────────┼──────────┼────────────────┤
-- │ 35184372088856 │    2     │ 10995116277782 │
-- └────────────────┴──────────┴────────────────┘


-- ============================================
-- SECTION 4: ANY SHORTEST PATTERN (->*)
-- Pattern: ANY SHORTEST ()->*() - Shortest paths, excludes source
-- ============================================

-- Test 4.1: Any shortest from vertex 14
-- Pattern: Shortest paths with zero-or-more
-- Expected: Shortest paths to all vertices, EXCLUDING source
CALL gpu_processing_graph("
    FROM GRAPH_TABLE (snb
         MATCH p = ANY SHORTEST (p1:person WHERE p1.id = 14)-[k:Person_knows_Person]->*(p2:person)
        COLUMNS (p2.id, distance, predecessor)
    )
");
-- Expected: 16 rows, NO vertex 14
-- ┌────────────────┬──────────┬────────────────┐
-- │       id       │ distance │  predecessor   │
-- │     int64      │  int64   │     int64      │
-- ├────────────────┼──────────┼────────────────┤
-- │ 10995116277782 │        1 │             14 │
-- │ 24189255811081 │        1 │             14 │
-- │ 24189255811109 │        2 │ 24189255811081 │
-- │ 26388279066641 │        2 │ 10995116277782 │
-- │ 26388279066658 │        2 │ 10995116277782 │
-- │ 26388279066668 │        1 │             14 │
-- │ 28587302322180 │        2 │ 10995116277782 │
-- │ 28587302322196 │        2 │ 24189255811081 │
-- │ 28587302322204 │        2 │ 10995116277782 │
-- │ 28587302322223 │        3 │ 28587302322180 │
-- │ 30786325577731 │        3 │ 28587302322180 │
-- │ 30786325577740 │        3 │ 26388279066658 │
-- │ 32985348833329 │        3 │ 28587302322180 │
-- │ 35184372088834 │        2 │ 24189255811081 │
-- │ 35184372088850 │        3 │ 26388279066658 │
-- │ 35184372088856 │        2 │ 10995116277782 │
-- ├────────────────┴──────────┴────────────────┤
-- │ 16 rows                          3 columns │
-- └────────────────────────────────────────────┘
-- Should match DuckPGQ:
-- FROM GRAPH_TABLE (snb
--     MATCH p = ANY SHORTEST (p1:person WHERE p1.id = 14)-[k:knows]->*(p2:person)
--     COLUMNS (p2.id, path_length(p))
-- );


-- Test 4.2: Any shortest with one-or-more
-- Pattern: ANY SHORTEST ()->+()
-- Expected: Same as 4.1 (both exclude source)
CALL gpu_processing_graph("
    FROM GRAPH_TABLE (snb
         MATCH p = ANY SHORTEST (p1:person WHERE p1.id = 14)-[k:Person_knows_Person]->+(p2:person)
        COLUMNS (p2.id, distance, predecessor)
    )
");
-- Expected: Identical to Test 4.1 (16 rows, no source)
-- ┌────────────────┬──────────┬────────────────┐
-- │       id       │ distance │  predecessor   │
-- │     int64      │  int64   │     int64      │
-- ├────────────────┼──────────┼────────────────┤
-- │ 10995116277782 │        1 │             14 │
-- │ 24189255811081 │        1 │             14 │
-- │ 24189255811109 │        2 │ 24189255811081 │
-- │ 26388279066641 │        2 │ 10995116277782 │
-- │ 26388279066658 │        2 │ 10995116277782 │
-- │ 26388279066668 │        1 │             14 │
-- │ 28587302322180 │        2 │ 10995116277782 │
-- │ 28587302322196 │        2 │ 24189255811081 │
-- │ 28587302322204 │        2 │ 10995116277782 │
-- │ 28587302322223 │        3 │ 28587302322180 │
-- │ 30786325577731 │        3 │ 28587302322180 │
-- │ 30786325577740 │        3 │ 26388279066658 │
-- │ 32985348833329 │        3 │ 28587302322180 │
-- │ 35184372088834 │        2 │ 24189255811081 │
-- │ 35184372088850 │        3 │ 26388279066658 │
-- │ 35184372088856 │        2 │ 10995116277782 │
-- ├────────────────┴──────────┴────────────────┤
-- │ 16 rows                          3 columns │
-- └────────────────────────────────────────────┘


-- Test 4.3: Any shortest from different source
-- Pattern: Shortest paths from hub vertex
CALL gpu_processing_graph("
    FROM GRAPH_TABLE (snb
         MATCH p = ANY SHORTEST (p1:person WHERE p1.id = 10995116277782)-[k:Person_knows_Person]->*(p2:person)
        COLUMNS (p2.id, distance)
    )
");
-- Expected: Shortest paths from hub vertex
-- ┌────────────────┬──────────┐
-- │       id       │ distance │
-- │     int64      │  int64   │
-- ├────────────────┼──────────┤
-- │ 26388279066641 │        1 │
-- │ 26388279066658 │        1 │
-- │ 28587302322180 │        1 │
-- │ 28587302322196 │        2 │
-- │ 28587302322204 │        1 │
-- │ 28587302322223 │        2 │
-- │ 30786325577731 │        2 │
-- │ 30786325577740 │        2 │
-- │ 32985348833329 │        2 │
-- │ 35184372088850 │        2 │
-- │ 35184372088856 │        1 │
-- ├────────────────┴──────────┤
-- │ 11 rows         2 columns │
-- └───────────────────────────┘


-- Test 4.4: Any shortest to specific target
-- Pattern: Single shortest path
CALL gpu_processing_graph("
    FROM GRAPH_TABLE (snb
         MATCH p = ANY SHORTEST (p1:person WHERE p1.id = 14)-[k:Person_knows_Person]->*(p2:person WHERE p2.id = 28587302322223)
        COLUMNS (p2.id, distance, predecessor)
    )
");
-- Expected: Single row with shortest path to target
-- Distance should be 3
-- ┌────────────────┬──────────┬────────────────┐
-- │       id       │ distance │  predecessor   │
-- │     int64      │  int64   │     int64      │
-- ├────────────────┼──────────┼────────────────┤
-- │ 28587302322223 │    3     │ 28587302322180 │
-- └────────────────┴──────────┴────────────────┘


-- ============================================
-- SECTION 5: MULTI-SOURCE PATTERNS
-- Pattern: WHERE p.id IN (...) with different operators
-- ============================================

-- Test 5.1: Multi-source edge traversal
-- Pattern: Direct neighbors from multiple sources
CALL gpu_processing_graph("
    FROM GRAPH_TABLE (snb
        MATCH (p1:person WHERE p1.id IN (14, 16))-[k:Person_knows_Person]->(p2:person)
        COLUMNS (p2.id, distance, predecessor)
    )
");
-- Expected: Union of direct neighbors from both vertices
-- ┌────────────────┬──────────┬─────────────┐
-- │       id       │ distance │ predecessor │
-- │     int64      │  int64   │    int64    │
-- ├────────────────┼──────────┼─────────────┤
-- │ 10995116277782 │        1 │          14 │
-- │ 24189255811081 │        1 │          14 │
-- │ 26388279066668 │        1 │          14 │
-- │  2199023255594 │        1 │          16 │
-- │ 26388279066655 │        1 │          16 │
-- │ 28587302322180 │        1 │          16 │
-- │ 28587302322204 │        1 │          16 │
-- └────────────────┴──────────┴─────────────┘


-- Test 5.2: Multi-source BFS zero-or-more
-- Pattern: Reachability from multiple sources (includes sources)
CALL gpu_processing_graph("
    FROM GRAPH_TABLE (snb
        MATCH (p1:person WHERE p1.id IN (14, 16))-[k:Person_knows_Person]->*(p2:person)
        COLUMNS (p2.id, distance, predecessor)
    )
");
-- Expected: All reachable from either source, INCLUDING both sources
-- ┌────────────────┬──────────┬────────────────┐
-- │       id       │ distance │  predecessor   │
-- │     int64      │  int64   │     int64      │
-- ├────────────────┼──────────┼────────────────┤
-- │             14 │        0 │             -1 │
-- │             16 │        0 │             -1 │
-- │  2199023255594 │        1 │             16 │
-- │  8796093022244 │        2 │  2199023255594 │
-- │ 10995116277761 │        2 │  2199023255594 │
-- │ 10995116277782 │        1 │             14 │
-- │ 13194139533342 │        2 │  2199023255594 │
-- │ 13194139533352 │        2 │  2199023255594 │
-- │ 13194139533355 │        3 │  8796093022244 │
-- │ 15393162788877 │        2 │  2199023255594 │
-- │ 17592186044443 │        3 │ 10995116277761 │
-- │ 17592186044461 │        2 │  2199023255594 │
-- │ 19791209299987 │        3 │ 13194139533352 │
-- │ 24189255811081 │        1 │             14 │
-- │ 24189255811109 │        2 │ 24189255811081 │
-- │ 26388279066641 │        2 │ 10995116277782 │
-- │ 26388279066655 │        1 │             16 │
-- │ 26388279066658 │        2 │  2199023255594 │
-- │ 26388279066668 │        1 │             14 │
-- │ 28587302322180 │        1 │             16 │
-- │ 28587302322196 │        2 │  2199023255594 │
-- │ 28587302322204 │        1 │             16 │
-- │ 28587302322223 │        2 │ 28587302322180 │
-- │ 30786325577731 │        2 │ 28587302322180 │
-- │ 30786325577740 │        2 │  2199023255594 │
-- │ 32985348833329 │        2 │ 28587302322180 │
-- │ 35184372088834 │        2 │ 24189255811081 │
-- │ 35184372088850 │        2 │  2199023255594 │
-- │ 35184372088856 │        2 │ 10995116277782 │
-- ├────────────────┴──────────┴────────────────┤
-- │ 29 rows                          3 columns │
-- └────────────────────────────────────────────┘


-- Test 5.3: Multi-source BFS one-or-more
-- Pattern: Reachability from multiple sources (excludes sources)
CALL gpu_processing_graph("
    FROM GRAPH_TABLE (snb
        MATCH (p1:person WHERE p1.id IN (14, 16))-[k:Person_knows_Person]->+(p2:person)
        COLUMNS (p2.id, distance, predecessor)
    )
");
-- Expected: All reachable from either source, EXCLUDING sources
-- ┌────────────────┬──────────┬────────────────┐
-- │       id       │ distance │  predecessor   │
-- │     int64      │  int64   │     int64      │
-- ├────────────────┼──────────┼────────────────┤
-- │  2199023255594 │        1 │             16 │
-- │  8796093022244 │        2 │  2199023255594 │
-- │ 10995116277761 │        2 │  2199023255594 │
-- │ 10995116277782 │        1 │             14 │
-- │ 13194139533342 │        2 │  2199023255594 │
-- │ 13194139533352 │        2 │  2199023255594 │
-- │ 13194139533355 │        3 │  8796093022244 │
-- │ 15393162788877 │        2 │  2199023255594 │
-- │ 17592186044443 │        3 │ 10995116277761 │
-- │ 17592186044461 │        2 │  2199023255594 │
-- │ 19791209299987 │        3 │ 13194139533352 │
-- │ 24189255811081 │        1 │             14 │
-- │ 24189255811109 │        2 │ 24189255811081 │
-- │ 26388279066641 │        2 │ 10995116277782 │
-- │ 26388279066655 │        1 │             16 │
-- │ 26388279066658 │        2 │  2199023255594 │
-- │ 26388279066668 │        1 │             14 │
-- │ 28587302322180 │        1 │             16 │
-- │ 28587302322196 │        2 │  2199023255594 │
-- │ 28587302322204 │        1 │             16 │
-- │ 28587302322223 │        2 │ 28587302322180 │
-- │ 30786325577731 │        2 │ 28587302322180 │
-- │ 30786325577740 │        2 │  2199023255594 │
-- │ 32985348833329 │        2 │ 28587302322180 │
-- │ 35184372088834 │        2 │ 24189255811081 │
-- │ 35184372088850 │        2 │  2199023255594 │
-- │ 35184372088856 │        2 │ 10995116277782 │
-- ├────────────────┴──────────┴────────────────┤
-- │ 27 rows                          3 columns │
-- └────────────────────────────────────────────┘


-- Test 5.4: Multi-source with three vertices
-- Pattern: Three starting points
CALL gpu_processing_graph("
    FROM GRAPH_TABLE (snb
        MATCH (p1:person WHERE p1.id IN (14, 16, 32))-[k:Person_knows_Person]->*(p2:person)
        COLUMNS (p2.id, distance)
    )
");
-- Expected: Maximum reachability from three sources
-- ┌────────────────┬──────────┐
-- │       id       │ distance │
-- │     int64      │  int64   │
-- ├────────────────┼──────────┤
-- │             14 │        0 │
-- │             16 │        0 │
-- │             32 │        0 │
-- │  2199023255594 │        1 │
-- │  8796093022244 │        2 │
-- │ 10995116277761 │        2 │
-- │ 10995116277782 │        1 │
-- │ 13194139533342 │        2 │
-- │ 13194139533352 │        1 │
-- │ 13194139533355 │        3 │
-- │ 15393162788877 │        2 │
-- │ 17592186044443 │        3 │
-- │ 17592186044461 │        1 │
-- │ 19791209299987 │        2 │
-- │ 24189255811081 │        1 │
-- │ 24189255811109 │        2 │
-- │ 26388279066641 │        2 │
-- │ 26388279066655 │        1 │
-- │ 26388279066658 │        1 │
-- │ 26388279066668 │        1 │
-- │ 28587302322180 │        1 │
-- │ 28587302322196 │        2 │
-- │ 28587302322204 │        1 │
-- │ 28587302322223 │        2 │
-- │ 30786325577731 │        2 │
-- │ 30786325577740 │        2 │
-- │ 32985348833329 │        2 │
-- │ 35184372088834 │        2 │
-- │ 35184372088850 │        2 │
-- │ 35184372088856 │        2 │
-- ├────────────────┴──────────┤
-- │ 30 rows         2 columns │
-- └───────────────────────────┘


-- Test 5.5: Multi-source ANY SHORTEST
-- Pattern: Shortest paths from multiple sources
CALL gpu_processing_graph("
    FROM GRAPH_TABLE (snb
         MATCH p = ANY SHORTEST (p1:person WHERE p1.id IN (14, 16))-[k:Person_knows_Person]->*(p2:person)
        COLUMNS (p2.id, distance)
    )
");
-- Expected: Shortest paths from either source (min distance)
-- ┌────────────────┬──────────┐
-- │       id       │ distance │
-- │     int64      │  int64   │
-- ├────────────────┼──────────┤
-- │  2199023255594 │        1 │
-- │  8796093022244 │        2 │
-- │ 10995116277761 │        2 │
-- │ 10995116277782 │        1 │
-- │ 13194139533342 │        2 │
-- │ 13194139533352 │        2 │
-- │ 13194139533355 │        3 │
-- │ 15393162788877 │        2 │
-- │ 17592186044443 │        3 │
-- │ 17592186044461 │        2 │
-- │ 19791209299987 │        3 │
-- │ 24189255811081 │        1 │
-- │ 24189255811109 │        2 │
-- │ 26388279066641 │        2 │
-- │ 26388279066655 │        1 │
-- │ 26388279066658 │        2 │
-- │ 26388279066668 │        1 │
-- │ 28587302322180 │        1 │
-- │ 28587302322196 │        2 │
-- │ 28587302322204 │        1 │
-- │ 28587302322223 │        2 │
-- │ 30786325577731 │        2 │
-- │ 30786325577740 │        2 │
-- │ 32985348833329 │        2 │
-- │ 35184372088834 │        2 │
-- │ 35184372088850 │        2 │
-- │ 35184372088856 │        2 │
-- ├────────────────┴──────────┤
-- │ 27 rows         2 columns │
-- └───────────────────────────┘


-- ============================================
-- SECTION 7: COMPARISON WITH DUCKPGQ
-- Side-by-side validation queries
-- ============================================

-- Test 7.1: GPU vs DuckPGQ - Edge Traversal
CALL gpu_processing_graph("
    FROM GRAPH_TABLE (snb
        MATCH (p1:person WHERE p1.id = 14)-[k:Person_knows_Person]->(p2:person)
        COLUMNS (p2.id, distance)
    )
");

-- DuckPGQ equivalent:
-- FROM GRAPH_TABLE (snb
--     MATCH (p1:person WHERE p1.id = 14)-[k:knows]->(p2:person)
--     COLUMNS (p2.id)
-- );


-- Test 7.2: GPU vs DuckPGQ - BFS zero-or-more
CALL gpu_processing_graph("
    FROM GRAPH_TABLE (snb
        MATCH (p1:person WHERE p1.id = 14)-[k:Person_knows_Person]->*(p2:person)
        COLUMNS (p2.id, distance)
    )
");

-- DuckPGQ equivalent:
-- FROM GRAPH_TABLE (snb
--     MATCH (p1:person WHERE p1.id = 14)-[k:knows]->*(p2:person)
--     COLUMNS (p2.id)
-- );


-- Test 7.3: GPU vs DuckPGQ - ANY SHORTEST
CALL gpu_processing_graph("
    FROM GRAPH_TABLE (snb
         MATCH p = ANY SHORTEST (p1:person WHERE p1.id = 14)-[k:Person_knows_Person]->*(p2:person)
        COLUMNS (p2.id, distance)
    )
");

-- DuckPGQ equivalent:
-- FROM GRAPH_TABLE (snb
--     MATCH p = ANY SHORTEST (p1:person WHERE p1.id = 14)-[k:knows]->*(p2:person)
--     COLUMNS (p2.id, path_length(p))
-- );