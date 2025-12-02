-- ============================================
-- Graph Query Tests for Sirius
-- ============================================

-- Initialize GPU buffer
CALL gpu_buffer_init('1GB', '1GB');

-- Create test graph data
DROP TABLE IF EXISTS Person;
DROP TABLE IF EXISTS Knows;

CREATE TABLE Person (id BIGINT, name VARCHAR);
INSERT INTO Person VALUES
                       (14, 'Alice'),
                       (25, 'Bob'),
                       (37, 'Charlie'),
                       (42, 'Dave');

CREATE TABLE Knows (src BIGINT, dst BIGINT);
INSERT INTO Knows VALUES
                      (14, 25),
                      (14, 37),
                      (25, 37),
                      (37, 42),
                      (14, 42);

-- Warm up GPU cache
CALL gpu_processing("SELECT * FROM Person");
CALL gpu_processing("SELECT * FROM Knows");

-- Register graph metadata, currently DOES NOT work
-- SELECT sirius_register_graph('social', 'Person', 'Knows', 'src', 'dst', NULL);

-- ============================================
-- Test 1: Edge Traversal (->)
-- ============================================
CALL gpu_processing_graph("
  SELECT * FROM GRAPH_TABLE (social
    MATCH (p:Person WHERE p.id=14)-[:knows]->(p2:Person)
    COLUMNS (p2.id)
  )
");

CALL gpu_processing_graph("
  SELECT * FROM GRAPH_TABLE (social
    MATCH (p:Person WHERE p.id=14)-[:knows]->(p2:Person)
    COLUMNS (p2.id, distance)
  )
");

CALL gpu_processing_graph("
  SELECT * FROM GRAPH_TABLE (social
    MATCH (p:Person WHERE p.id=14)-[:knows]->(p2:Person)
    COLUMNS (p2.id, distance, predecessor)
  )
");

-- ============================================
-- Test 2: BFS (Zero or more, ->*)
-- ============================================
CALL gpu_processing_graph("
  SELECT * FROM GRAPH_TABLE (social
    MATCH (p:Person WHERE p.id=14)-[:knows]->*(p2:Person)
    COLUMNS (p2.id)
  )
");

CALL gpu_processing_graph("
  SELECT * FROM GRAPH_TABLE (social
    MATCH (p:Person WHERE p.id=14)-[:knows]->*(p2:Person)
    COLUMNS (p2.id, distance)
  )
");

CALL gpu_processing_graph("
  SELECT * FROM GRAPH_TABLE (social
    MATCH (p:Person WHERE p.id=14)-[:knows]->*(p2:Person)
    COLUMNS (p2.id, distance, predecessor)
  )
");

-- ============================================
-- Test 3: BFS (One or more, ->+)
-- ============================================
CALL gpu_processing_graph("
  SELECT * FROM GRAPH_TABLE (social
    MATCH (p:Person WHERE p.id=14)-[:knows]->+(p2:Person)
    COLUMNS (p2.id)
  )
");

CALL gpu_processing_graph("
  SELECT * FROM GRAPH_TABLE (social
    MATCH (p:Person WHERE p.id=14)-[:knows]->+(p2:Person)
    COLUMNS (p2.id, distance)
  )
");

CALL gpu_processing_graph("
  SELECT * FROM GRAPH_TABLE (social
    MATCH (p:Person WHERE p.id=14)-[:knows]->+(p2:Person)
    COLUMNS (p2.id, distance, predecessor)
  )
");

-- ============================================
-- Test 4: BFS from different source
-- ============================================
CALL gpu_processing_graph("
  SELECT * FROM GRAPH_TABLE (social
    MATCH (p:Person WHERE p.id=25)-[:knows]->*(p2:Person)
    COLUMNS (p2.id)
  )
");

CALL gpu_processing_graph("
  SELECT * FROM GRAPH_TABLE (social
    MATCH (p:Person WHERE p.id=25)-[:knows]->*(p2:Person)
    COLUMNS (p2.id, distance, predecessor)
  )
");

-- ============================================
-- Test 5: Source with no outgoing edges
-- ============================================
CALL gpu_processing_graph("
  SELECT * FROM GRAPH_TABLE (social
    MATCH (p:Person WHERE p.id=42)-[:knows]->(p2:Person)
    COLUMNS (p2.id)
  )
");

-- TODO: Multi-source BFS
CALL gpu_processing_graph("
  SELECT * FROM GRAPH_TABLE (social
    MATCH (p:Person WHERE p.id IN (14, 25))-[:knows]->*(p2:Person)
    COLUMNS (p2.id, distance)
  )
");

-- ============================================
-- Future Tests (once features are implemented)
-- ============================================

-- TODO: Test 5: Weighted Shortest Path
CALL gpu_processing_graph("
  SELECT * FROM GRAPH_TABLE (social
    MATCH SHORTEST (p:Person WHERE p.id=14)-[:knows]->+(p2:Person WHERE p2.id=42)
    COLUMNS (p2.id)
  )
");

-- TODO: Test 6: Path reconstruction with predecessors
CALL gpu_processing_graph("
  SELECT * FROM GRAPH_TABLE (social
    MATCH p = (p1:Person WHERE p1.id=14)-[:knows]->*(p2:Person)
    COLUMNS (p2.id, length(p))
  )
");