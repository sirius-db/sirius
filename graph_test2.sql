-- ============================================
-- Graph Query Tests for Sirius
-- ============================================

-- Initialize GPU buffer
CALL gpu_buffer_init('1GB', '1GB');

-- Create test graph data
DROP TABLE IF EXISTS Person;
DROP TABLE IF EXISTS Knows;

CREATE TABLE Person_w (id BIGINT, name VARCHAR);
INSERT INTO Person_w VALUES
                       (14, 'Alice'),
                       (25, 'Bob'),
                       (37, 'Charlie'),
                       (42, 'Dave');

CREATE TABLE Knows_w (src BIGINT, dst BIGINT, weight DOUBLE);
INSERT INTO Knows_w VALUES
                      (14, 25, 1.0),
                      (14, 37, 2.5),
                      (25, 37, 1.5),
                      (37, 42, 3.0),
                      (14, 42, 10.0);

-- Warm up GPU cache
CALL gpu_processing("SELECT * FROM Person_w");
CALL gpu_processing("SELECT * FROM Knows_w");

-- Test weighted query
CALL gpu_processing_graph("
  SELECT * FROM GRAPH_TABLE (social
    MATCH SHORTEST (p:Person_w WHERE p.id=14)-[w:Knows_w]->+(p2:Person_w WHERE p2.id=42)
    COLUMNS (p2.id, distance)
  )
");

CALL gpu_processing_graph("
  SELECT * FROM GRAPH_TABLE (social
    MATCH SHORTEST (p:Person_w WHERE p.id=14)-[:Knows_w]->+(p2:Person_w WHERE p2.id=42)
    COLUMNS (p2.id)
  )
");