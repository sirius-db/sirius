-- test1.sql

-- Delete all data from the tables
DELETE FROM Person;
DELETE FROM Person_w;
DELETE FROM Knows;
DELETE FROM Knows_w;

-- Or drop and recreate the tables
DROP TABLE IF EXISTS Person;
DROP TABLE IF EXISTS Person_w;
DROP TABLE IF EXISTS Knows;
DROP TABLE IF EXISTS Knows_w;

CREATE TABLE Person (id BIGINT, name VARCHAR);
INSERT INTO Person VALUES (14, 'Alice'),
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