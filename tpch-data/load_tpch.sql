COPY lineitem FROM '/home/ubuntu/Camelot/tpch-data/dbgen/s1/lineitem.tbl' WITH (HEADER false, DELIMITER '|');
COPY orders FROM '/home/ubuntu/Camelot/tpch-data/dbgen/s1/orders.tbl' WITH (HEADER false, DELIMITER '|');
COPY supplier FROM '/home/ubuntu/Camelot/tpch-data/dbgen/s1/supplier.tbl' WITH (HEADER false, DELIMITER '|');
COPY part FROM '/home/ubuntu/Camelot/tpch-data/dbgen/s1/part.tbl' WITH (HEADER false, DELIMITER '|');
COPY customer FROM '/home/ubuntu/Camelot/tpch-data/dbgen/s1/customer.tbl' WITH (HEADER false, DELIMITER '|');
COPY partsupp FROM '/home/ubuntu/Camelot/tpch-data/dbgen/s1/partsupp.tbl' WITH (HEADER false, DELIMITER '|');
COPY nation FROM '/home/ubuntu/Camelot/tpch-data/dbgen/s1/nation.tbl' WITH (HEADER false, DELIMITER '|');
COPY region FROM '/home/ubuntu/Camelot/tpch-data/dbgen/s1/region.tbl' WITH (HEADER false, DELIMITER '|');
