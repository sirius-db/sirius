CREATE TABLE T (A BIGINT);
INSERT INTO T VALUES (1), (NULL), (3), (NULL), (5);
call gpu_buffer_init("1 GB", "1 GB");
call gpu_processing("select * from T");