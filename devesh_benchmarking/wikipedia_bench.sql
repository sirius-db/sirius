DROP TABLE IF EXISTS SPAM;

CREATE TABLE SPAM (
  P_VAL VARCHAR(256)
);

COPY SPAM from '/mnt/wiscdb/abigale/string_dataset_csvs/wikipedia_dataset.csv';

call gpu_buffer_init("17 GB", "17 GB");

.timer on
--SELECT * from SPAM WHERE P_VAL LIKE '%probabilistic%';
--SELECT * from SPAM WHERE contains(P_VAL, 'probabilistic');

--call gpu_processing("SELECT * from SPAM WHERE P_VAL LIKE '%probabilistic%'");
--call gpu_processing("SELECT * from SPAM WHERE P_VAL LIKE '%probabilistic%'");
--call gpu_processing("SELECT * from SPAM WHERE P_VAL LIKE '%probabilistic%'");
--call gpu_processing("SELECT * from SPAM WHERE P_VAL LIKE '%probabilistic%'");
--call gpu_processing("SELECT * from SPAM WHERE P_VAL LIKE '%probabilistic%'");
--call gpu_processing("SELECT * from SPAM WHERE P_VAL LIKE '%probabilistic%'");
--call gpu_processing("SELECT * from SPAM WHERE P_VAL LIKE '%probabilistic%'");
--call gpu_processing("SELECT * from SPAM WHERE P_VAL LIKE '%probabilistic%'");
--call gpu_processing("SELECT * from SPAM WHERE P_VAL LIKE '%probabilistic%'");
--call gpu_processing("SELECT * from SPAM WHERE P_VAL LIKE '%probabilistic%'");
--call gpu_processing("SELECT * from SPAM WHERE P_VAL LIKE '%probabilistic%'");
--call gpu_processing("SELECT * from SPAM WHERE P_VAL LIKE '%probabilistic%'");

--call gpu_processing("SELECT * from SPAM WHERE contains(P_VAL, 'probabilistic')");
call gpu_processing("SELECT * from SPAM WHERE contains(P_VAL, 'probabilistic')");
call gpu_processing("SELECT * from SPAM WHERE contains(P_VAL, 'probabilistic')");
call gpu_processing("SELECT * from SPAM WHERE contains(P_VAL, 'probabilistic')");
call gpu_processing("SELECT * from SPAM WHERE contains(P_VAL, 'probabilistic')");
call gpu_processing("SELECT * from SPAM WHERE contains(P_VAL, 'probabilistic')");
call gpu_processing("SELECT * from SPAM WHERE contains(P_VAL, 'probabilistic')");
call gpu_processing("SELECT * from SPAM WHERE contains(P_VAL, 'probabilistic')");
call gpu_processing("SELECT * from SPAM WHERE contains(P_VAL, 'probabilistic')");
call gpu_processing("SELECT * from SPAM WHERE contains(P_VAL, 'probabilistic')");
call gpu_processing("SELECT * from SPAM WHERE contains(P_VAL, 'probabilistic')");
call gpu_processing("SELECT * from SPAM WHERE contains(P_VAL, 'probabilistic')");
