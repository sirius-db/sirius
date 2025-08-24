import os
import duckdb

QUERIES_TO_TEST = [0, 42]
WRITE_QUERY_TEST = True
WRITE_QUERY_RESULT = True


def write_query_test(query_id, query_text, query_writer):
    # First write the title
    query_writer.write(f"# Q{query_id}\nquery I\n")

    # Now actually write the query
    query_writer.write(f'call gpu_processing("{query_text}");\n')

    # Finally write how the result should be checked
    query_writer.write(f"----\n<FILE>:test/answers/clickbench/q{query_id}.csv")
    query_writer.write('\n\n')

def write_query_result(db_conn, query_id, query_text):
    save_path = f"test/answers/clickbench/q{query_id}.csv"
    actual_query = query_text.replace(";", "").strip()
    query_to_run = f"COPY ({actual_query}) TO '{save_path}' (HEADER, DELIMITER '|');"
    db_conn.sql(query_to_run)

def main():
    # Load all of the queries
    with open('clickbench_queries.sql', 'r') as reader:
        clickbench_queries = reader.readlines()
    print(len(clickbench_queries))
    
    # Create the duckdb connection
    db_conn = duckdb.connect('/home/devesh/datasets/clickbench/test_clickbench.duckdb')

    # Now write all of the test cases
    with open('sql/clickbench_queries.test', 'a+') as query_test_writer:
        query_test_writer.write('\n')
        for query_idx in range(QUERIES_TO_TEST[0], QUERIES_TO_TEST[1] + 1):
            print("Processing query at idx", query_idx)
            curr_query = clickbench_queries[query_idx].strip()

            if WRITE_QUERY_TEST:
                write_query_test(query_idx + 1, curr_query, query_test_writer)

            if WRITE_QUERY_RESULT:
                write_query_result(db_conn, query_idx + 1, curr_query)

if __name__ == '__main__':
    main()