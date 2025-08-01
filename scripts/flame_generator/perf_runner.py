import subprocess
import sys
import signal
import time
import argparse
import tempfile
import os

def load_args():
    parser = argparse.ArgumentParser(description="Attach perf to a running process and save the profile data.")
    parser.add_argument(
        "--pid",
        type=int,
        help="Process ID to attach perf to"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="Path to save the resulting flame graph"
    )

    return parser.parse_args()

def create_temp_file(suffix):
    temp_file = tempfile.NamedTemporaryFile(delete=True, suffix=suffix)
    temp_file.close()
    return temp_file.name

def main(args):

    # Start the profiler when specified
    raw_save_path = create_temp_file(".raw")
    profiler_command = f"perf record -g -p {args.pid} -o {raw_save_path}"
    input("Enter to start the profiler ....")
    perf_proc = subprocess.Popen(profiler_command.split(" "))
    print("Started profiler")
    input("Enter again to stop the profiler ...")
    perf_proc.send_signal(signal.SIGINT)
    perf_proc.wait()

    perf_save_path = create_temp_file(".perf")
    perf_save_command = f"perf script -i {raw_save_path} > {perf_save_path}"
    subprocess.run(perf_save_command, shell = True)
    print("Saved perf result to", perf_save_path)

    # Now fold stack samples
    curr_file_dir = os.path.dirname(os.path.abspath(__file__))
    fold_exec_path = os.path.join(curr_file_dir, "stackcollapse-perf.pl")
    folded_save_path = create_temp_file(".folded")
    fold_command_to_run = f"{fold_exec_path} {perf_save_path} > {folded_save_path}"
    subprocess.run(fold_command_to_run, shell = True)
    print("Saved the folded result to", folded_save_path)
    
    # Finall generate the flame graph
    flame_exec_path = os.path.join(curr_file_dir, "flamegraph.pl")
    flame_comand_to_run = f"{flame_exec_path} {folded_save_path} > {args.save_path}"
    subprocess.run(flame_comand_to_run, shell = True)
    print("Saved the flame graph to", args.save_path)

if __name__ == "__main__":
    main(load_args())