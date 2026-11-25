#!/bin/bash
set -u
root=/home/ec2-user/m9gd24-duckdb200-sf1k-pairs-20261124
cd "$root"
export AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1 PYTHONUNBUFFERED=1
unset AWS_PROFILE AWS_DEFAULT_PROFILE
python=/home/ec2-user/duckdb200-venv/bin/python
"$python" -m pip freeze > environment.txt
"$python" preflight.py > preflight.log 2>&1
preflight_code=$?
printf '%s\n' "$preflight_code" > preflight-exit-code.txt
if [ "$preflight_code" -eq 0 ]; then
 "$python" run.py > campaign.log 2>&1
 printf '%s\n' "$?" > benchmark-exit-code.txt
fi
"$python" archive.py > packaging.log 2>&1
printf '%s\n' "$?" > finish-exit-code.txt
