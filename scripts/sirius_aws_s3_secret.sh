#!/usr/bin/env bash
# Writes a CREATE SECRET statement for Sirius from this instance's role credentials to
# /dev/shm/s3secret.sql (RAM, mode 600), for sirius_aws_run.sh --secret-file.
#   sirius_aws_s3_secret.sh BUCKET
# Sirius accepts only static keys (docs/super-sirius/scan.md), so the role's temporary keys are
# copied in: they expire after a few hours, and the date is printed. Write the file again then.
# See docs/super-sirius/distributed-experiments-aws.md.
set -euo pipefail
case ${1:-} in
  '' | -h | --help)
    sed -n '2,7s/^# \{0,1\}//p' "$0"
    exit 2
    ;;
esac
umask 077
imds=http://169.254.169.254/latest
token=$(curl -s -X PUT "$imds/api/token" -H 'X-aws-ec2-metadata-token-ttl-seconds: 300')
region=$(curl -s -H "X-aws-ec2-metadata-token: $token" "$imds/meta-data/placement/region")
role=$(curl -s -H "X-aws-ec2-metadata-token: $token" "$imds/meta-data/iam/security-credentials/")
[ -n "$role" ] || {
  echo "this instance has no role: attach an instance profile with access to the bucket" >&2
  exit 1
}
curl -s -H "X-aws-ec2-metadata-token: $token" "$imds/meta-data/iam/security-credentials/$role" |
  python3 -c '
import json, sys

c = json.load(sys.stdin)
print(
    "CREATE OR REPLACE SECRET bench_s3 (TYPE SIRIUS_S3, PROVIDER CONFIG, SCOPE \x27s3://%s/\x27, "
    "KEY_ID \x27%s\x27, SECRET \x27%s\x27, SESSION_TOKEN \x27%s\x27, REGION \x27%s\x27);"
    % (sys.argv[1], c["AccessKeyId"], c["SecretAccessKey"], c["Token"], sys.argv[2])
)
print("the keys expire " + c["Expiration"], file=sys.stderr)
' "$1" "$region" > /dev/shm/s3secret.sql
echo "wrote /dev/shm/s3secret.sql"
