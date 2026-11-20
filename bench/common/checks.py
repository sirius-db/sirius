# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
"""Read-only storage and dataset checks before an AWS benchmark starts."""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

TABLES = (
    "region",
    "nation",
    "supplier",
    "customer",
    "part",
    "partsupp",
    "orders",
    "lineitem",
)


def command_json(*command):
    return json.loads(subprocess.check_output(command, text=True))


def ec2_identity():
    # Never infer the instance region from a bucket name or the AWS CLI profile.
    opener = build_opener(ProxyHandler({}))
    url = "http://169.254.169.254/latest/"
    request = Request(
        url + "api/token",
        method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
    )
    with opener.open(request, timeout=2) as response:
        token = response.read().decode()
    request = Request(
        url + "dynamic/instance-identity/document",
        headers={"X-aws-ec2-metadata-token": token},
    )
    with opener.open(request, timeout=2) as response:
        return json.load(response)


def verify_s3(source, region):
    import boto3

    uri = urlsplit(source)
    if uri.scheme != "s3" or not uri.netloc or uri.query or uri.fragment:
        raise ValueError("Use an s3://bucket/prefix URI without query parameters")
    client = boto3.client("s3", region_name=region)
    location = client.get_bucket_location(Bucket=uri.netloc)["LocationConstraint"]
    location = {None: "us-east-1", "EU": "eu-west-1"}.get(location, location)
    if location != region:
        raise ValueError(
            f"S3 bucket is in {location}; EC2 is in {region}. Cross-region reads are forbidden."
        )
    prefix = uri.path.lstrip("/").rstrip("/") + "/"
    for table in TABLES:
        pages = client.get_paginator("list_objects_v2").paginate(
            Bucket=uri.netloc, Prefix=prefix + table
        )
        if not any(
            item["Key"].endswith(".parquet")
            and item["Key"][len(prefix) :].startswith((table + "/", table + "."))
            for page in pages
            for item in page.get("Contents", [])
        ):
            raise ValueError(f"No Parquet input found for {table} in {source}")
    return {"bucket": uri.netloc, "region": location, "tables_present": list(TABLES)}


def physical_disks(device):
    disks = []
    for node in device:
        if node["type"] == "disk":
            disks.append(node)
        disks.extend(physical_disks(node.get("children", [])))
    return disks


def verify_spill(config, allow_ebs=False):
    disk = config["sirius"]["memory"]["disk"]
    paths = disk["downgrade_root_dirs"]
    paths = [paths] if isinstance(paths, str) else paths
    capacity = str(disk["capacity_bytes"])
    match = re.fullmatch(r"(\d+)([KMGT]i)?B?", capacity)
    if not match:
        raise ValueError(f"Unsupported spill capacity: {capacity}")
    required = (
        int(match[1])
        * {None: 1, "Ki": 1024, "Mi": 1024**2, "Gi": 1024**3, "Ti": 1024**4}[match[2]]
    )
    evidence = []
    for value in paths:
        path = Path(value).resolve(strict=True)
        if not path.is_dir() or not os.access(path, os.W_OK):
            raise ValueError(
                f"Spill directory must already exist and be writable: {path}"
            )
        mount = command_json(
            "findmnt", "--json", "--target", str(path), "--output", "SOURCE,TARGET"
        )["filesystems"][0]
        source = mount["source"].split("[", 1)[0]
        devices = command_json(
            "lsblk",
            "--json",
            "--inverse",
            "--paths",
            "--output",
            "NAME,TYPE,MODEL",
            source,
        )["blockdevices"]
        physical = physical_disks(devices)
        allowed = {"Amazon EC2 NVMe Instance Storage"}
        if allow_ebs:
            allowed.add("Amazon Elastic Block Store")
        if not physical or any(
            (d.get("model") or "").strip() not in allowed for d in physical
        ):
            raise ValueError(f"Spill path is not verified EC2 instance storage: {path}")
        free = shutil.disk_usage(path).free
        if free < required:
            raise ValueError(
                f"Spill path has {free} free bytes; config requires {required}: {path}"
            )
        evidence.append(
            {
                "path": str(path),
                "mount": mount,
                "devices": devices,
                "free_bytes": free,
                "allow_ebs": allow_ebs,
            }
        )
    return evidence


def preflight(hardware, source, config, allow_ebs=False):
    if hardware == "gb300":
        params = Path("/proc/driver/nvidia/params").read_text()
        if not re.search(r'^CoherentGPUMemoryMode:\s*"driver"', params, re.M):
            raise ValueError("GB300 replay requires CDMM (driver mode)")
    evidence = {}
    if hardware == "g7e":
        identity = ec2_identity()
        if identity["instanceType"] != "g7e.8xlarge":
            raise ValueError("The saved G7e presets require g7e.8xlarge")
        evidence["ec2"] = identity
        evidence["spill"] = verify_spill(config, allow_ebs)
    if source.startswith("s3://"):
        identity = evidence.get("ec2") or ec2_identity()
        evidence["dataset"] = verify_s3(source, identity["region"])
    else:
        root = Path(source)
        if not root.is_dir():
            raise ValueError(f"Dataset directory is missing: {root}")
        missing = [
            table
            for table in TABLES
            if not (root / table).is_dir()
            and not (root / (table + ".parquet")).is_file()
        ]
        if missing:
            raise ValueError(f"Missing table paths: {missing}")
    evidence["gpu"] = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv",
        ],
        text=True,
    )
    return evidence
