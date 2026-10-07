#!/usr/bin/env bash
# Set up the EC2 instance-store NVMe as Sirius scratch/spill space at /mnt/nvme,
# and make it come back automatically on every boot.
#
#   sudo bash bench/s3-sf1000/setup-nvme-scratch.sh
#
# Instance store is ephemeral: after a stop/start the disk comes back BLANK (and
# may enumerate as a different /dev/nvmeXn1). A plain fstab entry would then fail
# to mount -- or hang the boot without `nofail`. So instead of fstab this installs
# a oneshot systemd unit that, on every boot, finds the instance-store disk by its
# model id, creates an XFS filesystem only if the disk has none, and mounts it.
# A reboot (not stop/start) keeps the data; the unit just remounts it.
#
# Safe to re-run: it never reformats a disk that already has a filesystem.
set -euo pipefail

MOUNT=/mnt/nvme
OWNER="${SUDO_USER:-ec2-user}"
HELPER=/usr/local/sbin/sirius-nvme-scratch
UNIT=/etc/systemd/system/sirius-nvme-scratch.service

[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }

cat > "$HELPER" <<EOF
#!/usr/bin/env bash
# Installed by sirius bench/s3-sf1000/setup-nvme-scratch.sh -- see that file.
set -euo pipefail
# By model id, not /dev/nvmeXn1: the index can change across a stop/start.
dev=\$(for l in /dev/disk/by-id/nvme-Amazon_EC2_NVMe_Instance_Storage_*; do
         [ -e "\$l" ] && readlink -f "\$l"
       done | grep -E 'nvme[0-9]+n1\$' | sort -u | head -1 || true)
if [ -z "\$dev" ]; then
  echo "sirius-nvme-scratch: no instance-store NVMe found; nothing to do"
  exit 0
fi
if mountpoint -q $MOUNT; then
  echo "sirius-nvme-scratch: $MOUNT already mounted"
else
  if ! blkid -p "\$dev" >/dev/null 2>&1; then
    echo "sirius-nvme-scratch: \$dev is blank, creating XFS"
    # XFS labels are at most 12 characters.
    mkfs.xfs -f -L sirius-scr "\$dev"
  fi
  mkdir -p $MOUNT
  mount -o noatime "\$dev" $MOUNT
fi
mkdir -p $MOUNT/sirius_spill
chown $OWNER:$OWNER $MOUNT $MOUNT/sirius_spill
echo "sirius-nvme-scratch: \$dev mounted at $MOUNT"
EOF
chmod 755 "$HELPER"

cat > "$UNIT" <<EOF
[Unit]
Description=Sirius scratch: format-if-blank and mount the EC2 instance-store NVMe at $MOUNT
After=local-fs.target
Wants=local-fs.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=$HELPER

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now sirius-nvme-scratch.service
systemctl --no-pager status sirius-nvme-scratch.service | tail -5
df -h "$MOUNT"
