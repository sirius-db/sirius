# shellcheck shell=bash
# Source (don't execute). IPv4 helpers for the launch scripts:
#   sirius_pick_address <networks>  "<interface> <address>" for this host's first address inside
#       one of <networks>, CIDRs separated by ';' or ',' (StarRocks' priority_networks), e.g.
#       "10.87.140.0/24;192.168.0.0/16"
#   sirius_default_address          "<interface> <address>" for the default route's interface
#   sirius_address_network <addr>   the CIDR of this host's interface holding <addr>, e.g.
#       10.87.140.32/27 for 10.87.140.53 on bond0
# Each fails if there is no such address.

_sirius_ip_int()
{
  local IFS=.
  # shellcheck disable=SC2086
  set -- $1
  echo $((($1 << 24) | ($2 << 16) | ($3 << 8) | $4))
}

_sirius_int_ip()
{
  echo "$(($1 >> 24 & 255)).$(($1 >> 16 & 255)).$(($1 >> 8 & 255)).$(($1 & 255))"
}

_sirius_mask()
{
  echo $(((0xffffffff << (32 - $1)) & 0xffffffff))
}

sirius_pick_address()
{
  local cidr net bits mask dev addr ip
  for cidr in ${1//[;,]/ }; do
    net=${cidr%/*}
    bits=32
    [[ $cidr == */* ]] && bits=${cidr#*/}
    mask=$(_sirius_mask "$bits")
    while read -r _ dev _ addr _; do
      ip=${addr%/*}
      if (((($(_sirius_ip_int "$ip") ^ $(_sirius_ip_int "$net")) & mask) == 0)); then
        echo "$dev $ip"
        return 0
      fi
    done < <(ip -o -4 addr show)
  done
  return 1
}

sirius_default_address()
{
  local dev addr
  dev=$(ip -o -4 route show default | awk '{ for (i = 1; i < NF; i++) if ($i == "dev") { print $(i + 1); exit } }')
  [ -n "$dev" ] || return 1
  addr=$(ip -o -4 addr show dev "$dev" | awk '{ print $4; exit }')
  [ -n "$addr" ] || return 1
  echo "$dev ${addr%/*}"
}

sirius_address_network()
{
  local addr bits
  while read -r _ _ _ addr _; do
    [ "${addr%/*}" = "$1" ] || continue
    bits=${addr#*/}
    echo "$(_sirius_int_ip $(($(_sirius_ip_int "$1") & $(_sirius_mask "$bits"))))/$bits"
    return 0
  done < <(ip -o -4 addr show)
  return 1
}
