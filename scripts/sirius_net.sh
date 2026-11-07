# Source (don't execute): sirius_pick_address <networks> prints "<interface> <IPv4 address>" for the
# first address of this host inside one of <networks>, CIDRs separated by ';' or ',' (StarRocks'
# priority_networks), e.g. "10.87.140.0/24;192.168.0.0/16". Fails if none is.

_sirius_ip_int()
{
  local IFS=.
  # shellcheck disable=SC2086
  set -- $1
  echo $((($1 << 24) | ($2 << 16) | ($3 << 8) | $4))
}

sirius_pick_address()
{
  local cidr net bits mask dev addr ip
  for cidr in ${1//[;,]/ }; do
    net=${cidr%/*}
    bits=32
    [[ $cidr == */* ]] && bits=${cidr#*/}
    mask=$(((0xffffffff << (32 - bits)) & 0xffffffff))
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
