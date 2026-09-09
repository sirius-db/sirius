import os
# Zone-map index cost per column, as a fraction of RAW and of COMPRESSED (2.84x, measured) bytes
RAW_COMP_RATIO=264/93
print(f"{'chunk rows':>11}{'per-row idx B (4B col)':>24}{'% of raw':>10}{'% of compressed':>17}"
      f"{'   |  per-row idx B (8B col)':>28}{'% raw':>8}{'% comp':>8}")
for cs in (1024,2048,4096,8192,16384,32768,65536,262144,1048576):
    for w,label in ((4,'4B'),(8,'8B')):
        pass
    r4=(2*4+4)/cs; r8=(2*8+4)/cs
    print(f"{cs:>11}{r4:>24.4f}{100*r4/4:>10.3f}{100*r4/4*RAW_COMP_RATIO:>17.3f}"
          f"{r8:>28.4f}{100*r8/8:>8.3f}{100*r8/8*RAW_COMP_RATIO:>8.3f}")

print("\nWhole-table sidecar, TPC-H SF1000 lineitem (6.0e9 rows), min+max+nullcount:")
W={'l_orderkey':8,'l_partkey':4,'l_suppkey':4,'l_linenumber':4,'l_quantity':8,'l_extendedprice':8,
   'l_discount':8,'l_tax':8,'l_returnflag':4,'l_linestatus':4,'l_shipdate':4,'l_commitdate':4,
   'l_receiptdate':4,'l_shipinstruct':4,'l_shipmode':4,'l_comment':4}
FILTERABLE=['l_shipdate','l_commitdate','l_receiptdate','l_quantity','l_discount','l_returnflag',
            'l_shipmode','l_shipinstruct','l_orderkey','l_partkey']
N=6_000_000_000
for scope,cols in (('all 16 cols',list(W)),('10 filterable cols',FILTERABLE),('3 date cols',['l_shipdate','l_commitdate','l_receiptdate'])):
    per=sum(2*W[c]+4 for c in cols)
    print(f"  {scope:<20}"+ "".join(f"{cs//1024}K:{per*(N//cs)/1e9:6.2f}GB  " for cs in (1024,8192,65536,1048576)))
print("  (compare: 93 GB compressed / 264 GB raw for the 7 hottest lineitem columns)")
