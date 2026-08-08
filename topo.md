 nvidia-smi topo -m
        GPU0    GPU1    GPU2    GPU3    NIC0    NIC1    NIC2    NIC3    NIC4    NIC5    NIC6    NIC7    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      NV18    NV18    NV18    SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     0-71    0,3-9,11-17             2
GPU1    NV18     X      NV18    NV18    SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     0-71    0,3-9,11-17             10
GPU2    NV18    NV18     X      NV18    SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     72-143  1,19-25,27-33           18
GPU3    NV18    NV18    NV18     X      SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     72-143  1,19-25,27-33           26
NIC0    SYS     SYS     SYS     SYS      X      SYS     SYS     SYS     SYS     SYS     SYS     SYS
NIC1    SYS     SYS     SYS     SYS     SYS      X      SYS     SYS     SYS     SYS     SYS     SYS
NIC2    SYS     SYS     SYS     SYS     SYS     SYS      X      PIX     SYS     SYS     SYS     SYS
NIC3    SYS     SYS     SYS     SYS     SYS     SYS     PIX      X      SYS     SYS     SYS     SYS
NIC4    SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS      X      SYS     SYS     SYS
NIC5    SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS      X      SYS     SYS
NIC6    SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS      X      PIX
NIC7    SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     PIX      X

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_0
  NIC1: mlx5_1
  NIC2: mlx5_2
  NIC3: mlx5_3
  NIC4: mlx5_4
  NIC5: mlx5_5
  NIC6: mlx5_6
  NIC7: mlx5_7