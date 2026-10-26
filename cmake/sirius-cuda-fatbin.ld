/* Keep large GPU images after host data to avoid relative-address overflows. */
SECTIONS
{
  .nv_fatbin : { *(.nv_fatbin) }
  __nv_relfatbin : { *(__nv_relfatbin) }
}
INSERT AFTER .bss;
