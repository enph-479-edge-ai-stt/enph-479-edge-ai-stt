# Create the Vivado project and block design for the KV260: the Zynq PS, an AXI DMA and
# sr_accelerator_axi (shared/specs/fabric_interface.md), wired as PYNQ expects.
#
#   vivado -mode batch -source create_project.tcl -tclargs <build dir> [PE_COUNT]
#
# The build dir must be outside OneDrive (sync locks Vivado's files mid-run). PE_COUNT
# defaults to the real 256; a small one (e.g. 16) builds in minutes to test the flow.
# Then build_bitstream.tcl makes the .bit and .hwh.

if {[llength $argv] < 1} {
    error "usage: vivado -mode batch -source create_project.tcl -tclargs <build dir> \[PE_COUNT\]"
}
set build_dir [file normalize [lindex $argv 0]]
set pe_count [expr {[llength $argv] > 1 ? [lindex $argv 1] : 256}]
set rtl_dir [file normalize [file join [file dirname [info script]] .. rtl]]

# The KV260's board files come from AMD's board store, once per machine.
if {[llength [get_board_parts -quiet xilinx.com:kv260_som:*]] == 0} {
    xhub::refresh_catalog [xhub::get_xstores xilinx_board_store]
    xhub::install [xhub::get_xitems xilinx.com:xilinx_board_store:kv260_som:*]
}
set board_part [lindex [lsort [get_board_parts xilinx.com:kv260_som:*]] end]

create_project sr_accel $build_dir -part xck26-sfvc784-2LV-c -force
set_property board_part $board_part [current_project]
add_files [glob $rtl_dir/*.v]

create_bd_design sr_accel

# Zynq PS with the KV260 preset. HPM0_FPD carries the register writes, HP0_FPD the DMA's
# traffic to DDR, pl_clk0 clocks the fabric at 100 MHz.
set ps [create_bd_cell -type ip -vlnv [get_ipdefs xilinx.com:ip:zynq_ultra_ps_e:*] zynq_ultra_ps_e_0]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset 1} $ps
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 1 \
    CONFIG.PSU__USE__M_AXI_GP1 0 \
    CONFIG.PSU__USE__M_AXI_GP2 0 \
    CONFIG.PSU__USE__S_AXI_GP2 1 \
    CONFIG.PSU__FPGA_PL0_ENABLE 1 \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ 100 \
] $ps

# Simple-mode AXI DMA (no scatter-gather), 32-bit streams. The buffer length register is
# 26 bits: the default 14 truncates a transfer at 16 KB and a weight load is ~1.2 MB.
set dma [create_bd_cell -type ip -vlnv [get_ipdefs xilinx.com:ip:axi_dma:*] axi_dma_0]
set_property -dict [list \
    CONFIG.c_include_sg 0 \
    CONFIG.c_sg_length_width 26 \
    CONFIG.c_include_mm2s 1 \
    CONFIG.c_include_s2mm 1 \
    CONFIG.c_m_axis_mm2s_tdata_width 32 \
    CONFIG.c_s_axis_s2mm_tdata_width 32 \
] $dma

set accel [create_bd_cell -type module -reference sr_accelerator_axi accel]
set_property CONFIG.PE_COUNT $pe_count $accel

# Registers: PS HPM0_FPD -> the DMA's control port and the accelerator's s_axil.
# Automation adds the interconnect, pl_clk0 on every AXI clock and a reset block.
foreach slave {/axi_dma_0/S_AXI_LITE /accel/s_axil} {
    apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list \
        Master /zynq_ultra_ps_e_0/M_AXI_HPM0_FPD Slave $slave \
        Clk_master Auto Clk_slave Auto Clk_xbar Auto intc_ip Auto master_apm 0] [get_bd_intf_pins $slave]
}
# Data: the DMA's two memory masters -> one SmartConnect -> PS HP0_FPD (DDR), on the
# clock and reset the register automation set up.
set smc [create_bd_cell -type ip -vlnv [get_ipdefs xilinx.com:ip:smartconnect:*] axi_smc_dma]
set_property -dict [list CONFIG.NUM_SI 2 CONFIG.NUM_MI 1] $smc
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_smc_dma/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_smc_dma/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_smc_dma/M00_AXI] [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP0_FPD]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins axi_smc_dma/aclk] \
    [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] \
    [get_bd_pins zynq_ultra_ps_e_0/saxihp0_fpd_aclk]
set rst [get_bd_cells -filter {VLNV =~ "*proc_sys_reset*"}]
connect_bd_net [get_bd_pins $rst/peripheral_aresetn] [get_bd_pins axi_smc_dma/aresetn]
# Streams: DMA MM2S -> weights and frames in; scores out -> DMA S2MM.
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] [get_bd_intf_pins accel/s_axis]
connect_bd_intf_net [get_bd_intf_pins accel/m_axis] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]

assign_bd_address
validate_bd_design
save_bd_design

set wrapper [make_wrapper -files [get_files sr_accel.bd] -top]
add_files -norecurse $wrapper
set_property top sr_accel_wrapper [current_fileset]
update_compile_order -fileset sources_1

puts "Created $build_dir/sr_accel.xpr (PE_COUNT $pe_count, board $board_part)"
