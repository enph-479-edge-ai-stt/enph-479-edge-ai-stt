# Synthesize, implement and write the bitstream for a project made by create_project.tcl,
# then copy the two files PYNQ needs into <build dir>/out with the same base name.
#
#   vivado -mode batch -source build_bitstream.tcl -tclargs <build dir> [design name]
#
# The design name defaults to sr_accel and must match the name passed to
# create_project.tcl. The copied files are <build dir>/out/<design name>.bit
# and <design name>.hwh.

if {[llength $argv] < 1} {
    error "usage: vivado -mode batch -source build_bitstream.tcl -tclargs <build dir> \[design name\]"
}
set build_dir [file normalize [lindex $argv 0]]
set design_name sr_accel
if {[llength $argv] > 1} {
    set design_name [lindex $argv 1]
}
set out_dir [file join $build_dir out]

open_project [file join $build_dir ${design_name}.xpr]
reset_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
    error "impl_1 failed: see [get_property DIRECTORY [get_runs impl_1]]/runme.log"
}

file mkdir $out_dir
open_run impl_1
report_utilization -file [file join $out_dir utilization.rpt]
report_timing_summary -file [file join $out_dir timing.rpt]
file copy -force [file join [get_property DIRECTORY [get_runs impl_1]] ${design_name}_wrapper.bit] \
    [file join $out_dir ${design_name}.bit]
file copy -force [file join $build_dir ${design_name}.gen sources_1 bd $design_name hw_handoff ${design_name}.hwh] \
    [file join $out_dir ${design_name}.hwh]

# A negative worst slack means the design may not run at 100 MHz. The bitstream is still
# written, so say so loudly instead of failing.
set wns [get_property STATS.WNS [get_runs impl_1]]
puts "Wrote $out_dir/${design_name}.bit and ${design_name}.hwh (worst negative slack $wns ns)"
if {$wns < 0} {
    puts "WARNING: timing not met at 100 MHz, see $out_dir/timing.rpt"
}
