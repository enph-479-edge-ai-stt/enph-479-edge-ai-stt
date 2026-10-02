`timescale 1ns / 1ps
 
module tb_sigmoid_lut;
 
    reg  [3:0] index;
    wire [7:0] sigmoid_output;
 
    integer errors = 0;
 
    // expected values, mirrored from the LUT table
    reg [7:0] expected [0:15];
 
    sigmoid_lut dut (
        .index          (index),
        .sigmoid_output (sigmoid_output)
    );
 
    // one check: drive an index, wait for combinational settle, compare
    task check(input [3:0] idx);
        begin
            index = idx;
            #1;
            if (sigmoid_output !== expected[idx]) begin
                $display("FAIL: index=%0d  got=%0d  expected=%0d",
                         idx, sigmoid_output, expected[idx]);
                errors = errors + 1;
            end else begin
                $display("pass: index=%0d  out=%0d", idx, sigmoid_output);
            end
        end
    endtask
 
    integer i;
    initial begin
        $dumpfile("tb_sigmoid_lut.vcd");
        $dumpvars(0, tb_sigmoid_lut);

        expected[0]=0;   expected[1]=0;   expected[2]=1;   expected[3]=2;
        expected[4]=5;   expected[5]=12;  expected[6]=30;  expected[7]=69;
        expected[8]=128; expected[9]=186; expected[10]=225; expected[11]=243;
        expected[12]=250; expected[13]=253; expected[14]=254; expected[15]=255;
 
        $display("--- sigmoid_lut: sweep all 16 entries ---");
        for (i = 0; i < 16; i = i + 1) check(i[3:0]);
 
        $display("--- shape checks ---");

        // sigmoid is monotonic non-decreasing, so each entry must be >= the
        // one before it. The midpoint (index 8 = 128) is already covered by
        // the sweep above.
        begin : mono
            reg [7:0] prev;
            integer fails;
            fails = 0;
            index = 4'd0; #1;
            prev = sigmoid_output;
            for (i = 1; i < 16; i = i + 1) begin
                index = i[3:0]; #1;
                if (sigmoid_output < prev) begin
                    $display("FAIL: not monotonic at index %0d (%0d < %0d)",
                             i, sigmoid_output, prev);
                    fails = fails + 1;
                end
                prev = sigmoid_output;
            end
            errors = errors + fails;
            if (fails == 0) $display("pass: monotonic non-decreasing");
        end
 
        if (errors == 0) $display("RESULT: all sigmoid_lut tests passed");
        else             $display("RESULT: %0d failures", errors);
        $finish;
    end

    // watchdog: fail rather than hang if a wait never returns
    initial begin
        #100000;
        $display("FAIL: timeout, testbench did not finish");
        $finish;
    end
 
endmodule