`timescale 1ns / 1ps
 
module tb_tanh_lut;
 
    reg  [3:0] index;
    wire signed [7:0] tanh_output;
 
    integer errors = 0;
    integer i;
 
    // expected values, mirrored from the LUT table
    reg signed [7:0] expected [0:15];
 
    tanh_lut dut (
        .index       (index),
        .tanh_output (tanh_output)
    );
 
    task check(input [3:0] idx);
        begin
            index = idx;
            #1;
            if (tanh_output !== expected[idx]) begin
                $display("FAIL: index=%0d  got=%0d  expected=%0d",
                         idx, tanh_output, expected[idx]);
                errors = errors + 1;
            end else begin
                $display("pass: index=%0d  out=%0d", idx, tanh_output);
            end
        end
    endtask
 
    initial begin
        $dumpfile("tb_tanh_lut.vcd");
        $dumpvars(0, tb_tanh_lut);

        expected[0]  = -127; expected[1]  = -127; expected[2]  = -127; expected[3]  = -127;
        expected[4]  = -127; expected[5]  = -126; expected[6]  = -122; expected[7]  = -97;
        expected[8]  =    0; expected[9]  =   97; expected[10] =  122; expected[11] =  126;
        expected[12] =  127; expected[13] =  127; expected[14] =  127; expected[15] =  127;
 
        $display("--- tanh_lut: sweep all 16 entries ---");
        for (i = 0; i < 16; i = i + 1) check(i[3:0]);
 
        $display("--- shape checks ---");
 
        // midpoint must be 0, unlike sigmoid which is 128 there
        index = 4'd8; #1;
        if (tanh_output !== 8'sd0) begin
            $display("FAIL: index 8 should be 0"); errors = errors + 1;
        end else $display("pass: midpoint is 0");
 
        // negative half must actually read as negative, catches a missing 'signed'
        index = 4'd7; #1;
        if (!(tanh_output < 0)) begin
            $display("FAIL: index 7 should be negative, got %0d", tanh_output);
            errors = errors + 1;
        end else $display("pass: lower half is negative");
 
        // odd symmetry: tanh(-x) = -tanh(x), so entries either side of 8 mirror
        for (i = 1; i <= 7; i = i + 1) begin
            index = 8 - i; #1;
            begin : sym
                reg signed [7:0] neg_side;
                neg_side = tanh_output;
                index = 8 + i; #1;
                if (neg_side !== -tanh_output) begin
                    $display("FAIL: symmetry broken at offset %0d (%0d vs %0d)",
                             i, neg_side, tanh_output);
                    errors = errors + 1;
                end else begin
                    $display("pass: symmetric at offset %0d (%0d / %0d)",
                             i, neg_side, tanh_output);
                end
            end
        end
 
        if (errors == 0) $display("RESULT: all tanh_lut tests passed");
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