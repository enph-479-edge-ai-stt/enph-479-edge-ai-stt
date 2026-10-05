`timescale 1ns / 1ps
 
module tb_weight_bram;
 
    localparam WEIGHT_COUNT = 123;
    localparam WEIGHT_ELEMENT_WIDTH = 6;
    localparam NEURON_COUNT = 256;
    localparam ROW_WIDTH = NEURON_COUNT * WEIGHT_ELEMENT_WIDTH;  // 1536
    localparam ADDR_WIDTH = $clog2(WEIGHT_COUNT);
 
    reg clk = 0;
    reg write_enable = 0;
    reg [ADDR_WIDTH-1:0] address = 0;
    reg [ROW_WIDTH-1:0]  data_in = 0;
    wire [ROW_WIDTH-1:0] data_out;
 
    integer errors = 0;
 
    always #5 clk = ~clk;   // 100 MHz
 
    weight_bram #(
        .WEIGHT_COUNT(WEIGHT_COUNT),
        .WEIGHT_ELEMENT_WIDTH(WEIGHT_ELEMENT_WIDTH),
        .NEURON_COUNT(NEURON_COUNT)
    ) dut (
        .clk(clk), .write_enable(write_enable),
        .address(address), .data_in(data_in), .data_out(data_out)
    );
 
    // Write one row
    task write_row(input [ADDR_WIDTH-1:0] a, input [ROW_WIDTH-1:0] d);
        begin
            @(negedge clk);
            address = a; data_in = d; write_enable = 1;
            @(negedge clk);
            write_enable = 0;
        end
    endtask
 
    // Read one row (registered output: valid one clock after address)
    task read_row(input [ADDR_WIDTH-1:0] a, input [ROW_WIDTH-1:0] exp);
        begin
            @(negedge clk);
            address = a;
            @(negedge clk);          // Data_out updates on the posedge between
            if (data_out !== exp) begin
                $display("FAIL: addr=%0d  got=%h  expected=%h",
                         a, data_out[31:0], exp[31:0]);
                errors = errors + 1;
            end else begin
                $display("pass: addr=%0d  low32=%h", a, data_out[31:0]);
            end
        end
    endtask
 
    // Build a row where neuron k holds weight (k mod 64)
    function [ROW_WIDTH-1:0] pattern_row(input integer seed);
        integer k;
        begin
            pattern_row = 0;
            for (k = 0; k < NEURON_COUNT; k = k + 1)
                pattern_row[k*WEIGHT_ELEMENT_WIDTH +: WEIGHT_ELEMENT_WIDTH] =
                    (k + seed) % (1 << WEIGHT_ELEMENT_WIDTH);
        end
    endfunction
 
    reg [ROW_WIDTH-1:0] row_a, row_b, row_c;
 
    initial begin
        $dumpfile("tb_weight_bram.vcd");
        $dumpvars(0, tb_weight_bram);

        row_a = pattern_row(0);
        row_b = pattern_row(7);
        row_c = {ROW_WIDTH{1'b1}};     // all ones
 
        $display("--- weight_bram: write then read back ---");
        write_row(0,   row_a);
        write_row(61,  row_b);
        write_row(122, row_c);         // last address
 
        read_row(0,   row_a);
        read_row(61,  row_b);
        read_row(122, row_c);
 
        // check a single neuron's slice came through intact
        @(negedge clk); address = 0; @(negedge clk);
        if (data_out[5:0] !== 6'd0 || data_out[11:6] !== 6'd1) begin
            $display("FAIL: per-neuron slices  neuron0=%0d (exp 0)  neuron1=%0d (exp 1)",
                     data_out[5:0], data_out[11:6]);
            errors = errors + 1;
        end else begin
            $display("pass: per-neuron slices readable");
        end
 
        // Overwrite and confirm the new value sticks
        write_row(0, row_c);
        read_row(0, row_c);
 
        if (errors == 0) $display("RESULT: all weight_bram tests passed");
        else             $display("RESULT: %0d failures", errors);
        $finish;
    end

    // Watchdog: fail rather than hang if a wait never returns
    initial begin
        #100000;
        $display("FAIL: timeout, testbench did not finish");
        $finish;
    end
 
endmodule