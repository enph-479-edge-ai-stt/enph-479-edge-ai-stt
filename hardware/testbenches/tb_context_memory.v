`timescale 1ns / 1ps

module tb_context_memory;

    localparam STATE_WIDTH = 24;
    localparam DEPTH       = 1024;
    localparam ADDR_WIDTH  = $clog2(DEPTH);

    reg clk = 0;
    reg write_enable = 0;
    reg [ADDR_WIDTH-1:0]  address = 0;
    reg [STATE_WIDTH-1:0] data_in = 0;
    wire [STATE_WIDTH-1:0] data_out;

    integer errors = 0;
    integer i;

    always #5 clk = ~clk;

    context_memory #(
        .STATE_WIDTH(STATE_WIDTH), 
        .DEPTH(DEPTH)
    ) dut (
        .clk(clk), 
        .write_enable(write_enable),
        .address(address), 
        .data_in(data_in), 
        .data_out(data_out)
    );

    task write_state(input [ADDR_WIDTH-1:0] a,
                     input signed [15:0] c_val,
                     input signed [7:0]  h_val);
        begin
            @(negedge clk);
            address = a; data_in = {c_val, h_val}; write_enable = 1;
            @(negedge clk);
            write_enable = 0;
        end
    endtask

    task read_state(input [ADDR_WIDTH-1:0] a,
                    input signed [15:0] exp_c,
                    input signed [7:0]  exp_h);
        begin
            @(negedge clk);
            address = a;
            @(negedge clk);
            if (data_out[23:8] !== exp_c || data_out[7:0] !== exp_h) begin
                $display("FAIL: addr=%0d  got c=%0d h=%0d  expected c=%0d h=%0d",
                         a, $signed(data_out[23:8]), $signed(data_out[7:0]), exp_c, exp_h);
                errors = errors + 1;
            end else begin
                $display("pass: addr=%0d  c=%0d  h=%0d",
                         a, $signed(data_out[23:8]), $signed(data_out[7:0]));
            end
        end
    endtask

    initial begin
        $dumpfile("tb_context_memory.vcd");
        $dumpvars(0, tb_context_memory);

        $display("--- context_memory: c_t / h_t pack and unpack ---");

        // positive values
        write_state(0, 16'sd1234, 8'sd56);
        read_state (0, 16'sd1234, 8'sd56);

        // negative values (state is signed)
        write_state(5, -16'sd4321, -8'sd12);
        read_state (5, -16'sd4321, -8'sd12);

        // extremes: most positive, then most negative. The negative
        // extremes are written as bit patterns, since -16'sd32768 only
        // lands on -32768 via two overflows.
        write_state(9, 16'sh7FFF, 8'sh7F);
        read_state (9, 16'sh7FFF, 8'sh7F);
        write_state(10, 16'sh8000, 8'sh80);
        read_state (10, 16'sh8000, 8'sh80);

        $display("--- separate entries stay independent ---");
        for (i = 0; i < 8; i = i + 1)
            write_state(i[ADDR_WIDTH-1:0], i*100, i);
        for (i = 0; i < 8; i = i + 1)
            read_state(i[ADDR_WIDTH-1:0], i*100, i);

        // overwrite, simulating a new time step
        write_state(3, 16'sd999, 8'sd7);
        read_state (3, 16'sd999, 8'sd7);

        $display("--- read-during-write returns old data (read-first) ---");
        // The EPU relies on this: reading and writing the same entry in one
        // cycle must hand back the previous state, not the value being written.
        write_state(20, 16'sd111, 8'sd11);
        @(negedge clk);
        address = 20; data_in = {16'sd222, 8'sd22}; write_enable = 1;
        @(negedge clk);
        write_enable = 0;
        if (data_out !== {16'sd111, 8'sd11}) begin
            $display("FAIL: read-during-write  got c=%0d h=%0d  expected old c=111 h=11",
                     $signed(data_out[23:8]), $signed(data_out[7:0]));
            errors = errors + 1;
        end else begin
            $display("pass: read-during-write returned old state");
        end
        // and the write itself still landed
        read_state(20, 16'sd222, 8'sd22);

        if (errors == 0) $display("RESULT: all context_memory tests passed");
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