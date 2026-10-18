`timescale 1ns / 1ps

module tb_pe_unit;
    localparam X_ELEMENT_WIDTH = 8;
    localparam WEIGHT_ELEMENT_WIDTH = 6;     
    localparam OUTPUT_ELEMENT_WIDTH = 24;

    reg clk = 0;
    reg rst;
    reg enable;

    reg load_bias;
    reg signed [OUTPUT_ELEMENT_WIDTH-1:0] bias;

    reg signed [X_ELEMENT_WIDTH-1:0] x_input;
    reg signed [WEIGHT_ELEMENT_WIDTH-1:0] weight;

    wire signed [OUTPUT_ELEMENT_WIDTH-1:0] pe_out;

    integer errors;
    integer i;

    pe_unit #(
        .X_ELEMENT_WIDTH        (X_ELEMENT_WIDTH),
        .WEIGHT_ELEMENT_WIDTH   (WEIGHT_ELEMENT_WIDTH),
        .OUTPUT_ELEMENT_WIDTH   (OUTPUT_ELEMENT_WIDTH)
    ) dut (
        .clk(clk),
        .rst(rst),
        .enable(enable),

        .load_bias(load_bias),
        .bias(bias),

        .x_input(x_input),
        .weight(weight),

        .pe_out(pe_out)
    );

    always #5 clk = ~clk;

    task check_value(input integer expected, input [8*40:1] label);
        begin
            $display("TEST: %0s", label);
            if (pe_out !== expected) begin
                $display("FAIL: got %0d and expected %0d", pe_out, expected);
                errors = errors + 1;
            end
            else
                $display("SUCCESS: pe_out is %0d", expected);
        end
    endtask

    task set_values(input integer r, input integer lb, input integer e, input signed [OUTPUT_ELEMENT_WIDTH-1:0] b, 
    input signed [X_ELEMENT_WIDTH-1:0] x, input signed [WEIGHT_ELEMENT_WIDTH-1:0] w);
        begin
            rst = r;
            load_bias = lb;
            enable = e;
            bias = b;
            x_input = x;
            weight = w;
            @(posedge clk);
            #1;
        end
    endtask

    initial begin
        $dumpfile("tb_pe_unit.vcd"); // Waveform dump
        $dumpvars(0, tb_pe_unit);

        errors = 0;

        // Check if pe_out is zeroed
        // Reset = 0
        set_values(1, 0, 0, 0, 0, 0);
        check_value(0, "Check if pe_out is zeroed");
        
        // Begin next phase: validate bias
        set_values(0, 1, 0, 1000, 0, 0);
        // Check if biased has been loaded
        check_value(1000, "Check if bias has been loaded");

        set_values(0, 1, 0, -1000, 0, 0);
        check_value(-1000, "Check if negative bias has been loaded");

        // Deassert all
        set_values(0, 0, 0, 0, 0, 0);
        check_value(-1000, "Hold when idle");

        // Start testing MAC operations
        // Signed MAC: load bias 0 before each so expected = product
        set_values(0, 1, 0, 0, 0, 0);
        set_values(0, 0, 1, 0, 10, 5);
        check_value(50, "MAC pos * pos");

        set_values(0, 1, 0, 0, 0, 0);
        set_values(0, 0, 1, 0, -10, 5);
        check_value(-50, "MAC neg * pos");

        set_values(0, 1, 0, 0, 0, 0);
        set_values(0, 0, 1, 0, 10, -5);
        check_value(-50, "MAC pos * neg");

        set_values(0, 1, 0, 0, 0, 0);
        set_values(0, 0, 1, 0, -10, -5);
        check_value(50, "MAC neg * neg");

        // Bias + MAC
        set_values(0, 1, 0, -1000, 0, 0);
        set_values(0, 0, 1, 0, 10, 5);
        check_value(-950, "MAC on top of bias: -1000 + 10*5");

        // Extreme operands: x in [-128, 127], w in [-32, 31]
        set_values(0, 1, 0, 0, 0, 0);
        set_values(0, 0, 1, 0, 127, 31);
        check_value(3937, "MAC max * max w");

        set_values(0, 1, 0, 0, 0, 0);
        set_values(0, 0, 1, 0, -128, 31);
        check_value(-3968, "MAC min x * max w");

        set_values(0, 1, 0, 0, 0, 0);
        set_values(0, 0, 1, 0, 127, -32);
        check_value(-4064, "MAC max * min w");

        set_values(0, 1, 0, 0, 0, 0);
        set_values(0, 0, 1, 0, -128, -32);
        check_value(4096, "MAC min * min (largest product)");

        // Accumulation over several cycles
        set_values(1, 0, 0, 0, 0, 0);
        set_values(0, 0, 1, 0, 10, 5);
        check_value(50, "Accumulate step 1: +10*5");
        set_values(0, 0, 1, 0, -3, 4);
        check_value(38, "Accumulate step 2: -3*4");
        set_values(0, 0, 1, 0, 7, -2);
        check_value(24, "Accumulate step 3: 7*-2");

        // Full gate dot product: 379 worst-case terms on a bias of 100
        set_values(0, 1, 0, 100, 0, 0);
        for (i = 0; i < 379; i = i + 1)
            set_values(0, 0, 1, 0, -128, -32);
        check_value(100 + 379 * 4096, "Accumulate 379 worst-case terms");

        // Enable gates the MAC even with nonzero operands
        set_values(1, 0, 0, 0, 0, 0);
        set_values(0, 0, 0, 0, 99, 9);
        check_value(0, "Hold with enable=0, nonzero x and w");

        // Priority: rst > load_bias > enable
        set_values(0, 1, 1, 500, 10, 5);
        check_value(500, "load_bias beats enable");

        set_values(1, 1, 0, 500, 0, 0);
        check_value(0, "rst beats load_bias");

        set_values(0, 1, 0, 0, 0, 0);
        set_values(0, 0, 1, 0, 10, 5);
        set_values(1, 0, 1, 0, 10, 5);
        check_value(0, "rst beats enable mid-accumulation");

        // Overflow: the 24-bit accumulator wraps, it does not saturate
        set_values(0, 1, 0, 8388607, 0, 0);
        set_values(0, 0, 1, 0, 1, 1);
        check_value(-8388608, "Overflow wraps 2^23-1 + 1");

        if (errors == 0) $display("RESULT: all pe_unit tests passed");
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