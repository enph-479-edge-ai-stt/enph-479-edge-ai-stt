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

    initial begin
        $dumpfile("tb_pe_unit.vcd"); // Waveform dump
        $dumpvars(0, tb_pe_unit);

        rst = 1;
        load_bias = 0;
        enable = 0;
        
        bias = 0;
        x_input = 0;
        weight = 0;

        errors = 0;

        // Test if pe_out is zeroed
        $display("Testing pe_out initial condition...");
        @(posedge clk);
        #1;
        if (pe_out !== 0) begin
            $display("FAIL: got %0d", pe_out);
            errors = errors + 1;
        end
        else
            $display("SUCCESS: pe_out is 0");
        
        // Begin next phase: validate bias
        rst = 0;
        load_bias = 1;
        bias = 1000;

        @(posedge clk);
        #1;

        // Test if bias has been loaded
        if (pe_out == 1000) begin
            $display("SUCCESS: bias value is 1000 and was loaded into pe_out");
        end
        else begin
            $display("FAIL: bias value loaded into pe_out is %0d", pe_out);
            errors = errors + 1;
        end

        bias = -1000; // Test signed bias

        @(posedge clk);
        #1;

        // Test if signed bias works too
        if(pe_out == -1000) begin
            $display("SUCCESS: bias value is -1000 and was loaded into pe_out");
        end
        else begin
            $display("FAIL: bias value loaded into pe_out is %0d", pe_out);
            errors = errors + 1;
        end

        load_bias = 0; // Finished testing bias, so deassert
        
        // Start testing MAC operations
        enable = 1;
        x_input = 10;
        weight = 5;

        @(posedge clk);
        #1;

        // Test if MAC operation has been successfuly performed and added to bias
        if(pe_out == -950)
            $display("SUCCESS: pe_out has been calculated successfully (%0d) with bias: %0d, weight: %0d, x_input: %0d", pe_out, bias, weight, x_input);
        else begin
            $display("FAILURE: pe_out was not calculated successfully. pe_out = %0d, bias = %0d, weight = %0d, x_input = %0d", pe_out, bias, weight, x_input);
            errors = errors + 1;
        end

        enable = 0;

    
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