`timescale 1ns / 1ps

// Both 256-PE arrays against a software model: random signed biases,
// weights and inputs, every lane of both outputs checked. Also covers
// reset, hold while enable is low, and load_bias winning over enable.
// Inputs change on the falling edge so the DUT samples them cleanly.

module tb_pe_array;
    localparam N = 256;
    localparam X_W = 16;              // as lstm_tile's PE_X_WIDTH
    localparam W_W = 6;
    localparam B_W = 24;
    localparam ACC_W = 24;
    localparam MAC_STEPS = 40;

    reg clk = 0;
    reg rst = 1;
    reg enable = 0;
    reg load_bias = 0;
    reg signed [X_W-1:0] data_input = 0;
    reg [W_W*N-1:0] weight0 = 0;
    reg [W_W*N-1:0] weight1 = 0;
    reg [B_W*N-1:0] bias0 = 0;
    reg [B_W*N-1:0] bias1 = 0;
    wire signed [ACC_W*N-1:0] pe_out0;
    wire signed [ACC_W*N-1:0] pe_out1;

    reg signed [ACC_W-1:0] model0 [0:N-1];
    reg signed [ACC_W-1:0] model1 [0:N-1];

    integer errors = 0;
    integer seed = 479;
    integer k, s;

    pe_array #(
        .PE_ARRAY_SIZE(N),
        .DATA_ELEMENT_WIDTH(X_W),
        .WEIGHT_INPUT_WIDTH(W_W),
        .BIAS_INPUT_WIDTH(B_W),
        .OUTPUT_ELEMENT_WIDTH(ACC_W)
    ) dut (
        .clk(clk), .rst(rst), .enable(enable), .load_bias(load_bias),
        .data_input(data_input),
        .weight_input0(weight0), .weight_input1(weight1),
        .bias_input0(bias0), .bias_input1(bias1),
        .pe_out0(pe_out0), .pe_out1(pe_out1)
    );

    always #5 clk = ~clk;

    task check_all;
        input [8*48-1:0] msg;
        integer bad;
        begin
            bad = 0;
            for (k = 0; k < N; k = k + 1) begin
                if (pe_out0[k*ACC_W +: ACC_W] !== model0[k]) begin
                    if (bad < 4)
                        $display("  array0 lane %0d: got %0d expected %0d",
                            k, $signed(pe_out0[k*ACC_W +: ACC_W]), model0[k]);
                    bad = bad + 1;
                end
                if (pe_out1[k*ACC_W +: ACC_W] !== model1[k]) begin
                    if (bad < 4)
                        $display("  array1 lane %0d: got %0d expected %0d",
                            k, $signed(pe_out1[k*ACC_W +: ACC_W]), model1[k]);
                    bad = bad + 1;
                end
            end
            if (bad == 0)
                $display("pass: %0s", msg);
            else begin
                $display("FAIL: %0s (%0d lanes wrong)", msg, bad);
                errors = errors + 1;
            end
        end
    endtask

    task randomize_weights;
        begin
            for (k = 0; k < N; k = k + 1) begin
                weight0[k*W_W +: W_W] = $random(seed);
                weight1[k*W_W +: W_W] = $random(seed);
            end
        end
    endtask

    // Random biases, one cycle of load_bias, model reset to the biases.
    task load_random_bias;
        begin
            @(negedge clk);
            for (k = 0; k < N; k = k + 1) begin
                bias0[k*B_W +: B_W] = $random(seed);
                bias1[k*B_W +: B_W] = $random(seed);
                model0[k] = $signed(bias0[k*B_W +: B_W]);
                model1[k] = $signed(bias1[k*B_W +: B_W]);
            end
            load_bias = 1;
            @(negedge clk);
            load_bias = 0;
        end
    endtask

    // MAC_STEPS cycles of random data and weights, model updated alongside.
    task random_macs;
        begin
            for (s = 0; s < MAC_STEPS; s = s + 1) begin
                data_input = $random(seed);
                randomize_weights;
                enable = 1;
                for (k = 0; k < N; k = k + 1) begin
                    model0[k] = model0[k] + data_input * $signed(weight0[k*W_W +: W_W]);
                    model1[k] = model1[k] + data_input * $signed(weight1[k*W_W +: W_W]);
                end
                @(negedge clk);
            end
            enable = 0;
        end
    endtask

    initial begin
        for (k = 0; k < N; k = k + 1) begin
            model0[k] = 0;
            model1[k] = 0;
        end
        repeat (2) @(negedge clk);
        check_all("reset clears every lane");
        rst = 0;

        load_random_bias;
        check_all("load_bias loads 24-bit biases");

        random_macs;
        check_all("random MACs match the model");

        // enable low: new inputs must not change anything.
        data_input = 8'sd77;
        randomize_weights;
        repeat (3) @(negedge clk);
        check_all("enable low holds the accumulators");

        // A second bias load replaces the sums, then accumulate again.
        load_random_bias;
        random_macs;
        check_all("second bias load and run");

        // load_bias and enable together: bias wins, no MAC.
        @(negedge clk);
        for (k = 0; k < N; k = k + 1) begin
            bias0[k*B_W +: B_W] = $random(seed);
            bias1[k*B_W +: B_W] = $random(seed);
            model0[k] = $signed(bias0[k*B_W +: B_W]);
            model1[k] = $signed(bias1[k*B_W +: B_W]);
        end
        data_input = 8'sd100;
        randomize_weights;
        load_bias = 1;
        enable = 1;
        @(negedge clk);
        load_bias = 0;
        enable = 0;
        check_all("load_bias wins over enable");

        // Reset in the middle of a run.
        rst = 1;
        for (k = 0; k < N; k = k + 1) begin
            model0[k] = 0;
            model1[k] = 0;
        end
        @(negedge clk);
        rst = 0;
        check_all("reset after accumulation");

        if (errors == 0)
            $display("ALL CHECKS PASSED");
        else
            $display("%0d CHECKS FAILED", errors);
        $finish;
    end
endmodule
