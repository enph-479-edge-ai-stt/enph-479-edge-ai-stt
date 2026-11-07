`timescale 1ns / 1ps

// Output projection: random weights and biases for two jobs, a full
// HIDDEN-long dot product per job driven with the one-cycle read latency
// the controller uses, every symbol checked against a software model.
// Running job 1 after job 0 checks that the bias and weight blocks are
// selected by address.

module tb_output_tile;
    localparam HIDDEN = 8;
    localparam VOCAB = 5;
    localparam N_JOBS = 2;
    localparam W_W = 6;
    localparam B_W = 16;
    localparam ACC_W = 24;
    localparam S_W = 8;
    localparam DEPTH = HIDDEN * N_JOBS;

    reg clk = 0;
    reg rst = 1;
    reg enable = 0;
    reg load_bias = 0;
    reg signed [7:0] h_element = 0;
    reg [3:0] weight_addr = 0;
    reg weight_we = 0;
    reg [VOCAB*W_W-1:0] weight_din = 0;
    reg bias_addr = 0;
    reg bias_we = 0;
    reg [VOCAB*B_W-1:0] bias_din = 0;
    wire signed [VOCAB*ACC_W-1:0] acc;
    wire signed [VOCAB*S_W-1:0] scores;

    reg [VOCAB*W_W-1:0] weights [0:DEPTH-1];
    reg [VOCAB*B_W-1:0] biases [0:N_JOBS-1];
    reg signed [7:0] h [0:HIDDEN-1];
    reg signed [ACC_W-1:0] model [0:VOCAB-1];

    integer errors = 0;
    integer seed = 30;
    integer j, k, v;

    output_tile #(
        .HIDDEN(HIDDEN), .VOCAB(VOCAB), .N_JOBS(N_JOBS)
    ) dut (
        .clk(clk), .rst(rst), .enable(enable), .load_bias(load_bias),
        .h_element(h_element),
        .weight_addr(weight_addr), .weight_we(weight_we), .weight_din(weight_din),
        .bias_addr(bias_addr), .bias_we(bias_we), .bias_din(bias_din),
        .acc(acc), .scores(scores)
    );

    always #5 clk = ~clk;

    // Bias read, load, then one MAC per h with the row issued a cycle early.
    task run_job;
        input integer job;
        begin
            for (k = 0; k < HIDDEN; k = k + 1)
                h[k] = $random(seed);
            for (v = 0; v < VOCAB; v = v + 1) begin
                model[v] = $signed(biases[job][v*B_W +: B_W]);
                for (k = 0; k < HIDDEN; k = k + 1)
                    model[v] = model[v] + h[k] * $signed(weights[job*HIDDEN + k][v*W_W +: W_W]);
            end

            @(negedge clk);
            bias_addr = job;
            @(negedge clk);
            load_bias = 1;
            @(negedge clk);
            load_bias = 0;
            for (k = 0; k <= HIDDEN; k = k + 1) begin
                if (k < HIDDEN)
                    weight_addr = job * HIDDEN + k;
                enable = (k > 0);
                h_element = (k > 0) ? h[k-1] : 8'sd0;
                @(negedge clk);
            end
            enable = 0;
        end
    endtask

    task check_job;
        input integer job;
        integer bad;
        begin
            bad = 0;
            for (v = 0; v < VOCAB; v = v + 1) begin
                if (acc[v*ACC_W +: ACC_W] !== model[v]) begin
                    $display("  job %0d symbol %0d: acc %0d expected %0d",
                        job, v, $signed(acc[v*ACC_W +: ACC_W]), model[v]);
                    bad = bad + 1;
                end
                // SCORE_LSB = 0: score is the accumulator's low byte (placeholder).
                if (scores[v*S_W +: S_W] !== model[v][S_W-1:0]) begin
                    $display("  job %0d symbol %0d: score %h expected %h",
                        job, v, scores[v*S_W +: S_W], model[v][S_W-1:0]);
                    bad = bad + 1;
                end
            end
            if (bad == 0)
                $display("pass: job %0d, %0d symbols over %0d inputs", job, VOCAB, HIDDEN);
            else begin
                $display("FAIL: job %0d", job);
                errors = errors + 1;
            end
        end
    endtask

    initial begin
        repeat (2) @(negedge clk);
        rst = 0;

        for (j = 0; j < DEPTH; j = j + 1) begin
            for (v = 0; v < VOCAB; v = v + 1)
                weights[j][v*W_W +: W_W] = $random(seed);
            @(negedge clk);
            weight_addr = j;
            weight_din = weights[j];
            weight_we = 1;
        end
        @(negedge clk);
        weight_we = 0;
        for (j = 0; j < N_JOBS; j = j + 1) begin
            for (v = 0; v < VOCAB; v = v + 1)
                biases[j][v*B_W +: B_W] = $random(seed);
            @(negedge clk);
            bias_addr = j;
            bias_din = biases[j];
            bias_we = 1;
        end
        @(negedge clk);
        bias_we = 0;

        run_job(0);
        check_job(0);
        run_job(1);
        check_job(1);
        run_job(0);
        check_job(0);

        // Scores hold after the job while enable is low.
        @(negedge clk);
        h_element = 8'sd99;
        repeat (3) @(negedge clk);
        check_job(0);

        if (errors == 0)
            $display("ALL CHECKS PASSED");
        else
            $display("%0d CHECKS FAILED", errors);
        $finish;
    end
endmodule
