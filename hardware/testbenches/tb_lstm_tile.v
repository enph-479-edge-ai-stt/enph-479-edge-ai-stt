`timescale 1ns / 1ps

// One LSTM layer through the tile, driven the way fsm_controller drives
// it, for two timesteps. Random weights, biases, inputs and starting
// context; the x-phase shift is 2 (the AM's) on step 0 and 7 (the CM's)
// on step 1. Checks:
//   - every gate pre-activation against a software model: x shifted by
//     x_shift, and the h_{t-1} phase that reads h from context memory
//     (sel_in = 1) unshifted
//   - c_t and h_t for every neuron against a second lstm_epu fed with the
//     model's gates (tb_lstm_epu covers the EPU arithmetic itself)
//   - every context word afterwards: this layer's rewritten, others kept
// Uses layer slot 1 and context base 8 so a stuck-at-zero address shows.

module tb_lstm_tile;
    localparam PE = 4;
    localparam D = 3;                 // inputs from below
    localparam W_W = 6;
    localparam B_W = 24;
    localparam WEIGHT_DEPTH = 16;     // uses 2 * (D + PE) = 14
    localparam CONTEXT_DEPTH = 16;
    localparam N_LAYERS = 2;
    localparam GL = 1;                // layer slot for the biases
    localparam CTX = 8;               // this layer's context base

    reg clk = 0;
    reg rst = 1;
    reg pe_enable = 0, load_bias = 0, sel_in = 0;
    reg signed [7:0] d_in = 0;
    reg [2:0] x_shift = 0;
    reg [3:0] weight_addr = 0;
    reg weight0_we = 0, weight1_we = 0;
    reg [PE*W_W-1:0] weight0_din = 0, weight1_din = 0;
    reg [1:0] bias_addr = 0;
    reg bias_we = 0;
    reg [PE*B_W-1:0] bias0_din = 0, bias1_din = 0;
    reg capture_gates = 0, gate_pair = 0;
    reg [1:0] neuron_addr = 0;
    reg [3:0] context_addr = 0;
    reg context_we = 0, context_use_ext = 0;
    reg [23:0] context_ext_din = 0;
    wire signed [15:0] c_t;
    wire signed [7:0] h_t;
    wire [23:0] context_dout;
    wire signed [15:0] pe_i, pe_f, pe_o, pe_c;

    // Software copies of every memory.
    reg [PE*W_W-1:0] wm0 [0:WEIGHT_DEPTH-1];
    reg [PE*W_W-1:0] wm1 [0:WEIGHT_DEPTH-1];
    reg [PE*B_W-1:0] bm0 [0:2*N_LAYERS-1];
    reg [PE*B_W-1:0] bm1 [0:2*N_LAYERS-1];
    reg [23:0] ctxm [0:CONTEXT_DEPTH-1];

    reg signed [7:0] x [0:D-1];
    reg signed [15:0] xs;
    reg signed [23:0] acc0 [0:PE-1];
    reg signed [23:0] acc1 [0:PE-1];
    reg signed [15:0] gi [0:PE-1];
    reg signed [15:0] gf [0:PE-1];
    reg signed [15:0] go [0:PE-1];
    reg signed [15:0] gc [0:PE-1];

    // Reference EPU, driven from the model.
    reg signed [15:0] r_c = 0;
    reg signed [63:0] r_gates = 0;
    wire signed [15:0] ref_c;
    wire signed [7:0] ref_h;

    integer errors = 0;
    integer seed = 2026;
    integer j, k, n, p, t, bad;

    lstm_tile #(
        .PE_COUNT(PE),
        .WEIGHT_DEPTH(WEIGHT_DEPTH),
        .CONTEXT_DEPTH(CONTEXT_DEPTH),
        .N_LSTM_LAYERS(N_LAYERS)
    ) dut (
        .clk(clk), .rst(rst),
        .pe_enable(pe_enable), .load_bias(load_bias), .sel_in(sel_in), .d_in(d_in),
        .x_shift(x_shift),
        .weight_addr(weight_addr), .weight0_we(weight0_we), .weight1_we(weight1_we),
        .weight0_din(weight0_din), .weight1_din(weight1_din),
        .bias_addr(bias_addr), .bias_we(bias_we), .bias0_din(bias0_din), .bias1_din(bias1_din),
        .capture_gates(capture_gates), .gate_pair(gate_pair), .neuron_addr(neuron_addr),
        .context_addr(context_addr), .context_we(context_we),
        .context_use_ext(context_use_ext), .context_ext_din(context_ext_din),
        .c_t(c_t), .h_t(h_t), .context_dout(context_dout),
        .pe_i(pe_i), .pe_f(pe_f), .pe_o(pe_o), .pe_c(pe_c)
    );

    lstm_epu ref_epu (
        .clk(1'b0), .rst(1'b0), .enable(1'b1),
        .old_hidden_state(r_c),
        .pe_input(r_gates),
        .hidden_state_output(ref_c),
        .output_state(ref_h)
    );

    always #5 clk = ~clk;

    task fail;
        input [8*64-1:0] msg;
        begin
            $display("FAIL: %0s", msg);
            errors = errors + 1;
        end
    endtask

    task load_memories;
        begin
            for (j = 0; j < WEIGHT_DEPTH; j = j + 1) begin
                for (n = 0; n < PE; n = n + 1) begin
                    wm0[j][n*W_W +: W_W] = $random(seed);
                    wm1[j][n*W_W +: W_W] = $random(seed);
                end
                @(negedge clk);
                weight_addr = j;
                weight0_din = wm0[j];
                weight1_din = wm1[j];
                weight0_we = 1;
                weight1_we = 1;
            end
            @(negedge clk);
            weight0_we = 0;
            weight1_we = 0;

            for (j = 0; j < 2 * N_LAYERS; j = j + 1) begin
                for (n = 0; n < PE; n = n + 1) begin
                    bm0[j][n*B_W +: B_W] = $random(seed);
                    bm1[j][n*B_W +: B_W] = $random(seed);
                end
                @(negedge clk);
                bias_addr = j;
                bias0_din = bm0[j];
                bias1_din = bm1[j];
                bias_we = 1;
            end
            @(negedge clk);
            bias_we = 0;

            // Starting context through the external write port.
            for (j = 0; j < CONTEXT_DEPTH; j = j + 1) begin
                ctxm[j] = $random(seed);
                @(negedge clk);
                context_addr = j;
                context_ext_din = ctxm[j];
                context_use_ext = 1;
                context_we = 1;
            end
            @(negedge clk);
            context_we = 0;
            context_use_ext = 0;
        end
    endtask

    // Model of one pass: bias, D inputs from x shifted left by x_shift,
    // PE inputs of h from context.
    task model_pass;
        input integer pass;
        integer base;
        begin
            base = pass * (D + PE);
            for (n = 0; n < PE; n = n + 1) begin
                acc0[n] = $signed(bm0[GL*2 + pass][n*B_W +: B_W]);
                acc1[n] = $signed(bm1[GL*2 + pass][n*B_W +: B_W]);
                for (k = 0; k < D; k = k + 1) begin
                    xs = x[k] <<< x_shift;
                    acc0[n] = acc0[n] + xs * $signed(wm0[base + k][n*W_W +: W_W]);
                    acc1[n] = acc1[n] + xs * $signed(wm1[base + k][n*W_W +: W_W]);
                end
                for (k = 0; k < PE; k = k + 1) begin
                    acc0[n] = acc0[n] + $signed(ctxm[CTX + k][7:0])
                        * $signed(wm0[base + D + k][n*W_W +: W_W]);
                    acc1[n] = acc1[n] + $signed(ctxm[CTX + k][7:0])
                        * $signed(wm1[base + D + k][n*W_W +: W_W]);
                end
                // ACC_LSB = 0: the gate is the accumulator's low 16 bits.
                if (pass == 0) begin
                    gi[n] = acc0[n][15:0];
                    gf[n] = acc1[n][15:0];
                end else begin
                    go[n] = acc0[n][15:0];
                    gc[n] = acc1[n][15:0];
                end
            end
        end
    endtask

    // Drive one pass with the controller's timing: every read address is
    // issued one cycle before pe_enable uses its data.
    task drive_pass;
        input integer pass;
        integer base;
        begin
            base = pass * (D + PE);
            @(negedge clk);
            bias_addr = GL * 2 + pass;
            @(negedge clk);
            load_bias = 1;
            @(negedge clk);
            load_bias = 0;
            for (k = 0; k <= D; k = k + 1) begin
                if (k < D)
                    weight_addr = base + k;
                sel_in = 0;
                d_in = (k > 0) ? x[k-1] : 8'sd0;
                pe_enable = (k > 0);
                @(negedge clk);
            end
            for (k = 0; k <= PE; k = k + 1) begin
                if (k < PE) begin
                    weight_addr = base + D + k;
                    context_addr = CTX + k;
                end
                sel_in = 1;
                pe_enable = (k > 0);
                @(negedge clk);
            end
            pe_enable = 0;
            sel_in = 0;
            gate_pair = pass;
            capture_gates = 1;
            @(negedge clk);
            capture_gates = 0;
        end
    endtask

    task check_gates;
        input integer pass;
        begin
            bad = 0;
            for (n = 0; n < PE; n = n + 1) begin
                neuron_addr = n;
                #1;
                if (pe_i !== gi[n] || pe_f !== gf[n]) begin
                    $display("  t%0d n%0d i/f: got %0d %0d expected %0d %0d",
                        t, n, pe_i, pe_f, gi[n], gf[n]);
                    bad = bad + 1;
                end
                if (pass == 1 && (pe_o !== go[n] || pe_c !== gc[n])) begin
                    $display("  t%0d n%0d o/c: got %0d %0d expected %0d %0d",
                        t, n, pe_o, pe_c, go[n], gc[n]);
                    bad = bad + 1;
                end
            end
            if (bad == 0)
                $display("pass: step %0d pass %0d gate pre-activations", t, pass);
            else
                fail("gate pre-activations");
        end
    endtask

    // EPU sweep: present addresses, check c_t/h_t next cycle, write back.
    task epu_sweep;
        begin
            bad = 0;
            for (n = 0; n < PE; n = n + 1) begin
                @(negedge clk);
                context_we = 0;
                context_addr = CTX + n;
                neuron_addr = n;
                @(negedge clk);
                r_c = ctxm[CTX + n][23:8];
                r_gates = {gc[n], go[n], gf[n], gi[n]};
                #1;
                if (c_t !== ref_c || h_t !== ref_h) begin
                    $display("  t%0d n%0d: c_t %0d h_t %0d, expected %0d %0d",
                        t, n, c_t, h_t, ref_c, ref_h);
                    bad = bad + 1;
                end
                ctxm[CTX + n] = {ref_c, ref_h};
                context_we = 1;
            end
            @(negedge clk);
            context_we = 0;
            if (bad == 0)
                $display("pass: step %0d c_t and h_t for every neuron", t);
            else
                fail("EPU outputs");
        end
    endtask

    task check_context;
        begin
            bad = 0;
            for (j = 0; j < CONTEXT_DEPTH; j = j + 1) begin
                @(negedge clk);
                context_addr = j;
                @(negedge clk);
                if (context_dout !== ctxm[j]) begin
                    $display("  t%0d context[%0d] = %h, expected %h", t, j, context_dout, ctxm[j]);
                    bad = bad + 1;
                end
            end
            if (bad == 0)
                $display("pass: step %0d all %0d context words", t, CONTEXT_DEPTH);
            else
                fail("context contents");
        end
    endtask

    initial begin
        repeat (2) @(negedge clk);
        rst = 0;
        load_memories;

        // Step 1 reads h_{t-1} that step 0 wrote.
        for (t = 0; t < 2; t = t + 1) begin
            x_shift = (t == 0) ? 3'd2 : 3'd7;
            for (k = 0; k < D; k = k + 1)
                x[k] = $random(seed);
            for (p = 0; p < 2; p = p + 1) begin
                model_pass(p);
                drive_pass(p);
                check_gates(p);
            end
            epu_sweep;
            check_context;
        end

        if (errors == 0)
            $display("ALL CHECKS PASSED");
        else
            $display("%0d CHECKS FAILED", errors);
        $finish;
    end
endmodule
