`timescale 1ns / 1ps

// Whole accelerator at small sizes (4 PEs, 2-layer AM, 2-layer LM, two
// beams) with random nonzero weights, biases and frames. A software
// model follows every job with the controller's memory map: weight rows
// per layer and pass, bias slots, context bases per layer and beam, the
// projection block per job, and layer 0's x-phase shift (2 for the AM,
// 7 for the LM). After every job each score and every context word is
// checked.
//
// The model's EPU is a second lstm_epu instance (tb_lstm_epu checks the
// EPU arithmetic), so this test is about sequencing and addressing.
//
// Sequence: clear all, two AM frames (the second reads the first's h),
// LM steps on beam 0, beam 1, beam 0, clear beam 0, clear the AM.

module tb_sr_accelerator;
    localparam PE = 4;
    localparam AM_LAYERS = 2;
    localparam LM_LAYERS = 2;
    localparam AM_IN = 3;
    localparam LM_IN = 5;
    localparam VOCAB = 5;
    localparam N_BEAMS = 2;
    localparam W_W = 6;
    localparam B_W = 24;
    localparam AM_X_SHIFT = 2;
    localparam LM_X_SHIFT = 7;
    localparam S_W = 8;

    localparam AM_WEIGHT_ROWS = (AM_IN + PE) * 2 + (AM_LAYERS - 1) * PE * 4;
    localparam WEIGHT_DEPTH = AM_WEIGHT_ROWS + (LM_IN + PE) * 2 + (LM_LAYERS - 1) * PE * 4;
    localparam AM_CTX = AM_LAYERS * PE;
    localparam LM_CTX = LM_LAYERS * PE;
    localparam CONTEXT_DEPTH = AM_CTX + N_BEAMS * LM_CTX;
    localparam N_LSTM = AM_LAYERS + LM_LAYERS;
    localparam WEIGHT_AW = $clog2(WEIGHT_DEPTH);
    localparam CONTEXT_AW = $clog2(CONTEXT_DEPTH);
    localparam BIAS_AW = $clog2(N_LSTM * 2);
    localparam FRAME_AW = $clog2(AM_IN);
    localparam BEAM_W = $clog2(N_BEAMS);
    localparam CHAR_W = $clog2(VOCAB);
    localparam SCORE_AW = $clog2(VOCAB);
    localparam OUT_AW = $clog2(PE * 2);
    localparam MAX_IN = (AM_IN > LM_IN) ? AM_IN : ((LM_IN > PE) ? LM_IN : PE);

    localparam JOB_AM = 2'd0;
    localparam JOB_LM = 2'd1;
    localparam JOB_CLEAR_AM = 2'd2;
    localparam JOB_CLEAR_BEAM = 2'd3;

    reg clk = 0;
    reg rst = 1;
    reg start = 0;
    reg [1:0] job = 0;
    reg [BEAM_W-1:0] beam_id = 0;
    reg [CHAR_W-1:0] lm_char = 0;
    reg frame_we = 0;
    reg [FRAME_AW-1:0] frame_addr = 0;
    reg signed [7:0] frame_data = 0;
    reg [CONTEXT_AW-1:0] host_context_addr = 0;
    reg host_weight0_we = 0;
    reg host_weight1_we = 0;
    reg [WEIGHT_AW-1:0] host_weight_addr = 0;
    reg [PE*W_W-1:0] host_weight0_din = 0;
    reg [PE*W_W-1:0] host_weight1_din = 0;
    reg host_bias_we = 0;
    reg [BIAS_AW-1:0] host_bias_addr = 0;
    reg [PE*B_W-1:0] host_bias0_din = 0;
    reg [PE*B_W-1:0] host_bias1_din = 0;
    reg host_out_weight_we = 0;
    reg [OUT_AW-1:0] host_out_weight_addr = 0;
    reg [VOCAB*W_W-1:0] host_out_weight_din = 0;
    reg host_out_bias_we = 0;
    reg host_out_bias_addr = 0;
    reg [VOCAB*B_W-1:0] host_out_bias_din = 0;

    wire busy, done, error, score_valid;
    wire [3:0] state_out;
    wire [SCORE_AW-1:0] score_idx;
    wire signed [S_W-1:0] score;
    wire signed [VOCAB*S_W-1:0] scores;
    wire [23:0] context_dout;

    // Software copies of every memory.
    reg [PE*W_W-1:0] wm0 [0:WEIGHT_DEPTH-1];
    reg [PE*W_W-1:0] wm1 [0:WEIGHT_DEPTH-1];
    reg [PE*B_W-1:0] bm0 [0:2*N_LSTM-1];
    reg [PE*B_W-1:0] bm1 [0:2*N_LSTM-1];
    reg [VOCAB*W_W-1:0] owm [0:2*PE-1];
    reg [VOCAB*B_W-1:0] obm [0:1];
    reg signed [7:0] frame_m [0:AM_IN-1];
    reg [23:0] ctxm [0:CONTEXT_DEPTH-1];

    reg signed [15:0] xin [0:MAX_IN-1];
    reg signed [23:0] acc0 [0:PE-1];
    reg signed [23:0] acc1 [0:PE-1];
    reg signed [15:0] gi [0:PE-1];
    reg signed [15:0] gf [0:PE-1];
    reg signed [15:0] go [0:PE-1];
    reg signed [15:0] gc [0:PE-1];
    reg signed [23:0] oacc;
    reg signed [S_W-1:0] exp_score [0:VOCAB-1];
    reg signed [S_W-1:0] got_score [0:7];

    reg signed [15:0] r_c = 0;
    reg signed [63:0] r_gates = 0;
    wire signed [15:0] ref_c;
    wire signed [7:0] ref_h;

    integer errors = 0;
    integer seed = 1610;
    integer guard, score_n, bad;
    integer i, j, k, n, v, p, l;

    sr_accelerator_top #(
        .PE_COUNT(PE),
        .AM_LAYERS(AM_LAYERS),
        .LM_LAYERS(LM_LAYERS),
        .AM_IN(AM_IN),
        .LM_IN(LM_IN),
        .VOCAB(VOCAB),
        .N_BEAMS(N_BEAMS),
        .AM_X_SHIFT(AM_X_SHIFT),
        .LM_X_SHIFT(LM_X_SHIFT),
        .WEIGHT_DEPTH(WEIGHT_DEPTH),
        .CONTEXT_DEPTH(CONTEXT_DEPTH)
    ) dut (
        .clk(clk), .rst(rst), .start(start), .job(job),
        .beam_id(beam_id), .lm_char(lm_char),
        .frame_we(frame_we), .frame_addr(frame_addr), .frame_data(frame_data),
        .host_context_addr(host_context_addr),
        .host_weight0_we(host_weight0_we), .host_weight1_we(host_weight1_we),
        .host_weight_addr(host_weight_addr),
        .host_weight0_din(host_weight0_din), .host_weight1_din(host_weight1_din),
        .host_bias_we(host_bias_we), .host_bias_addr(host_bias_addr),
        .host_bias0_din(host_bias0_din), .host_bias1_din(host_bias1_din),
        .host_out_weight_we(host_out_weight_we),
        .host_out_weight_addr(host_out_weight_addr),
        .host_out_weight_din(host_out_weight_din),
        .host_out_bias_we(host_out_bias_we), .host_out_bias_addr(host_out_bias_addr),
        .host_out_bias_din(host_out_bias_din),
        .busy(busy), .done(done), .error(error), .state_out(state_out),
        .score_valid(score_valid), .score_idx(score_idx), .score(score),
        .scores(scores), .context_dout(context_dout)
    );

    lstm_epu ref_epu (
        .clk(1'b0), .rst(1'b0), .enable(1'b1),
        .old_hidden_state(r_c),
        .pe_input(r_gates),
        .hidden_state_output(ref_c),
        .output_state(ref_h)
    );

    always #5 clk = ~clk;

    always @(posedge clk) begin
        if (score_valid && score_n < 8) begin
            got_score[score_n] = score;
            score_n = score_n + 1;
        end
    end

    task check;
        input integer cond;
        input [8*64-1:0] msg;
        begin
            if (!cond) begin
                $display("FAIL: %0s", msg);
                errors = errors + 1;
            end else
                $display("pass: %0s", msg);
        end
    endtask

    task load_memories;
        begin
            for (i = 0; i < WEIGHT_DEPTH; i = i + 1) begin
                for (n = 0; n < PE; n = n + 1) begin
                    wm0[i][n*W_W +: W_W] = $random(seed);
                    wm1[i][n*W_W +: W_W] = $random(seed);
                end
                @(negedge clk);
                host_weight_addr = i;
                host_weight0_din = wm0[i];
                host_weight1_din = wm1[i];
                host_weight0_we = 1;
                host_weight1_we = 1;
            end
            @(negedge clk);
            host_weight0_we = 0;
            host_weight1_we = 0;

            for (i = 0; i < 2 * N_LSTM; i = i + 1) begin
                for (n = 0; n < PE; n = n + 1) begin
                    bm0[i][n*B_W +: B_W] = $random(seed);
                    bm1[i][n*B_W +: B_W] = $random(seed);
                end
                @(negedge clk);
                host_bias_addr = i;
                host_bias0_din = bm0[i];
                host_bias1_din = bm1[i];
                host_bias_we = 1;
            end
            @(negedge clk);
            host_bias_we = 0;

            for (i = 0; i < 2 * PE; i = i + 1) begin
                for (v = 0; v < VOCAB; v = v + 1)
                    owm[i][v*W_W +: W_W] = $random(seed);
                @(negedge clk);
                host_out_weight_addr = i;
                host_out_weight_din = owm[i];
                host_out_weight_we = 1;
            end
            @(negedge clk);
            host_out_weight_we = 0;

            for (i = 0; i < 2; i = i + 1) begin
                for (v = 0; v < VOCAB; v = v + 1)
                    obm[i][v*B_W +: B_W] = $random(seed);
                @(negedge clk);
                host_out_bias_addr = i;
                host_out_bias_din = obm[i];
                host_out_bias_we = 1;
            end
            @(negedge clk);
            host_out_bias_we = 0;
        end
    endtask

    task load_frame;
        begin
            for (i = 0; i < AM_IN; i = i + 1) begin
                frame_m[i] = $random(seed);
                @(negedge clk);
                frame_addr = i;
                frame_data = frame_m[i];
                frame_we = 1;
            end
            @(negedge clk);
            frame_we = 0;
        end
    endtask

    // The model of one AM or LM job, updating ctxm and exp_score.
    task model_job;
        input is_lm;
        input integer beam;
        input integer ch;
        integer wbase, rows_x, gl, ctx, base;
        begin
            wbase = is_lm ? AM_WEIGHT_ROWS : 0;
            for (l = 0; l < (is_lm ? LM_LAYERS : AM_LAYERS); l = l + 1) begin
                rows_x = (l == 0) ? (is_lm ? LM_IN : AM_IN) : PE;
                gl = is_lm ? AM_LAYERS + l : l;
                ctx = is_lm ? (AM_CTX + beam * LM_CTX + l * PE) : (l * PE);

                // Layer 0: frame or one-hot, shifted to h's scale.
                // Above: this step's h from below, unshifted.
                for (k = 0; k < rows_x; k = k + 1) begin
                    if (l != 0)
                        xin[k] = $signed(ctxm[ctx - PE + k][7:0]);
                    else if (is_lm)
                        xin[k] = (k == ch) ? (16'sd1 <<< LM_X_SHIFT) : 16'sd0;
                    else
                        xin[k] = frame_m[k] <<< AM_X_SHIFT;
                end

                for (p = 0; p < 2; p = p + 1) begin
                    base = wbase + p * (rows_x + PE);
                    for (n = 0; n < PE; n = n + 1) begin
                        acc0[n] = $signed(bm0[gl*2 + p][n*B_W +: B_W]);
                        acc1[n] = $signed(bm1[gl*2 + p][n*B_W +: B_W]);
                        for (k = 0; k < rows_x; k = k + 1) begin
                            acc0[n] = acc0[n] + xin[k] * $signed(wm0[base + k][n*W_W +: W_W]);
                            acc1[n] = acc1[n] + xin[k] * $signed(wm1[base + k][n*W_W +: W_W]);
                        end
                        for (k = 0; k < PE; k = k + 1) begin
                            acc0[n] = acc0[n] + $signed(ctxm[ctx + k][7:0])
                                * $signed(wm0[base + rows_x + k][n*W_W +: W_W]);
                            acc1[n] = acc1[n] + $signed(ctxm[ctx + k][7:0])
                                * $signed(wm1[base + rows_x + k][n*W_W +: W_W]);
                        end
                        if (p == 0) begin
                            gi[n] = acc0[n][15:0];
                            gf[n] = acc1[n][15:0];
                        end else begin
                            go[n] = acc0[n][15:0];
                            gc[n] = acc1[n][15:0];
                        end
                    end
                end

                // Each neuron reads and rewrites only its own entry.
                for (n = 0; n < PE; n = n + 1) begin
                    r_c = ctxm[ctx + n][23:8];
                    r_gates = {gc[n], go[n], gf[n], gi[n]};
                    #1;
                    ctxm[ctx + n] = {ref_c, ref_h};
                end
                wbase = wbase + 2 * (rows_x + PE);
            end

            // Projection of the top layer's h; ctx is still the top layer.
            for (v = 0; v < VOCAB; v = v + 1) begin
                oacc = $signed(obm[is_lm][v*B_W +: B_W]);
                for (k = 0; k < PE; k = k + 1)
                    oacc = oacc + $signed(ctxm[ctx + k][7:0])
                        * $signed(owm[(is_lm ? PE : 0) + k][v*W_W +: W_W]);
                exp_score[v] = oacc[S_W-1:0];
            end
        end
    endtask

    task run_job;
        input [1:0] which;
        input integer beam;
        input integer ch;
        begin
            score_n = 0;
            @(negedge clk);
            job = which;
            beam_id = beam;
            lm_char = ch;
            start = 1;
            @(negedge clk);
            start = 0;
            guard = 0;
            while (!done && guard < 20000) begin
                @(negedge clk);
                guard = guard + 1;
            end
            if (!(done === 1 && error === 0)) begin
                $display("FAIL: job %0d did not finish cleanly (state %0d)", which, state_out);
                errors = errors + 1;
            end
        end
    endtask

    task check_scores;
        input [8*64-1:0] msg;
        begin
            bad = (score_n != VOCAB);
            for (v = 0; v < VOCAB; v = v + 1) begin
                if (got_score[v] !== exp_score[v] || scores[v*S_W +: S_W] !== exp_score[v]) begin
                    $display("  symbol %0d: streamed %0d, bus %0d, expected %0d",
                        v, got_score[v], $signed(scores[v*S_W +: S_W]), exp_score[v]);
                    bad = 1;
                end
            end
            check(!bad, msg);
        end
    endtask

    task check_context;
        input [8*64-1:0] msg;
        begin
            bad = 0;
            for (i = 0; i < CONTEXT_DEPTH; i = i + 1) begin
                @(negedge clk);
                host_context_addr = i;
                @(negedge clk);
                if (context_dout !== ctxm[i]) begin
                    $display("  context[%0d] = %h, expected %h", i, context_dout, ctxm[i]);
                    bad = 1;
                end
            end
            check(!bad, msg);
        end
    endtask

    task clear_model;
        input integer from;
        input integer count;
        begin
            for (i = from; i < from + count; i = i + 1)
                ctxm[i] = 24'd0;
        end
    endtask

    initial begin
        repeat (4) @(negedge clk);
        rst = 0;
        @(negedge clk);
        load_memories;

        run_job(JOB_CLEAR_AM, 0, 0);
        clear_model(0, AM_CTX);
        run_job(JOB_CLEAR_BEAM, 0, 0);
        clear_model(AM_CTX, LM_CTX);
        run_job(JOB_CLEAR_BEAM, 1, 0);
        clear_model(AM_CTX + LM_CTX, LM_CTX);
        check_context("clear jobs zero all context");

        load_frame;
        model_job(0, 0, 0);
        run_job(JOB_AM, 0, 0);
        check_scores("AM frame 1 scores");
        check_context("AM frame 1 context");

        load_frame;
        model_job(0, 0, 0);
        run_job(JOB_AM, 0, 0);
        check_scores("AM frame 2 scores (uses frame 1's h)");
        check_context("AM frame 2 context");

        model_job(1, 0, 3);
        run_job(JOB_LM, 0, 3);
        check_scores("LM beam 0, char 3 scores");
        check_context("LM beam 0 context, AM and beam 1 kept");

        model_job(1, 1, 1);
        run_job(JOB_LM, 1, 1);
        check_scores("LM beam 1, char 1 scores");

        model_job(1, 0, 4);
        run_job(JOB_LM, 0, 4);
        check_scores("LM beam 0, char 4 scores (uses its own h)");
        check_context("context after three LM steps");

        run_job(JOB_CLEAR_BEAM, 0, 0);
        clear_model(AM_CTX, LM_CTX);
        check_context("clear beam 0 leaves beam 1 and the AM");

        run_job(JOB_CLEAR_AM, 0, 0);
        clear_model(0, AM_CTX);
        check_context("clear AM leaves the LM beams");

        if (errors == 0)
            $display("ALL CHECKS PASSED");
        else
            $display("%0d CHECKS FAILED", errors);
        $finish;
    end
endmodule
