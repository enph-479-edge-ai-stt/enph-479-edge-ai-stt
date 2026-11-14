`timescale 1ns / 1ps

// sr_accelerator_axi against a bare sr_accelerator_top that is loaded
// through its host ports with the same random weights, biases and frames.
// Every job runs on both, and the wrapper's streamed scores and every
// context word (read back through CTX_ADDR / CTX_DATA) must match the
// bare accelerator's. tb_sr_accelerator checks the bare accelerator
// against a model, so this test is about the wrapper: the AXI-Lite
// registers, the loader's rows for every target (the weights in two
// transfers, to cover LOAD_ADDR), the score stream, and the rules for
// when a start is taken: ignored while scores wait, held while a row is
// still being written (the second frame's job starts that way). Input
// words arrive with random gaps and the score stream sees random
// backpressure.

module tb_sr_accelerator_axi;
    localparam PE = 4;
    localparam AM_LAYERS = 2;
    localparam LM_LAYERS = 2;
    localparam AM_IN = 3;
    localparam LM_IN = 5;
    localparam VOCAB = 5;
    localparam N_BEAMS = 2;
    localparam W_W = 6;
    localparam B_W = 24;
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
    localparam OUT_AW = $clog2(PE * 2);

    localparam WT_WORDS = (2 * PE * W_W + 31) / 32;
    localparam BS_WORDS = (2 * PE * B_W + 31) / 32;
    localparam OW_WORDS = (VOCAB * W_W + 31) / 32;
    localparam OB_WORDS = (VOCAB * B_W + 31) / 32;
    localparam OUT_WORDS = (VOCAB * S_W + 31) / 32;
    localparam FRAME_WORDS = (AM_IN + 3) / 4;
    localparam ROW_BUF = 32 * 64;

    localparam A_CTRL = 6'h00;
    localparam A_JOB = 6'h04;
    localparam A_BEAM = 6'h08;
    localparam A_CHAR = 6'h0C;
    localparam A_STATUS = 6'h10;
    localparam A_LOAD_TARGET = 6'h14;
    localparam A_LOAD_ADDR = 6'h18;
    localparam A_LOAD_COUNT = 6'h1C;
    localparam A_CTX_ADDR = 6'h20;
    localparam A_CTX_DATA = 6'h24;

    localparam JOB_AM = 2'd0;
    localparam JOB_LM = 2'd1;
    localparam JOB_CLEAR_AM = 2'd2;
    localparam JOB_CLEAR_BEAM = 2'd3;

    reg clk = 0;
    reg rst_n = 0;

    // Wrapper ports.
    reg [5:0] s_axil_awaddr = 0;
    reg s_axil_awvalid = 0;
    wire s_axil_awready;
    reg [31:0] s_axil_wdata = 0;
    reg s_axil_wvalid = 0;
    wire s_axil_wready;
    wire [1:0] s_axil_bresp;
    wire s_axil_bvalid;
    reg [5:0] s_axil_araddr = 0;
    reg s_axil_arvalid = 0;
    wire s_axil_arready;
    wire [31:0] s_axil_rdata;
    wire [1:0] s_axil_rresp;
    wire s_axil_rvalid;
    reg [31:0] s_axis_tdata = 0;
    reg s_axis_tvalid = 0;
    wire s_axis_tready;
    wire [31:0] m_axis_tdata;
    wire [3:0] m_axis_tkeep;
    wire m_axis_tvalid;
    reg m_axis_tready = 0;
    wire m_axis_tlast;

    // Reference accelerator ports.
    reg r_start = 0;
    reg [1:0] r_job = 0;
    reg [BEAM_W-1:0] r_beam_id = 0;
    reg [CHAR_W-1:0] r_lm_char = 0;
    reg r_frame_we = 0;
    reg [FRAME_AW-1:0] r_frame_addr = 0;
    reg signed [7:0] r_frame_data = 0;
    reg [CONTEXT_AW-1:0] r_context_addr = 0;
    reg r_weight_we = 0;
    reg [WEIGHT_AW-1:0] r_weight_addr = 0;
    reg [PE*W_W-1:0] r_weight0_din = 0;
    reg [PE*W_W-1:0] r_weight1_din = 0;
    reg r_bias_we = 0;
    reg [BIAS_AW-1:0] r_bias_addr = 0;
    reg [PE*B_W-1:0] r_bias0_din = 0;
    reg [PE*B_W-1:0] r_bias1_din = 0;
    reg r_out_weight_we = 0;
    reg [OUT_AW-1:0] r_out_weight_addr = 0;
    reg [VOCAB*W_W-1:0] r_out_weight_din = 0;
    reg r_out_bias_we = 0;
    reg r_out_bias_addr = 0;
    reg [VOCAB*B_W-1:0] r_out_bias_din = 0;
    wire r_busy, r_done, r_error;
    wire [VOCAB*S_W-1:0] r_scores;
    wire [23:0] r_context_dout;

    // Software copies of every memory.
    reg [PE*W_W-1:0] wm0 [0:WEIGHT_DEPTH-1];
    reg [PE*W_W-1:0] wm1 [0:WEIGHT_DEPTH-1];
    reg [PE*B_W-1:0] bm0 [0:2*N_LSTM-1];
    reg [PE*B_W-1:0] bm1 [0:2*N_LSTM-1];
    reg [VOCAB*W_W-1:0] owm [0:2*PE-1];
    reg [VOCAB*B_W-1:0] obm [0:1];
    reg [7:0] frame_m [0:AM_IN-1];

    reg [ROW_BUF-1:0] row;
    reg [OUT_WORDS*32-1:0] got_scores;
    reg [OUT_WORDS*32-1:0] ref_scores;
    reg [31:0] rd;

    integer errors = 0;
    integer seed = 2610;
    integer i, k, n, bad, last_ok, guard;

    sr_accelerator_axi #(
        .PE_COUNT(PE), .AM_LAYERS(AM_LAYERS), .LM_LAYERS(LM_LAYERS),
        .AM_IN(AM_IN), .LM_IN(LM_IN), .VOCAB(VOCAB), .N_BEAMS(N_BEAMS),
        .WEIGHT_DEPTH(WEIGHT_DEPTH), .CONTEXT_DEPTH(CONTEXT_DEPTH)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .s_axil_awaddr(s_axil_awaddr), .s_axil_awvalid(s_axil_awvalid), .s_axil_awready(s_axil_awready),
        .s_axil_wdata(s_axil_wdata), .s_axil_wstrb(4'hF), .s_axil_wvalid(s_axil_wvalid),
        .s_axil_wready(s_axil_wready), .s_axil_bresp(s_axil_bresp), .s_axil_bvalid(s_axil_bvalid),
        .s_axil_bready(1'b1),
        .s_axil_araddr(s_axil_araddr), .s_axil_arvalid(s_axil_arvalid), .s_axil_arready(s_axil_arready),
        .s_axil_rdata(s_axil_rdata), .s_axil_rresp(s_axil_rresp), .s_axil_rvalid(s_axil_rvalid),
        .s_axil_rready(1'b1),
        .s_axis_tdata(s_axis_tdata), .s_axis_tvalid(s_axis_tvalid), .s_axis_tready(s_axis_tready),
        .m_axis_tdata(m_axis_tdata), .m_axis_tkeep(m_axis_tkeep), .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready), .m_axis_tlast(m_axis_tlast)
    );

    sr_accelerator_top #(
        .PE_COUNT(PE), .AM_LAYERS(AM_LAYERS), .LM_LAYERS(LM_LAYERS),
        .AM_IN(AM_IN), .LM_IN(LM_IN), .VOCAB(VOCAB), .N_BEAMS(N_BEAMS),
        .WEIGHT_DEPTH(WEIGHT_DEPTH), .CONTEXT_DEPTH(CONTEXT_DEPTH)
    ) ref_accel (
        .clk(clk), .rst(!rst_n), .start(r_start), .job(r_job),
        .beam_id(r_beam_id), .lm_char(r_lm_char),
        .frame_we(r_frame_we), .frame_addr(r_frame_addr), .frame_data(r_frame_data),
        .host_context_addr(r_context_addr),
        .host_weight0_we(r_weight_we), .host_weight1_we(r_weight_we),
        .host_weight_addr(r_weight_addr),
        .host_weight0_din(r_weight0_din), .host_weight1_din(r_weight1_din),
        .host_bias_we(r_bias_we), .host_bias_addr(r_bias_addr),
        .host_bias0_din(r_bias0_din), .host_bias1_din(r_bias1_din),
        .host_out_weight_we(r_out_weight_we), .host_out_weight_addr(r_out_weight_addr),
        .host_out_weight_din(r_out_weight_din),
        .host_out_bias_we(r_out_bias_we), .host_out_bias_addr(r_out_bias_addr),
        .host_out_bias_din(r_out_bias_din),
        .busy(r_busy), .done(r_done), .error(r_error), .state_out(),
        .score_valid(), .score_idx(), .score(), .scores(r_scores),
        .context_dout(r_context_dout)
    );

    always #5 clk = ~clk;

    initial begin
        #20000000;
        $display("FAIL: timeout");
        $finish;
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

    // ---------------- AXI-Lite master ----------------
    task axil_write;
        input [5:0] a;
        input [31:0] d;
        begin
            @(negedge clk);
            s_axil_awaddr = a;
            s_axil_wdata = d;
            s_axil_awvalid = 1;
            s_axil_wvalid = 1;
            #1;
            while (!(s_axil_awready && s_axil_wready)) begin
                @(negedge clk);
                #1;
            end
            @(negedge clk);
            s_axil_awvalid = 0;
            s_axil_wvalid = 0;
        end
    endtask

    task axil_read;
        input [5:0] a;
        output [31:0] d;
        begin
            @(negedge clk);
            s_axil_araddr = a;
            s_axil_arvalid = 1;
            #1;
            while (!s_axil_arready) begin
                @(negedge clk);
                #1;
            end
            @(negedge clk);
            s_axil_arvalid = 0;
            d = s_axil_rdata;
        end
    endtask

    // ---------------- stream in ----------------
    task send_word;
        input [31:0] w;
        begin
            @(negedge clk);
            if (($random(seed) & 3) == 0)
                @(negedge clk);
            s_axis_tdata = w;
            s_axis_tvalid = 1;
            #1;
            while (!s_axis_tready) begin
                @(negedge clk);
                #1;
            end
            @(negedge clk);
            s_axis_tvalid = 0;
        end
    endtask

    task send_row;
        input integer words;
        begin
            for (k = 0; k < words; k = k + 1)
                send_word(row[k*32 +: 32]);
        end
    endtask

    // ---------------- stream out ----------------
    task recv_scores;
        begin
            n = 0;
            last_ok = 1;
            got_scores = 0;
            guard = 0;
            while (n < OUT_WORDS && guard < 1000) begin
                @(negedge clk);
                m_axis_tready = $random(seed) & 1;
                #1;
                if (m_axis_tvalid && m_axis_tready) begin
                    got_scores[n*32 +: 32] = m_axis_tdata;
                    if (m_axis_tlast !== (n == OUT_WORDS - 1))
                        last_ok = 0;
                    n = n + 1;
                end
                guard = guard + 1;
            end
            @(negedge clk);
            m_axis_tready = 0;
        end
    endtask

    // ---------------- loading ----------------
    task make_memories;
        begin
            for (i = 0; i < WEIGHT_DEPTH; i = i + 1)
                for (n = 0; n < PE; n = n + 1) begin
                    wm0[i][n*W_W +: W_W] = $random(seed);
                    wm1[i][n*W_W +: W_W] = $random(seed);
                end
            for (i = 0; i < 2 * N_LSTM; i = i + 1)
                for (n = 0; n < PE; n = n + 1) begin
                    bm0[i][n*B_W +: B_W] = $random(seed);
                    bm1[i][n*B_W +: B_W] = $random(seed);
                end
            for (i = 0; i < 2 * PE; i = i + 1)
                for (n = 0; n < VOCAB; n = n + 1)
                    owm[i][n*W_W +: W_W] = $random(seed);
            for (i = 0; i < 2; i = i + 1)
                for (n = 0; n < VOCAB; n = n + 1)
                    obm[i][n*B_W +: B_W] = $random(seed);
        end
    endtask

    task ref_load;
        begin
            for (i = 0; i < WEIGHT_DEPTH; i = i + 1) begin
                @(negedge clk);
                r_weight_addr = i;
                r_weight0_din = wm0[i];
                r_weight1_din = wm1[i];
                r_weight_we = 1;
            end
            @(negedge clk);
            r_weight_we = 0;
            for (i = 0; i < 2 * N_LSTM; i = i + 1) begin
                @(negedge clk);
                r_bias_addr = i;
                r_bias0_din = bm0[i];
                r_bias1_din = bm1[i];
                r_bias_we = 1;
            end
            @(negedge clk);
            r_bias_we = 0;
            for (i = 0; i < 2 * PE; i = i + 1) begin
                @(negedge clk);
                r_out_weight_addr = i;
                r_out_weight_din = owm[i];
                r_out_weight_we = 1;
            end
            @(negedge clk);
            r_out_weight_we = 0;
            for (i = 0; i < 2; i = i + 1) begin
                @(negedge clk);
                r_out_bias_addr = i;
                r_out_bias_din = obm[i];
                r_out_bias_we = 1;
            end
            @(negedge clk);
            r_out_bias_we = 0;
        end
    endtask

    // The weights go in two transfers so the second starts at LOAD_ADDR.
    task dut_load;
        begin
            axil_write(A_LOAD_TARGET, 1);
            axil_write(A_LOAD_ADDR, 0);
            for (i = 0; i < WEIGHT_DEPTH / 2; i = i + 1) begin
                row = 0;
                row[2*PE*W_W-1:0] = {wm1[i], wm0[i]};
                send_row(WT_WORDS);
            end
            repeat (2) @(negedge clk);
            axil_read(A_LOAD_COUNT, rd);
            check(rd == WEIGHT_DEPTH / 2, "LOAD_COUNT after the first weight transfer");
            axil_write(A_LOAD_ADDR, WEIGHT_DEPTH / 2);
            for (i = WEIGHT_DEPTH / 2; i < WEIGHT_DEPTH; i = i + 1) begin
                row = 0;
                row[2*PE*W_W-1:0] = {wm1[i], wm0[i]};
                send_row(WT_WORDS);
            end
            repeat (2) @(negedge clk);
            axil_read(A_LOAD_COUNT, rd);
            check(rd == WEIGHT_DEPTH - WEIGHT_DEPTH / 2, "LOAD_COUNT after the second, from LOAD_ADDR");

            axil_write(A_LOAD_TARGET, 2);
            axil_write(A_LOAD_ADDR, 0);
            for (i = 0; i < 2 * N_LSTM; i = i + 1) begin
                row = 0;
                row[2*PE*B_W-1:0] = {bm1[i], bm0[i]};
                send_row(BS_WORDS);
            end

            axil_write(A_LOAD_TARGET, 3);
            axil_write(A_LOAD_ADDR, 0);
            for (i = 0; i < 2 * PE; i = i + 1) begin
                row = 0;
                row[VOCAB*W_W-1:0] = owm[i];
                send_row(OW_WORDS);
            end

            axil_write(A_LOAD_TARGET, 4);
            axil_write(A_LOAD_ADDR, 0);
            for (i = 0; i < 2; i = i + 1) begin
                row = 0;
                row[VOCAB*B_W-1:0] = obm[i];
                send_row(OB_WORDS);
            end
            repeat (2) @(negedge clk);
            axil_read(A_LOAD_COUNT, rd);
            check(rd == 2, "LOAD_COUNT after the output biases");
        end
    endtask

    // A new random frame into the bare accelerator; row holds it for the
    // wrapper, with a byte past the frame that must be dropped.
    task ref_frame;
        begin
            for (i = 0; i < AM_IN; i = i + 1)
                frame_m[i] = $random(seed);
            for (i = 0; i < AM_IN; i = i + 1) begin
                @(negedge clk);
                r_frame_addr = i;
                r_frame_data = frame_m[i];
                r_frame_we = 1;
            end
            @(negedge clk);
            r_frame_we = 0;
            row = 0;
            for (i = 0; i < AM_IN; i = i + 1)
                row[8*i +: 8] = frame_m[i];
            row[8*AM_IN +: 8] = 8'hA5;
        end
    endtask

    task dut_frame;
        begin
            axil_write(A_LOAD_TARGET, 0);
            axil_write(A_LOAD_ADDR, 0);
            send_row(FRAME_WORDS);
            repeat (5) @(negedge clk);
            axil_read(A_LOAD_COUNT, rd);
            check(rd == AM_IN, "frame: LOAD_COUNT counts only real features");
        end
    endtask

    // ---------------- jobs ----------------
    task ref_job;
        input [1:0] which;
        input integer beam;
        input integer ch;
        begin
            @(negedge clk);
            r_job = which;
            r_beam_id = beam;
            r_lm_char = ch;
            r_start = 1;
            @(negedge clk);
            r_start = 0;
            guard = 0;
            while (!r_done && guard < 20000) begin
                @(negedge clk);
                guard = guard + 1;
            end
            ref_scores = r_scores;
        end
    endtask

    task dut_setup;
        input [1:0] which;
        input integer beam;
        input integer ch;
        begin
            axil_write(A_JOB, which);
            axil_write(A_BEAM, beam);
            axil_write(A_CHAR, ch);
        end
    endtask

    // After CTRL: the stream must be held off, then wait for done and take
    // the scores.
    task dut_finish;
        input [1:0] which;
        input [8*48-1:0] name;
        begin
            bad = 0;
            for (k = 0; k < 3; k = k + 1) begin
                @(negedge clk);
                #1;
                if (s_axis_tready)
                    bad = 1;
            end
            guard = 0;
            rd = 0;
            while (!rd[1] && guard < 20000) begin
                axil_read(A_STATUS, rd);
                guard = guard + 1;
            end
            if (which == JOB_AM || which == JOB_LM) begin
                check(rd[3] == 1, "scores waiting after the job");
                check(!bad, "input stream held off while the job runs");
                // A start while the scores wait must be ignored.
                axil_write(A_CTRL, 1);
                axil_read(A_STATUS, rd);
                check(rd[0] == 0, "start ignored while scores wait");
                recv_scores;
                check(n == OUT_WORDS && last_ok, "score stream: word count and tlast");
                check(got_scores[VOCAB*S_W-1:0] === ref_scores[VOCAB*S_W-1:0], name);
                check(got_scores[OUT_WORDS*32-1:VOCAB*S_W] == 0, "score padding is zero");
                axil_read(A_STATUS, rd);
                check(rd[3] == 0, "scores taken");
            end else begin
                check(rd[3] == 0, "clear job sends no scores");
            end
        end
    endtask

    task dut_job;
        input [1:0] which;
        input integer beam;
        input integer ch;
        input [8*48-1:0] name;
        begin
            dut_setup(which, beam, ch);
            axil_write(A_CTRL, 1);
            dut_finish(which, name);
        end
    endtask

    task check_context;
        input [8*48-1:0] name;
        begin
            bad = 0;
            for (i = 0; i < CONTEXT_DEPTH; i = i + 1) begin
                axil_write(A_CTX_ADDR, i);
                axil_read(A_CTX_DATA, rd);
                @(negedge clk);
                r_context_addr = i;
                @(negedge clk);
                if (rd[23:0] !== r_context_dout) begin
                    $display("  context[%0d]: wrapper %h, bare %h", i, rd[23:0], r_context_dout);
                    bad = 1;
                end
            end
            check(!bad, name);
        end
    endtask

    task both;
        input [1:0] which;
        input integer beam;
        input integer ch;
        input [8*48-1:0] name;
        begin
            ref_job(which, beam, ch);
            dut_job(which, beam, ch, name);
        end
    endtask

    initial begin
        repeat (4) @(negedge clk);
        rst_n = 1;

        axil_read(A_STATUS, rd);
        check(rd == 0, "STATUS is idle after reset");
        axil_write(A_BEAM, 1);
        axil_read(A_BEAM, rd);
        check(rd == 1, "BEAM reads back");

        make_memories;
        ref_load;
        dut_load;

        both(JOB_CLEAR_AM, 0, 0, "clear AM");
        both(JOB_CLEAR_BEAM, 0, 0, "clear beam 0");
        both(JOB_CLEAR_BEAM, 1, 0, "clear beam 1");
        check_context("context matches after the clears");

        ref_frame;
        dut_frame;
        both(JOB_AM, 0, 0, "AM frame 1 scores match");

        // Frame 2: CTRL is taken on the same clock as the frame's last word,
        // so the frame's bytes are still to be written; the start must wait
        // for them or the accelerator drops the later ones.
        ref_frame;
        ref_job(JOB_AM, 0, 0);
        dut_setup(JOB_AM, 0, 0);
        axil_write(A_LOAD_TARGET, 0);
        axil_write(A_LOAD_ADDR, 0);
        fork
            send_row(FRAME_WORDS);
            begin
                @(negedge clk);
                #2;
                while (!(s_axis_tvalid && s_axis_tready)) begin
                    @(negedge clk);
                    #2;
                end
                s_axil_awaddr = A_CTRL;
                s_axil_wdata = 1;
                s_axil_awvalid = 1;
                s_axil_wvalid = 1;
                #1;
                if (!(s_axil_awready && s_axil_wready))
                    check(0, "CTRL write lined up with the last frame word");
                @(negedge clk);
                s_axil_awvalid = 0;
                s_axil_wvalid = 0;
            end
        join
        axil_read(A_STATUS, rd);
        check(rd[0] == 1 || rd[1] == 1, "start during the frame write was taken");
        dut_finish(JOB_AM, "AM frame 2 scores match (start was held)");
        both(JOB_LM, 0, 3, "LM beam 0, char 3 scores match");
        both(JOB_LM, 1, 1, "LM beam 1, char 1 scores match");
        both(JOB_LM, 0, 4, "LM beam 0, char 4 scores match");
        check_context("context matches after the AM and LM jobs");

        both(JOB_CLEAR_BEAM, 0, 0, "clear beam 0 again");
        check_context("context matches after clearing beam 0");

        if (errors == 0)
            $display("ALL CHECKS PASSED");
        else
            $display("%0d CHECKS FAILED", errors);
        $finish;
    end
endmodule
