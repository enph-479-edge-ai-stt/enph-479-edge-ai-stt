`timescale 1ns / 1ps

// Tiny AM job, then an LM job on another context base.
// Weights are zero, so each gate result is the bias and the projection
// score is the output bias. The LM job must leave the AM context alone.
//
// From hardware/rtl:
//   iverilog -g2012 -o /tmp/tb_sr.vvp ../testbenches/tb_sr_accelerator.v \
//     pe_unit.v pe_array.v pe_buffer.v weight_bram.v context_memory.v \
//     sigmoid_lut.v tanh_lut.v lstm_epu.v lstm_tile.v output_tile.v \
//     fsm_controller.v sr_accelerator_top.v
//   vvp /tmp/tb_sr.vvp

module tb_sr_accelerator;
    localparam PE_COUNT = 2;
    localparam AM_LAYERS = 1;
    localparam LM_LAYERS = 1;
    localparam AM_IN = 2;
    localparam LM_IN = 2;
    localparam VOCAB = 2;
    localparam N_BEAMS = 2;
    localparam X_WIDTH = 8;
    localparam W_WIDTH = 6;
    localparam BIAS_WIDTH = 16;
    localparam SCORE_WIDTH = 8;
    localparam WEIGHT_DEPTH = ((AM_IN + PE_COUNT) * 2) + ((LM_IN + PE_COUNT) * 2);
    localparam CONTEXT_DEPTH = (AM_LAYERS * PE_COUNT) + (N_BEAMS * LM_LAYERS * PE_COUNT);
    localparam N_LSTM = AM_LAYERS + LM_LAYERS;
    localparam WEIGHT_AW = $clog2(WEIGHT_DEPTH);
    localparam CONTEXT_AW = $clog2(CONTEXT_DEPTH);
    localparam BIAS_AW = $clog2(N_LSTM * 2);
    localparam PEEP_AW = $clog2(N_LSTM * PE_COUNT);
    localparam FRAME_AW = $clog2(AM_IN);
    localparam BEAM_W = $clog2(N_BEAMS);
    localparam CHAR_W = $clog2(VOCAB);
    localparam SCORE_AW = $clog2(VOCAB);
    localparam OUT_AW = $clog2(PE_COUNT * 2);
    localparam PEEP_N = N_LSTM * PE_COUNT;
    localparam BIAS_N = N_LSTM * 2;

    localparam JOB_AM = 2'd0;
    localparam JOB_LM = 2'd1;
    localparam JOB_CLEAR_AM = 2'd2;
    localparam JOB_CLEAR_BEAM = 2'd3;

    localparam signed [15:0] GATE_BIAS = 16'h0020;
    localparam [23:0] AM_CTX_WORD = {16'hFF81, 8'h7F};

    reg clk = 0;
    reg rst = 1;
    reg start = 0;
    reg [1:0] job = 0;
    reg [BEAM_W-1:0] beam_id = 0;
    reg [CHAR_W-1:0] lm_char = 0;
    reg frame_we = 0;
    reg [FRAME_AW-1:0] frame_addr = 0;
    reg signed [X_WIDTH-1:0] frame_data = 0;
    reg [CONTEXT_AW-1:0] host_context_addr = 0;
    reg host_weight0_we = 0;
    reg host_weight1_we = 0;
    reg [WEIGHT_AW-1:0] host_weight_addr = 0;
    reg [PE_COUNT*W_WIDTH-1:0] host_weight0_din = 0;
    reg [PE_COUNT*W_WIDTH-1:0] host_weight1_din = 0;
    reg host_bias_we = 0;
    reg [BIAS_AW-1:0] host_bias_addr = 0;
    reg signed [PE_COUNT*BIAS_WIDTH-1:0] host_bias0_din = 0;
    reg signed [PE_COUNT*BIAS_WIDTH-1:0] host_bias1_din = 0;
    reg host_peep_we = 0;
    reg [PEEP_AW-1:0] host_peep_addr = 0;
    reg signed [23:0] host_peep_din = 0;
    reg host_out_weight_we = 0;
    reg [OUT_AW-1:0] host_out_weight_addr = 0;
    reg [VOCAB*W_WIDTH-1:0] host_out_weight_din = 0;
    reg host_out_bias_we = 0;
    reg host_out_bias_addr = 0;
    reg signed [VOCAB*BIAS_WIDTH-1:0] host_out_bias_din = 0;

    wire busy, done, error, score_valid;
    wire [3:0] state_out;
    wire [SCORE_AW-1:0] score_idx;
    wire signed [SCORE_WIDTH-1:0] score;
    wire signed [VOCAB*SCORE_WIDTH-1:0] scores;
    wire [23:0] context_dout;

    integer errors = 0;
    integer guard = 0;
    integer score_n = 0;
    reg signed [SCORE_WIDTH-1:0] got_score [0:7];

    always #5 clk = ~clk;

    sr_accelerator_top #(
        .PE_COUNT(PE_COUNT),
        .AM_LAYERS(AM_LAYERS),
        .LM_LAYERS(LM_LAYERS),
        .AM_IN(AM_IN),
        .LM_IN(LM_IN),
        .VOCAB(VOCAB),
        .N_BEAMS(N_BEAMS),
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
        .host_peep_we(host_peep_we), .host_peep_addr(host_peep_addr),
        .host_peep_din(host_peep_din),
        .host_out_weight_we(host_out_weight_we),
        .host_out_weight_addr(host_out_weight_addr),
        .host_out_weight_din(host_out_weight_din),
        .host_out_bias_we(host_out_bias_we), .host_out_bias_addr(host_out_bias_addr),
        .host_out_bias_din(host_out_bias_din),
        .busy(busy), .done(done), .error(error), .state_out(state_out),
        .score_valid(score_valid), .score_idx(score_idx), .score(score),
        .scores(scores), .context_dout(context_dout)
    );

    always @(posedge clk) begin
        if (score_valid && score_n < 8) begin
            got_score[score_n] = score;
            score_n = score_n + 1;
        end
    end

    task check;
        input integer cond;
        input [255:0] msg;
        begin
            if (!cond) begin
                $display("FAIL: %0s (state %0d)", msg, state_out);
                errors = errors + 1;
            end
        end
    endtask

    task write_zeros;
        integer i;
        begin
            for (i = 0; i < WEIGHT_DEPTH; i = i + 1) begin
                @(negedge clk);
                host_weight_addr = i;
                host_weight0_din = 0;
                host_weight1_din = 0;
                host_weight0_we = 1;
                host_weight1_we = 1;
            end
            @(negedge clk);
            host_weight0_we = 0;
            host_weight1_we = 0;
            for (i = 0; i < PEEP_N; i = i + 1) begin
                @(negedge clk);
                host_peep_addr = i;
                host_peep_din = 0;
                host_peep_we = 1;
            end
            @(negedge clk);
            host_peep_we = 0;
            for (i = 0; i < PE_COUNT * 2; i = i + 1) begin
                @(negedge clk);
                host_out_weight_addr = i;
                host_out_weight_din = 0;
                host_out_weight_we = 1;
            end
            @(negedge clk);
            host_out_weight_we = 0;
            for (i = 0; i < AM_IN; i = i + 1) begin
                @(negedge clk);
                frame_addr = i;
                frame_data = 0;
                frame_we = 1;
            end
            @(negedge clk);
            frame_we = 0;
        end
    endtask

    task write_bias;
        input [BIAS_AW-1:0] addr;
        input signed [15:0] value;
        begin
            @(negedge clk);
            host_bias_addr = addr;
            host_bias0_din = {value, value};
            host_bias1_din = {value, value};
            host_bias_we = 1;
            @(negedge clk);
            host_bias_we = 0;
        end
    endtask

    task write_out_bias;
        input addr;
        input signed [15:0] value;
        begin
            @(negedge clk);
            host_out_bias_addr = addr;
            host_out_bias_din = {value, value};
            host_out_bias_we = 1;
            @(negedge clk);
            host_out_bias_we = 0;
        end
    endtask

    task run_job;
        input [1:0] which;
        input [BEAM_W-1:0] beam;
        input [CHAR_W-1:0] ch;
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
            while (!done && guard < 5000) begin
                @(negedge clk);
                guard = guard + 1;
            end
            check(done === 1, "job finished");
            check(error === 0, "job not flagged");
        end
    endtask

    task read_context;
        input [CONTEXT_AW-1:0] addr;
        output [23:0] value;
        begin
            @(negedge clk);
            host_context_addr = addr;
            @(negedge clk);
            value = context_dout;
        end
    endtask

    reg [23:0] ctx0, ctx1, ctx2, ctx3;

    initial begin
        host_bias0_din = 0;
        host_bias1_din = 0;
        repeat (4) @(negedge clk);
        rst = 0;
        @(negedge clk);

        write_zeros;
        write_bias(0, GATE_BIAS);
        write_bias(1, GATE_BIAS);
        write_bias(2, 16'sd0);
        write_bias(3, 16'sd0);
        write_out_bias(0, 16'sd5);
        write_out_bias(1, 16'sd9);

        run_job(JOB_CLEAR_AM, 0, 0);
        run_job(JOB_CLEAR_BEAM, 0, 0);

        run_job(JOB_AM, 0, 0);
        check(score_n == 2, "AM emitted two scores");
        check(got_score[0] === 8'sd5, "AM score 0 is the projection bias");
        check(got_score[1] === 8'sd5, "AM score 1 is the projection bias");

        read_context(0, ctx0);
        read_context(1, ctx1);
        $display("AM context 0 = %h", ctx0);
        $display("AM context 1 = %h", ctx1);
        check(ctx0 === AM_CTX_WORD, "AM neuron 0 context");
        check(ctx1 === AM_CTX_WORD, "AM neuron 1 context");

        run_job(JOB_LM, 0, 0);
        check(score_n == 2, "LM emitted two scores");
        check(got_score[0] === 8'sd9, "LM score 0 is the other projection bias");
        check(got_score[1] === 8'sd9, "LM score 1 is the other projection bias");

        read_context(0, ctx0);
        read_context(1, ctx1);
        read_context(2, ctx2);
        read_context(3, ctx3);
        check(ctx0 === AM_CTX_WORD, "LM job left AM neuron 0");
        check(ctx1 === AM_CTX_WORD, "LM job left AM neuron 1");
        check(ctx2 === 24'd0, "LM neuron 0 context");
        check(ctx3 === 24'd0, "LM neuron 1 context");

        if (errors == 0)
            $display("ALL CHECKS PASSED");
        else
            $display("%0d CHECKS FAILED", errors);
        $finish;
    end
endmodule
