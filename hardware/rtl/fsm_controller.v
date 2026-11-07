`timescale 1ns / 1ps

//=====================================================================
// Module: fsm_controller.v
// Description:
//  Sequences one LSTM tile and one output tile for two networks.
//  A job is accepted only from IDLE. AM, LM, and the clear jobs never
//  preempt each other. Switching networks is a different weight base
//  and context base; the datapath is not reconfigured.
//
//  Weight, bias, peephole, context, and projection reads are one
//  cycle late. pe_enable and the output-tile enable are asserted on
//  the cycle that data is valid, one step after the address is issued.
//  The EPU is two cycles per neuron: present the context and peephole
//  address, then write {c_t, h_t} on the next cycle with that same
//  address still held.
//
//  Layer 0 input is the frame (AM) or a one-hot character (LM).
//  Later layers read the previous layer's h from context. Every layer
//  then MACs h_{t-1} from that layer's own context.
//
// Jobs:
//  0 AM          3 layers, 123-d frame, context base 0
//  1 LM          2 layers, one-hot lm_char, context base from beam_id
//  2 CLEAR_AM    zero the acoustic context
//  3 CLEAR_BEAM  zero one beam's character-LM context
//=====================================================================

module fsm_controller #(
    parameter PE_COUNT = 256,
    parameter AM_LAYERS = 3,
    parameter LM_LAYERS = 2,
    parameter AM_IN = 123,
    parameter LM_IN = 30,
    parameter VOCAB = 30,
    parameter N_BEAMS = 128,
    parameter X_WIDTH = 8,
    parameter SCORE_WIDTH = 8,
    parameter MAX_DIM = (AM_IN > PE_COUNT)
        ? ((AM_IN > LM_IN) ? AM_IN : LM_IN)
        : ((PE_COUNT > LM_IN) ? PE_COUNT : LM_IN),
    parameter STEP_W = $clog2(MAX_DIM + 3),
    parameter WEIGHT_DEPTH =
        ((AM_IN + PE_COUNT) * 2)
        + ((AM_LAYERS > 1) ? (AM_LAYERS - 1) : 0) * (PE_COUNT * 4)
        + ((LM_IN + PE_COUNT) * 2)
        + ((LM_LAYERS > 1) ? (LM_LAYERS - 1) : 0) * (PE_COUNT * 4),
    parameter CONTEXT_DEPTH = (AM_LAYERS * PE_COUNT) + (N_BEAMS * LM_LAYERS * PE_COUNT),
    parameter WEIGHT_AW = $clog2(WEIGHT_DEPTH),
    parameter CONTEXT_AW = $clog2(CONTEXT_DEPTH),
    parameter N_LSTM_LAYERS = AM_LAYERS + LM_LAYERS,
    parameter BIAS_AW = $clog2(N_LSTM_LAYERS * 2),
    parameter PEEP_AW = $clog2(N_LSTM_LAYERS * PE_COUNT),
    parameter NEURON_AW = (PE_COUNT <= 1) ? 1 : $clog2(PE_COUNT),
    parameter BEAM_W = (N_BEAMS <= 1) ? 1 : $clog2(N_BEAMS),
    parameter CHAR_W = (VOCAB <= 1) ? 1 : $clog2(VOCAB),
    parameter SCORE_AW = (VOCAB <= 1) ? 1 : $clog2(VOCAB),
    parameter OUT_AW = $clog2(PE_COUNT * 2),
    parameter OUT_BIAS_AW = 1
) (
    input wire clk,
    input wire rst,

    input wire start,
    input wire [1:0] job,
    input wire [BEAM_W-1:0] beam_id,
    input wire [CHAR_W-1:0] lm_char,
    input wire signed [X_WIDTH-1:0] frame_elem,

    output reg idle,
    output reg busy,
    output reg done,
    output reg error,

    output reg pe_enable,
    output reg load_bias,
    output reg sel_in,
    output reg signed [X_WIDTH-1:0] d_in,
    output reg [WEIGHT_AW-1:0] weight_addr,
    output reg [BIAS_AW-1:0] bias_addr,
    output reg capture_gates,
    output reg gate_pair,
    output reg [NEURON_AW-1:0] neuron_addr,
    output reg [PEEP_AW-1:0] peep_addr,
    output reg [CONTEXT_AW-1:0] context_addr,
    output reg context_we,
    output reg context_use_ext,
    output reg [STEP_W-1:0] x_idx,

    output reg out_enable,
    output reg out_load_bias,
    output reg [OUT_AW-1:0] out_weight_addr,
    output reg [OUT_BIAS_AW-1:0] out_bias_addr,
    output reg score_valid,
    output reg [SCORE_AW-1:0] score_idx,

    output wire [3:0] state_out
);

    localparam JOB_AM = 2'd0;
    localparam JOB_LM = 2'd1;
    localparam JOB_CLEAR_AM = 2'd2;
    localparam JOB_CLEAR_BEAM = 2'd3;

    localparam S_IDLE      = 4'd0;
    localparam S_BIAS_ADDR = 4'd1;
    localparam S_BIAS_LOAD = 4'd2;
    localparam S_MAC_X     = 4'd3;
    localparam S_MAC_H     = 4'd4;
    localparam S_CAPTURE   = 4'd5;
    localparam S_EPU_RD    = 4'd6;
    localparam S_EPU_WR    = 4'd7;
    localparam S_OUT_ADDR  = 4'd8;
    localparam S_OUT_LOAD  = 4'd9;
    localparam S_OUT_MAC   = 4'd10;
    localparam S_OUT_EMIT  = 4'd11;
    localparam S_CLEAR     = 4'd12;
    localparam S_DONE      = 4'd13;

    localparam AM_CTX = AM_LAYERS * PE_COUNT;
    localparam LM_CTX = LM_LAYERS * PE_COUNT;
    localparam LAST_SCORE = VOCAB - 1;
    localparam AM_L0_ROWS = (AM_IN + PE_COUNT) * 2;
    localparam HID_ROWS = PE_COUNT * 4;
    localparam AM_WEIGHT_ROWS = AM_L0_ROWS + ((AM_LAYERS > 1) ? (AM_LAYERS - 1) : 0) * HID_ROWS;

    reg [3:0] state;
    reg is_lm;
    reg [BEAM_W-1:0] beam_q;
    reg [CHAR_W-1:0] char_q;
    reg [3:0] layer;
    reg pair;
    reg [CONTEXT_AW-1:0] n;
    reg [STEP_W-1:0] mac_step;
    reg [STEP_W-1:0] mac_len_r;
    reg [WEIGHT_AW-1:0] mac_base_r;
    reg [WEIGHT_AW-1:0] layer_wbase;
    reg [CONTEXT_AW-1:0] clear_base;
    reg [CONTEXT_AW-1:0] clear_count;
    reg signed [X_WIDTH-1:0] d_pipe;

    assign state_out = state;

    wire [3:0] n_layers = is_lm ? LM_LAYERS[3:0] : AM_LAYERS[3:0];
    wire [STEP_W-1:0] rows_x = (layer == 0)
        ? (is_lm ? LM_IN[STEP_W-1:0] : AM_IN[STEP_W-1:0])
        : PE_COUNT[STEP_W-1:0];
    wire [WEIGHT_AW-1:0] pair_span = rows_x + PE_COUNT[WEIGHT_AW-1:0];
    wire [WEIGHT_AW-1:0] x_base = layer_wbase + (pair ? pair_span : {WEIGHT_AW{1'b0}});
    wire [WEIGHT_AW-1:0] h_base = x_base + rows_x;

    wire [3:0] global_layer = is_lm ? (AM_LAYERS[3:0] + layer) : layer;
    wire [CONTEXT_AW-1:0] cur_ctx = is_lm
        ? (AM_CTX + beam_q * LM_CTX + layer * PE_COUNT)
        : (layer * PE_COUNT);
    wire [STEP_W-1:0] issue_off = (mac_step < mac_len_r) ? mac_step : (mac_len_r - 1'b1);
    wire mac_fire = (mac_step != 0) && (mac_step <= mac_len_r);
    wire signed [X_WIDTH-1:0] layer0_elem = is_lm
        ? ((mac_step == char_q) ? 8'sd1 : 8'sd0)
        : frame_elem;

    always @(*) begin
        idle = (state == S_IDLE) || (state == S_DONE);
        busy = (state != S_IDLE) && (state != S_DONE);
        done = (state == S_DONE);
        pe_enable = 0;
        load_bias = 0;
        sel_in = 0;
        d_in = d_pipe;
        weight_addr = {WEIGHT_AW{1'b0}};
        bias_addr = {BIAS_AW{1'b0}};
        capture_gates = 0;
        gate_pair = pair;
        neuron_addr = {NEURON_AW{1'b0}};
        peep_addr = {PEEP_AW{1'b0}};
        context_addr = {CONTEXT_AW{1'b0}};
        context_we = 0;
        context_use_ext = 0;
        x_idx = mac_step;
        out_enable = 0;
        out_load_bias = 0;
        out_weight_addr = {OUT_AW{1'b0}};
        out_bias_addr = is_lm ? 1'b1 : 1'b0;
        score_valid = 0;

        case (state)
            S_BIAS_ADDR, S_BIAS_LOAD: begin
                bias_addr = global_layer * 2 + pair;
                load_bias = (state == S_BIAS_LOAD);
            end
            S_MAC_X, S_MAC_H: begin
                pe_enable = mac_fire;
                sel_in = (state == S_MAC_H) || (layer != 0);
                weight_addr = mac_base_r + issue_off;
                if (sel_in)
                    context_addr = ((state == S_MAC_X) ? (cur_ctx - PE_COUNT) : cur_ctx) + issue_off;
            end
            S_CAPTURE: begin
                capture_gates = 1;
            end
            S_EPU_RD, S_EPU_WR: begin
                context_addr = cur_ctx + n;
                peep_addr = global_layer * PE_COUNT + n;
                neuron_addr = n[NEURON_AW-1:0];
                context_we = (state == S_EPU_WR);
            end
            S_OUT_ADDR, S_OUT_LOAD: begin
                out_load_bias = (state == S_OUT_LOAD);
            end
            S_OUT_MAC: begin
                out_enable = mac_fire;
                out_weight_addr = mac_base_r + issue_off;
                context_addr = cur_ctx + issue_off;
            end
            S_OUT_EMIT: begin
                score_valid = 1;
            end
            S_CLEAR: begin
                context_addr = clear_base + n;
                context_we = 1;
                context_use_ext = 1;
            end
            default: begin
            end
        endcase
    end

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            is_lm <= 0;
            beam_q <= 0;
            char_q <= 0;
            layer <= 0;
            pair <= 0;
            n <= 0;
            mac_step <= 0;
            mac_len_r <= 0;
            mac_base_r <= 0;
            layer_wbase <= 0;
            clear_base <= 0;
            clear_count <= 0;
            d_pipe <= 0;
            score_idx <= 0;
            error <= 0;
        end else begin
            case (state)
                S_IDLE: begin
                    if (start) begin
                        beam_q <= beam_id;
                        char_q <= lm_char;
                        layer <= 0;
                        pair <= 0;
                        n <= 0;
                        mac_step <= 0;
                        score_idx <= 0;
                        error <= 0;
                        if (job == JOB_AM) begin
                            is_lm <= 0;
                            layer_wbase <= 0;
                            state <= S_BIAS_ADDR;
                        end else if (job == JOB_LM) begin
                            is_lm <= 1;
                            layer_wbase <= AM_WEIGHT_ROWS[WEIGHT_AW-1:0];
                            state <= S_BIAS_ADDR;
                        end else if (job == JOB_CLEAR_AM) begin
                            clear_base <= 0;
                            clear_count <= AM_CTX[CONTEXT_AW-1:0];
                            state <= S_CLEAR;
                        end else if (job == JOB_CLEAR_BEAM) begin
                            clear_base <= AM_CTX + beam_id * LM_CTX;
                            clear_count <= LM_CTX[CONTEXT_AW-1:0];
                            state <= S_CLEAR;
                        end else begin
                            error <= 1;
                            state <= S_DONE;
                        end
                    end
                end
                S_BIAS_ADDR: begin
                    state <= S_BIAS_LOAD;
                end
                S_BIAS_LOAD: begin
                    state <= S_MAC_X;
                    mac_step <= 0;
                    mac_len_r <= rows_x;
                    mac_base_r <= x_base;
                end
                S_MAC_X: begin
                    if (mac_step == mac_len_r + 1'b1) begin
                        state <= S_MAC_H;
                        mac_step <= 0;
                        mac_len_r <= PE_COUNT[STEP_W-1:0];
                        mac_base_r <= h_base;
                    end else begin
                        if ((layer == 0) && (mac_step < mac_len_r))
                            d_pipe <= layer0_elem;
                        mac_step <= mac_step + 1'b1;
                    end
                end
                S_MAC_H: begin
                    if (mac_step == mac_len_r + 1'b1)
                        state <= S_CAPTURE;
                    else
                        mac_step <= mac_step + 1'b1;
                end
                S_CAPTURE: begin
                    if (pair == 1'b0) begin
                        pair <= 1'b1;
                        state <= S_BIAS_ADDR;
                    end else begin
                        pair <= 1'b0;
                        n <= 0;
                        state <= S_EPU_RD;
                    end
                end
                S_EPU_RD: begin
                    state <= S_EPU_WR;
                end
                S_EPU_WR: begin
                    if (n == PE_COUNT - 1) begin
                        if (layer + 1'b1 == n_layers) begin
                            state <= S_OUT_ADDR;
                        end else begin
                            layer_wbase <= layer_wbase + {pair_span, 1'b0};
                            layer <= layer + 1'b1;
                            pair <= 1'b0;
                            state <= S_BIAS_ADDR;
                        end
                    end else begin
                        n <= n + 1'b1;
                        state <= S_EPU_RD;
                    end
                end
                S_OUT_ADDR: begin
                    state <= S_OUT_LOAD;
                end
                S_OUT_LOAD: begin
                    state <= S_OUT_MAC;
                    mac_step <= 0;
                    mac_len_r <= PE_COUNT[STEP_W-1:0];
                    mac_base_r <= is_lm ? PE_COUNT[WEIGHT_AW-1:0] : {WEIGHT_AW{1'b0}};
                    score_idx <= 0;
                end
                S_OUT_MAC: begin
                    if (mac_step == mac_len_r + 1'b1)
                        state <= S_OUT_EMIT;
                    else
                        mac_step <= mac_step + 1'b1;
                end
                S_OUT_EMIT: begin
                    if (score_idx == LAST_SCORE[SCORE_AW-1:0])
                        state <= S_DONE;
                    else
                        score_idx <= score_idx + 1'b1;
                end
                S_CLEAR: begin
                    if (n == clear_count - 1'b1)
                        state <= S_DONE;
                    else
                        n <= n + 1'b1;
                end
                S_DONE: begin
                    if (!start)
                        state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
