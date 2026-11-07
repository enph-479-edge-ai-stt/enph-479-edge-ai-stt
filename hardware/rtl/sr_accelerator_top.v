`timescale 1ns / 1ps

//=====================================================================
// Module: sr_accelerator_top.v
// Description:
//  One LSTM tile, one output tile, and the controller that switches
//  them between the acoustic model and the character LM.
//
//  While idle or done, the host writes weights, biases, peepholes,
//  the projection, and the next feature frame. context_dout follows
//  host_context_addr one cycle later so a testbench can read state
//  back. During a job those ports are ignored and the controller
//  drives the tiles.
//
//  start is sampled only in IDLE. done stays high until start falls.
//  score_valid / score_idx / score walk the projection once per symbol
//  before done. scores holds the whole vector until the next job.
//=====================================================================

module sr_accelerator_top #(
    parameter PE_COUNT = 256,
    parameter AM_LAYERS = 3,
    parameter LM_LAYERS = 2,
    parameter AM_IN = 123,
    parameter LM_IN = 30,
    parameter VOCAB = 30,
    parameter N_BEAMS = 128,
    parameter X_WIDTH = 8,
    parameter W_WIDTH = 6,
    parameter BIAS_WIDTH = 16,
    parameter SCORE_WIDTH = 8,
    parameter WEIGHT_DEPTH =
        ((AM_IN + PE_COUNT) * 2)
        + ((AM_LAYERS > 1) ? (AM_LAYERS - 1) : 0) * (PE_COUNT * 4)
        + ((LM_IN + PE_COUNT) * 2)
        + ((LM_LAYERS > 1) ? (LM_LAYERS - 1) : 0) * (PE_COUNT * 4),
    parameter CONTEXT_DEPTH = (AM_LAYERS * PE_COUNT) + (N_BEAMS * LM_LAYERS * PE_COUNT),
    parameter N_LSTM_LAYERS = AM_LAYERS + LM_LAYERS,
    parameter WEIGHT_AW = $clog2(WEIGHT_DEPTH),
    parameter CONTEXT_AW = $clog2(CONTEXT_DEPTH),
    parameter BIAS_AW = $clog2(N_LSTM_LAYERS * 2),
    parameter PEEP_AW = $clog2(N_LSTM_LAYERS * PE_COUNT),
    parameter FRAME_AW = (AM_IN <= 1) ? 1 : $clog2(AM_IN),
    parameter BEAM_W = (N_BEAMS <= 1) ? 1 : $clog2(N_BEAMS),
    parameter CHAR_W = (VOCAB <= 1) ? 1 : $clog2(VOCAB),
    parameter SCORE_AW = (VOCAB <= 1) ? 1 : $clog2(VOCAB),
    parameter OUT_AW = $clog2(PE_COUNT * 2)
) (
    input wire clk,
    input wire rst,

    input wire start,
    input wire [1:0] job,
    input wire [BEAM_W-1:0] beam_id,
    input wire [CHAR_W-1:0] lm_char,

    input wire frame_we,
    input wire [FRAME_AW-1:0] frame_addr,
    input wire signed [X_WIDTH-1:0] frame_data,

    input wire [CONTEXT_AW-1:0] host_context_addr,

    input wire host_weight0_we,
    input wire host_weight1_we,
    input wire [WEIGHT_AW-1:0] host_weight_addr,
    input wire [PE_COUNT*W_WIDTH-1:0] host_weight0_din,
    input wire [PE_COUNT*W_WIDTH-1:0] host_weight1_din,

    input wire host_bias_we,
    input wire [BIAS_AW-1:0] host_bias_addr,
    input wire signed [PE_COUNT*BIAS_WIDTH-1:0] host_bias0_din,
    input wire signed [PE_COUNT*BIAS_WIDTH-1:0] host_bias1_din,

    input wire host_peep_we,
    input wire [PEEP_AW-1:0] host_peep_addr,
    input wire signed [23:0] host_peep_din,

    input wire host_out_weight_we,
    input wire [OUT_AW-1:0] host_out_weight_addr,
    input wire [VOCAB*W_WIDTH-1:0] host_out_weight_din,
    input wire host_out_bias_we,
    input wire host_out_bias_addr,
    input wire signed [VOCAB*BIAS_WIDTH-1:0] host_out_bias_din,

    output wire busy,
    output wire done,
    output wire error,
    output wire [3:0] state_out,

    output wire score_valid,
    output wire [SCORE_AW-1:0] score_idx,
    output wire signed [SCORE_WIDTH-1:0] score,
    output wire signed [VOCAB*SCORE_WIDTH-1:0] scores,
    output wire [23:0] context_dout
);

    localparam C_WIDTH = 16;
    localparam H_WIDTH = 8;
    localparam NEURON_AW = (PE_COUNT <= 1) ? 1 : $clog2(PE_COUNT);
    localparam MAX_DIM = (AM_IN > PE_COUNT)
        ? ((AM_IN > LM_IN) ? AM_IN : LM_IN)
        : ((PE_COUNT > LM_IN) ? PE_COUNT : LM_IN);
    localparam STEP_W = $clog2(MAX_DIM + 3);

    wire idle;
    wire signed [X_WIDTH-1:0] d_in;
    wire signed [X_WIDTH-1:0] frame_elem;
    wire ctrl_pe_enable, ctrl_load_bias, ctrl_sel_in, ctrl_capture;
    wire ctrl_gate_pair, ctrl_context_we, ctrl_use_ext;
    wire ctrl_out_enable, ctrl_out_load, ctrl_out_bias_addr;
    wire [WEIGHT_AW-1:0] ctrl_weight_addr;
    wire [BIAS_AW-1:0] ctrl_bias_addr;
    wire [PEEP_AW-1:0] ctrl_peep_addr;
    wire [CONTEXT_AW-1:0] ctrl_context_addr;
    wire [OUT_AW-1:0] ctrl_out_weight_addr;
    wire [NEURON_AW-1:0] ctrl_neuron_addr;
    wire [STEP_W-1:0] x_idx;

    reg signed [X_WIDTH-1:0] frame_mem [0:AM_IN-1];
    wire [FRAME_AW-1:0] frame_rd;

    fsm_controller #(
        .PE_COUNT(PE_COUNT),
        .AM_LAYERS(AM_LAYERS),
        .LM_LAYERS(LM_LAYERS),
        .AM_IN(AM_IN),
        .LM_IN(LM_IN),
        .VOCAB(VOCAB),
        .N_BEAMS(N_BEAMS),
        .X_WIDTH(X_WIDTH),
        .SCORE_WIDTH(SCORE_WIDTH),
        .WEIGHT_DEPTH(WEIGHT_DEPTH),
        .CONTEXT_DEPTH(CONTEXT_DEPTH),
        .WEIGHT_AW(WEIGHT_AW),
        .CONTEXT_AW(CONTEXT_AW),
        .BIAS_AW(BIAS_AW),
        .PEEP_AW(PEEP_AW),
        .OUT_AW(OUT_AW)
    ) ctrl (
        .clk(clk),
        .rst(rst),
        .start(start),
        .job(job),
        .beam_id(beam_id),
        .lm_char(lm_char),
        .frame_elem(frame_elem),
        .idle(idle),
        .busy(busy),
        .done(done),
        .error(error),
        .pe_enable(ctrl_pe_enable),
        .load_bias(ctrl_load_bias),
        .sel_in(ctrl_sel_in),
        .d_in(d_in),
        .weight_addr(ctrl_weight_addr),
        .bias_addr(ctrl_bias_addr),
        .capture_gates(ctrl_capture),
        .gate_pair(ctrl_gate_pair),
        .neuron_addr(ctrl_neuron_addr),
        .peep_addr(ctrl_peep_addr),
        .context_addr(ctrl_context_addr),
        .context_we(ctrl_context_we),
        .context_use_ext(ctrl_use_ext),
        .x_idx(x_idx),
        .out_enable(ctrl_out_enable),
        .out_load_bias(ctrl_out_load),
        .out_weight_addr(ctrl_out_weight_addr),
        .out_bias_addr(ctrl_out_bias_addr),
        .score_valid(score_valid),
        .score_idx(score_idx),
        .state_out(state_out)
    );

    // The controller's x_idx width is STEP_W. Connect through a wide wire
    // by letting the port size win; this assign keeps the frame index in range.
    assign frame_rd = (x_idx < AM_IN) ? x_idx[FRAME_AW-1:0] : {FRAME_AW{1'b0}};
    assign frame_elem = frame_mem[frame_rd];

    always @(posedge clk) begin
        if (frame_we && idle)
            frame_mem[frame_addr] <= frame_data;
    end

    wire [WEIGHT_AW-1:0] weight_addr = idle ? host_weight_addr : ctrl_weight_addr;
    wire weight0_we = idle & host_weight0_we;
    wire weight1_we = idle & host_weight1_we;
    wire [BIAS_AW-1:0] bias_addr = idle ? host_bias_addr : ctrl_bias_addr;
    wire bias_we = idle & host_bias_we;
    wire [PEEP_AW-1:0] peep_addr = idle ? host_peep_addr : ctrl_peep_addr;
    wire peep_we = idle & host_peep_we;
    wire [CONTEXT_AW-1:0] context_addr = idle ? host_context_addr : ctrl_context_addr;
    wire [OUT_AW-1:0] out_weight_addr = idle ? host_out_weight_addr : ctrl_out_weight_addr;
    wire out_weight_we = idle & host_out_weight_we;
    wire out_bias_addr = idle ? host_out_bias_addr : ctrl_out_bias_addr;
    wire out_bias_we = idle & host_out_bias_we;

    wire signed [H_WIDTH-1:0] h_to_proj = context_dout[H_WIDTH-1:0];
    wire signed [VOCAB*SCORE_WIDTH-1:0] score_bus;
    assign scores = score_bus;
    assign score = score_bus[score_idx*SCORE_WIDTH +: SCORE_WIDTH];

    lstm_tile #(
        .PE_COUNT(PE_COUNT),
        .X_WIDTH(X_WIDTH),
        .W_WIDTH(W_WIDTH),
        .BIAS_WIDTH(BIAS_WIDTH),
        .WEIGHT_DEPTH(WEIGHT_DEPTH),
        .WEIGHT_AW(WEIGHT_AW),
        .CONTEXT_DEPTH(CONTEXT_DEPTH),
        .CONTEXT_AW(CONTEXT_AW),
        .N_LSTM_LAYERS(N_LSTM_LAYERS),
        .BIAS_DEPTH(N_LSTM_LAYERS * 2),
        .BIAS_AW(BIAS_AW),
        .PEEP_DEPTH(N_LSTM_LAYERS * PE_COUNT),
        .PEEP_AW(PEEP_AW),
        .NEURON_AW(NEURON_AW)
    ) lstm (
        .clk(clk),
        .rst(rst),
        .pe_enable(idle ? 1'b0 : ctrl_pe_enable),
        .load_bias(idle ? 1'b0 : ctrl_load_bias),
        .sel_in(idle ? 1'b0 : ctrl_sel_in),
        .d_in(d_in),
        .weight_addr(weight_addr),
        .weight0_we(weight0_we),
        .weight1_we(weight1_we),
        .weight0_din(host_weight0_din),
        .weight1_din(host_weight1_din),
        .bias_addr(bias_addr),
        .bias_we(bias_we),
        .bias0_din(host_bias0_din),
        .bias1_din(host_bias1_din),
        .capture_gates(idle ? 1'b0 : ctrl_capture),
        .gate_pair(ctrl_gate_pair),
        .neuron_addr(ctrl_neuron_addr),
        .peep_addr(peep_addr),
        .peep_we(peep_we),
        .peep_din(host_peep_din),
        .context_addr(context_addr),
        .context_we(idle ? 1'b0 : ctrl_context_we),
        .context_use_ext(idle ? 1'b0 : ctrl_use_ext),
        .context_ext_din({(C_WIDTH+H_WIDTH){1'b0}}),
        .context_dout(context_dout)
    );

    output_tile #(
        .HIDDEN(PE_COUNT),
        .VOCAB(VOCAB),
        .N_JOBS(2),
        .X_WIDTH(X_WIDTH),
        .W_WIDTH(W_WIDTH),
        .BIAS_WIDTH(BIAS_WIDTH),
        .SCORE_WIDTH(SCORE_WIDTH)
    ) proj (
        .clk(clk),
        .rst(rst),
        .enable(idle ? 1'b0 : ctrl_out_enable),
        .load_bias(idle ? 1'b0 : ctrl_out_load),
        .h_element(h_to_proj),
        .weight_addr(out_weight_addr),
        .weight_we(out_weight_we),
        .weight_din(host_out_weight_din),
        .bias_addr(out_bias_addr),
        .bias_we(out_bias_we),
        .bias_din(host_out_bias_din),
        .scores(score_bus)
    );

endmodule
