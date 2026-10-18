`timescale 1ns / 1ps

//=====================================================================
// Module: lstm_tile.v
// Description:
//  Datapath for one LSTM tile, shared by the acoustic model and the
//  character LM. No state machine lives here. The controller drives
//  every enable and address. Both networks stay in the memories; a
//  model switch is a different weight address and context address.
//
//  Weight, bias, peephole, and context reads are synchronous. The
//  address on cycle N is valid at the data output on cycle N+1.
//  The gate-buffer read and the EPU are combinational, so neuron_addr
//  and a ready peephole/context word produce c_t and h_t in that
//  same cycle. context_we then stores {c_t, h_t}.
//
//  Per layer the controller is expected to:
//    1. Present the i/f bias address, then pulse load_bias.
//    2. MAC d_in (sel_in = 0) across the input vector.
//    3. MAC h_{t-1} (sel_in = 1) from the context word.
//    4. Pulse capture_gates with gate_pair = 0. That copies both
//       accumulator buses into memories i and f.
//    5. Repeat 1-4 for gates o and c with gate_pair = 1.
//    6. For neuron_addr = 0 .. PE_COUNT-1, present that neuron's
//       context and peephole addresses one cycle early, then pulse
//       context_we to store the EPU result.
//
//  Accumulator bits [ACC_LSB +: GATE_WIDTH] are what gets captured.
//  ACC_LSB stays 0 until the fixed-point scale is frozen.
//
//  Context entries are {c_t, h_t}. c_t is the top C_WIDTH bits.
//=====================================================================

module lstm_tile #(
    parameter PE_COUNT = 256,
    parameter X_WIDTH = 8,
    parameter W_WIDTH = 6,
    parameter BIAS_WIDTH = 16,
    parameter ACC_WIDTH = 24,
    parameter GATE_WIDTH = 16,
    parameter C_WIDTH = 16,
    parameter H_WIDTH = 8,
    parameter PEEP_WIDTH = 24,
    parameter ACC_LSB = 0,
    parameter WEIGHT_DEPTH = 4402,
    parameter CONTEXT_DEPTH = 66304,
    parameter N_LSTM_LAYERS = 5,
    parameter BIAS_DEPTH = N_LSTM_LAYERS * 2,
    parameter PEEP_DEPTH = N_LSTM_LAYERS * PE_COUNT,
    parameter WEIGHT_AW = $clog2(WEIGHT_DEPTH),
    parameter CONTEXT_AW = $clog2(CONTEXT_DEPTH),
    parameter BIAS_AW = $clog2(BIAS_DEPTH),
    parameter PEEP_AW = $clog2(PEEP_DEPTH),
    parameter NEURON_AW = (PE_COUNT <= 1) ? 1 : $clog2(PE_COUNT)
) (
    input wire clk,
    input wire rst,

    input wire pe_enable,
    input wire load_bias,
    input wire sel_in,
    input wire signed [X_WIDTH-1:0] d_in,

    input wire [WEIGHT_AW-1:0] weight_addr,
    input wire weight0_we,
    input wire weight1_we,
    input wire [PE_COUNT*W_WIDTH-1:0] weight0_din,
    input wire [PE_COUNT*W_WIDTH-1:0] weight1_din,

    input wire [BIAS_AW-1:0] bias_addr,
    input wire bias_we,
    input wire signed [PE_COUNT*BIAS_WIDTH-1:0] bias0_din,
    input wire signed [PE_COUNT*BIAS_WIDTH-1:0] bias1_din,

    input wire capture_gates,
    input wire gate_pair,
    input wire [NEURON_AW-1:0] neuron_addr,

    input wire [PEEP_AW-1:0] peep_addr,
    input wire peep_we,
    input wire signed [PEEP_WIDTH-1:0] peep_din,

    input wire [CONTEXT_AW-1:0] context_addr,
    input wire context_we,
    input wire context_use_ext,
    input wire [C_WIDTH+H_WIDTH-1:0] context_ext_din,

    output wire signed [C_WIDTH-1:0] c_t,
    output wire signed [H_WIDTH-1:0] h_t,
    output wire [C_WIDTH+H_WIDTH-1:0] context_dout,
    output wire signed [GATE_WIDTH-1:0] pe_i,
    output wire signed [GATE_WIDTH-1:0] pe_f,
    output wire signed [GATE_WIDTH-1:0] pe_o,
    output wire signed [GATE_WIDTH-1:0] pe_c
);

    localparam STATE_WIDTH = C_WIDTH + H_WIDTH;

    wire [PE_COUNT*W_WIDTH-1:0] weight0;
    wire [PE_COUNT*W_WIDTH-1:0] weight1;
    wire signed [PE_COUNT*ACC_WIDTH-1:0] pe_out0;
    wire signed [PE_COUNT*ACC_WIDTH-1:0] pe_out1;
    wire signed [PE_COUNT*GATE_WIDTH-1:0] gate_bus0;
    wire signed [PE_COUNT*GATE_WIDTH-1:0] gate_bus1;
    wire signed [PEEP_WIDTH-1:0] peep_dout;
    wire signed [X_WIDTH-1:0] h_from_context;
    wire signed [X_WIDTH-1:0] pe_data;
    wire signed [C_WIDTH-1:0] c_prev;
    wire [STATE_WIDTH-1:0] context_din;

    reg signed [PE_COUNT*BIAS_WIDTH-1:0] bias0_mem [0:BIAS_DEPTH-1];
    reg signed [PE_COUNT*BIAS_WIDTH-1:0] bias1_mem [0:BIAS_DEPTH-1];
    reg signed [PE_COUNT*BIAS_WIDTH-1:0] bias0_q;
    reg signed [PE_COUNT*BIAS_WIDTH-1:0] bias1_q;

    reg signed [PEEP_WIDTH-1:0] peep_mem [0:PEEP_DEPTH-1];
    reg signed [PEEP_WIDTH-1:0] peep_q;

    assign h_from_context = context_dout[H_WIDTH-1:0];
    assign c_prev = context_dout[STATE_WIDTH-1:H_WIDTH];
    assign pe_data = sel_in ? h_from_context : d_in;
    assign context_din = context_use_ext ? context_ext_din : {c_t, h_t};
    assign peep_dout = peep_q;

    weight_bram #(
        .WEIGHT_COUNT(WEIGHT_DEPTH),
        .WEIGHT_ELEMENT_WIDTH(W_WIDTH),
        .NEURON_COUNT(PE_COUNT),
        .ADDR_WIDTH(WEIGHT_AW)
    ) weight0_bram (
        .clk(clk),
        .write_enable(weight0_we),
        .address(weight_addr),
        .data_in(weight0_din),
        .data_out(weight0)
    );

    weight_bram #(
        .WEIGHT_COUNT(WEIGHT_DEPTH),
        .WEIGHT_ELEMENT_WIDTH(W_WIDTH),
        .NEURON_COUNT(PE_COUNT),
        .ADDR_WIDTH(WEIGHT_AW)
    ) weight1_bram (
        .clk(clk),
        .write_enable(weight1_we),
        .address(weight_addr),
        .data_in(weight1_din),
        .data_out(weight1)
    );

    always @(posedge clk) begin
        if (bias_we) begin
            bias0_mem[bias_addr] <= bias0_din;
            bias1_mem[bias_addr] <= bias1_din;
        end
        bias0_q <= bias0_mem[bias_addr];
        bias1_q <= bias1_mem[bias_addr];
    end

    always @(posedge clk) begin
        if (peep_we)
            peep_mem[peep_addr] <= peep_din;
        peep_q <= peep_mem[peep_addr];
    end

    pe_array #(
        .PE_ARRAY_SIZE(PE_COUNT),
        .DATA_ELEMENT_WIDTH(X_WIDTH),
        .WEIGHT_INPUT_WIDTH(W_WIDTH),
        .BIAS_INPUT_WIDTH(BIAS_WIDTH),
        .OUTPUT_ELEMENT_WIDTH(ACC_WIDTH)
    ) pe_arrays (
        .clk(clk),
        .rst(rst),
        .enable(pe_enable),
        .load_bias(load_bias),
        .data_input(pe_data),
        .weight_input0(weight0),
        .weight_input1(weight1),
        .bias_input0(bias0_q),
        .bias_input1(bias1_q),
        .pe_out0(pe_out0),
        .pe_out1(pe_out1)
    );

    genvar g;
    generate
        for (g = 0; g < PE_COUNT; g = g + 1) begin: gate_slice
            wire signed [ACC_WIDTH-1:0] acc0;
            wire signed [ACC_WIDTH-1:0] acc1;
            assign acc0 = pe_out0[g*ACC_WIDTH +: ACC_WIDTH];
            assign acc1 = pe_out1[g*ACC_WIDTH +: ACC_WIDTH];
            assign gate_bus0[g*GATE_WIDTH +: GATE_WIDTH] = acc0[ACC_LSB +: GATE_WIDTH];
            assign gate_bus1[g*GATE_WIDTH +: GATE_WIDTH] = acc1[ACC_LSB +: GATE_WIDTH];
        end
    endgenerate

    pe_buffer #(
        .GATE_WIDTH(GATE_WIDTH),
        .PE_COUNT(PE_COUNT),
        .ADDR_WIDTH(NEURON_AW)
    ) gates (
        .clk(clk),
        .rst(rst),
        .capture(capture_gates),
        .gate_pair(gate_pair),
        .pe_in0(gate_bus0),
        .pe_in1(gate_bus1),
        .addr(neuron_addr),
        .pe_i(pe_i),
        .pe_f(pe_f),
        .pe_o(pe_o),
        .pe_c(pe_c)
    );

    lstm_epu #(
        .HIDDEN_STATE_WIDTH(C_WIDTH),
        .PE_INPUT_WIDTH(GATE_WIDTH),
        .PEEPHOLE_WEIGHT_WIDTH(PEEP_WIDTH),
        .OUTPUT_ELEMENT_WIDTH(H_WIDTH)
    ) epu (
        .clk(clk),
        .rst(rst),
        .enable(1'b1),
        .old_hidden_state(c_prev),
        .pe_input({pe_c, pe_o, pe_f, pe_i}),
        .peephole_weight(peep_dout),
        .hidden_state_output(c_t),
        .output_state(h_t)
    );

    context_memory #(
        .STATE_WIDTH(STATE_WIDTH),
        .DEPTH(CONTEXT_DEPTH),
        .ADDR_WIDTH(CONTEXT_AW)
    ) context_store (
        .clk(clk),
        .write_enable(context_we),
        .address(context_addr),
        .data_in(context_din),
        .data_out(context_dout)
    );

endmodule
