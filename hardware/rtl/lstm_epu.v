`timescale 1ns / 1ps

//=====================================================================
// Module: lstm_epu.v
// Description:
//  LSTM extra processing unit for one neuron. Combinational. The PE
//  buffers supply the four gate pre-activations. This block applies the
//  sigmoid and tanh lookup tables and produces the new cell and hidden
//  state of a standard LSTM (no peepholes, as nn.LSTM):
//
//    i = sigmoid(pe_i)
//    f = sigmoid(pe_f)
//    c_tilde = tanh(pe_c)
//    c_t = f * c_prev + i * c_tilde
//    o = sigmoid(pe_o)
//    h_t = o * tanh(c_t)
//
//  Fixed-point scales are not frozen yet. The slices below match the
//  previous alignment and will move when shared/specs pins them down.
//
//  The LUTs hold zero at index 8 (offset binary), so each signed 4-bit
//  slice has its top bit flipped before it indexes a table. Sigmoid
//  outputs are unsigned 0..255 and are zero-extended before multiplying.
//
//  clk, rst, and enable are unused. The datapath has no registers;
//  context_memory is what stores c_t and h_t.
//
// Ports:
//  old_hidden_state     - c_{t-1}, 16-bit cell (not h)
//  pe_input             - {pe_c, pe_o, pe_f, pe_i}, each PE_INPUT_WIDTH
//  hidden_state_output  - c_t
//  output_state         - h_t
//=====================================================================

module lstm_epu #(
    parameter HIDDEN_STATE_WIDTH = 16,
    parameter PE_INPUT_WIDTH = 16,
    parameter OUTPUT_ELEMENT_WIDTH = 8
) (
    input wire clk,
    input wire rst,
    input wire enable,

    input wire signed [HIDDEN_STATE_WIDTH-1:0] old_hidden_state,
    input wire signed [PE_INPUT_WIDTH*4-1:0] pe_input,

    output wire signed [HIDDEN_STATE_WIDTH-1:0] hidden_state_output,
    output wire signed [OUTPUT_ELEMENT_WIDTH-1:0] output_state
);

    wire signed [PE_INPUT_WIDTH-1:0] pe_input_i = pe_input[PE_INPUT_WIDTH-1:0];
    wire signed [PE_INPUT_WIDTH-1:0] pe_input_f = pe_input[2*PE_INPUT_WIDTH-1:PE_INPUT_WIDTH];
    wire signed [PE_INPUT_WIDTH-1:0] pe_input_o = pe_input[3*PE_INPUT_WIDTH-1:2*PE_INPUT_WIDTH];
    wire signed [PE_INPUT_WIDTH-1:0] pe_input_c = pe_input[4*PE_INPUT_WIDTH-1:3*PE_INPUT_WIDTH];

    // Two's complement slice -> offset-binary LUT index. [7:4] is the
    // current index.
    wire [3:0] index_i = {~pe_input_i[7], pe_input_i[6:4]};
    wire [3:0] index_f = {~pe_input_f[7], pe_input_f[6:4]};
    wire [3:0] index_o = {~pe_input_o[7], pe_input_o[6:4]};
    wire [3:0] index_c = {~pe_input_c[7], pe_input_c[6:4]};

    wire [OUTPUT_ELEMENT_WIDTH-1:0] sigmoid_i_out;
    wire [OUTPUT_ELEMENT_WIDTH-1:0] sigmoid_f_out;
    wire [OUTPUT_ELEMENT_WIDTH-1:0] sigmoid_o_out;
    wire [OUTPUT_ELEMENT_WIDTH-1:0] tanh_c_out;

    sigmoid_lut #(
        .LUT_BIT_WIDTH(8),
        .LUT_PRECISION(16),
        .INPUT_BIT_WIDTH(4),
        .SIGMOID_OUTPUT_WIDTH(OUTPUT_ELEMENT_WIDTH)
    ) sig_i (
        .index(index_i),
        .sigmoid_output(sigmoid_i_out)
    );

    sigmoid_lut #(
        .LUT_BIT_WIDTH(8),
        .LUT_PRECISION(16),
        .INPUT_BIT_WIDTH(4),
        .SIGMOID_OUTPUT_WIDTH(OUTPUT_ELEMENT_WIDTH)
    ) sig_f (
        .index(index_f),
        .sigmoid_output(sigmoid_f_out)
    );

    sigmoid_lut #(
        .LUT_BIT_WIDTH(8),
        .LUT_PRECISION(16),
        .INPUT_BIT_WIDTH(4),
        .SIGMOID_OUTPUT_WIDTH(OUTPUT_ELEMENT_WIDTH)
    ) sig_o (
        .index(index_o),
        .sigmoid_output(sigmoid_o_out)
    );

    tanh_lut #(
        .LUT_BIT_WIDTH(8),
        .LUT_PRECISION(16),
        .INPUT_BIT_WIDTH(4),
        .TANH_OUTPUT_WIDTH(OUTPUT_ELEMENT_WIDTH)
    ) tanh_c (
        .index(index_c),
        .tanh_output(tanh_c_out)
    );

    // 0..255 as a non-negative signed operand.
    wire signed [OUTPUT_ELEMENT_WIDTH:0] gate_i = {1'b0, sigmoid_i_out};
    wire signed [OUTPUT_ELEMENT_WIDTH:0] gate_f = {1'b0, sigmoid_f_out};
    wire signed [OUTPUT_ELEMENT_WIDTH:0] gate_o = {1'b0, sigmoid_o_out};

    // f * c_prev is 24 bits. Drop the low 8 bits to line up with i * c_tilde.
    wire signed [OUTPUT_ELEMENT_WIDTH+HIDDEN_STATE_WIDTH-1:0] f_times_c_prev;
    assign f_times_c_prev = gate_f * $signed(old_hidden_state);

    wire signed [2*OUTPUT_ELEMENT_WIDTH-1:0] i_times_c_tilde;
    assign i_times_c_tilde = gate_i * $signed(tanh_c_out);

    wire signed [HIDDEN_STATE_WIDTH-1:0] f_term =
        f_times_c_prev[OUTPUT_ELEMENT_WIDTH+HIDDEN_STATE_WIDTH-1:OUTPUT_ELEMENT_WIDTH];

    assign hidden_state_output = f_term + i_times_c_tilde;

    wire [3:0] index_ct = {~hidden_state_output[11], hidden_state_output[10:8]};

    wire signed [OUTPUT_ELEMENT_WIDTH-1:0] tanh_ct_out;
    tanh_lut #(
        .LUT_BIT_WIDTH(8),
        .LUT_PRECISION(16),
        .INPUT_BIT_WIDTH(4),
        .TANH_OUTPUT_WIDTH(OUTPUT_ELEMENT_WIDTH)
    ) tanh_ct_inst (
        .index(index_ct),
        .tanh_output(tanh_ct_out)
    );

    wire signed [2*OUTPUT_ELEMENT_WIDTH-1:0] h_t_full;
    assign h_t_full = gate_o * tanh_ct_out;
    assign output_state = h_t_full[OUTPUT_ELEMENT_WIDTH-1:0];

endmodule
