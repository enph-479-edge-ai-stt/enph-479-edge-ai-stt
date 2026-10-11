`timescale 1ns / 1ps

//=====================================================================
// Module: lstm_epu.v
// Description:
//  LSTM extra processing unit for one neuron. Combinational. The PE
//  buffers supply the four gate pre-activations. This block adds the
//  peephole terms, applies the sigmoid and tanh lookup tables, and
//  produces the new cell and hidden state:
//
//    i = sigmoid(pe_i + c_prev * w_ci)
//    f = sigmoid(pe_f + c_prev * w_cf)
//    c_tilde = tanh(pe_c)
//    c_t = f * c_prev + i * c_tilde
//    o = sigmoid(pe_o + c_t * w_co)
//    h_t = o * tanh(c_t)
//
//  The output gate uses c_t, so it is computed after the cell update.
//  w_ci, w_cf, w_co are packed in peephole_weight from MSB to LSB.
//  Fixed-point scales are not frozen yet. The slices below match the
//  previous alignment and will move when shared/specs pins them down.
//
//  clk, rst, and enable are unused. The datapath has no registers;
//  context_memory is what stores c_t and h_t.
//
// Ports:
//  old_hidden_state     - c_{t-1}, 16-bit cell (not h)
//  pe_input             - {pe_c, pe_o, pe_f, pe_i}, each PE_INPUT_WIDTH
//  peephole_weight      - {w_ci, w_cf, w_co}
//  hidden_state_output  - c_t
//  output_state         - h_t
//=====================================================================

module lstm_epu #(
    parameter HIDDEN_STATE_WIDTH = 16,
    parameter PE_INPUT_WIDTH = 16,
    parameter PEEPHOLE_WEIGHT_WIDTH = 24,
    parameter OUTPUT_ELEMENT_WIDTH = 8
) (
    input wire clk,
    input wire rst,
    input wire enable,

    input wire signed [HIDDEN_STATE_WIDTH-1:0] old_hidden_state,
    input wire signed [PE_INPUT_WIDTH*4-1:0] pe_input,
    input wire signed [PEEPHOLE_WEIGHT_WIDTH-1:0] peephole_weight,

    output wire signed [HIDDEN_STATE_WIDTH-1:0] hidden_state_output,
    output wire signed [OUTPUT_ELEMENT_WIDTH-1:0] output_state
);

    wire signed [PE_INPUT_WIDTH-1:0] pe_input_i = pe_input[PE_INPUT_WIDTH-1:0];
    wire signed [PE_INPUT_WIDTH-1:0] pe_input_f = pe_input[2*PE_INPUT_WIDTH-1:PE_INPUT_WIDTH];
    wire signed [PE_INPUT_WIDTH-1:0] pe_input_o = pe_input[3*PE_INPUT_WIDTH-1:2*PE_INPUT_WIDTH];
    wire signed [PE_INPUT_WIDTH-1:0] pe_input_c = pe_input[4*PE_INPUT_WIDTH-1:3*PE_INPUT_WIDTH];

    wire signed [OUTPUT_ELEMENT_WIDTH-1:0] w_ci =
        peephole_weight[PEEPHOLE_WEIGHT_WIDTH-1 -: OUTPUT_ELEMENT_WIDTH];
    wire signed [OUTPUT_ELEMENT_WIDTH-1:0] w_cf =
        peephole_weight[PEEPHOLE_WEIGHT_WIDTH-OUTPUT_ELEMENT_WIDTH-1 -: OUTPUT_ELEMENT_WIDTH];
    wire signed [OUTPUT_ELEMENT_WIDTH-1:0] w_co =
        peephole_weight[PEEPHOLE_WEIGHT_WIDTH-2*OUTPUT_ELEMENT_WIDTH-1 -: OUTPUT_ELEMENT_WIDTH];

    // Sums are truncated to the LUT input width. [7:4] is the current index.
    wire signed [OUTPUT_ELEMENT_WIDTH-1:0] activation_i_input =
        pe_input_i + old_hidden_state * w_ci;
    wire signed [OUTPUT_ELEMENT_WIDTH-1:0] activation_f_input =
        pe_input_f + old_hidden_state * w_cf;

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
        .index(activation_i_input[7:4]),
        .sigmoid_output(sigmoid_i_out)
    );

    sigmoid_lut #(
        .LUT_BIT_WIDTH(8),
        .LUT_PRECISION(16),
        .INPUT_BIT_WIDTH(4),
        .SIGMOID_OUTPUT_WIDTH(OUTPUT_ELEMENT_WIDTH)
    ) sig_f (
        .index(activation_f_input[7:4]),
        .sigmoid_output(sigmoid_f_out)
    );

    tanh_lut #(
        .LUT_BIT_WIDTH(8),
        .LUT_PRECISION(16),
        .INPUT_BIT_WIDTH(4),
        .TANH_OUTPUT_WIDTH(OUTPUT_ELEMENT_WIDTH)
    ) tanh_c (
        .index(pe_input_c[7:4]),
        .tanh_output(tanh_c_out)
    );

    // f * c_prev is 24 bits. Drop the low 8 bits to line up with i * c_tilde.
    wire signed [OUTPUT_ELEMENT_WIDTH+HIDDEN_STATE_WIDTH-1:0] f_times_c_prev;
    assign f_times_c_prev = $signed(sigmoid_f_out) * $signed(old_hidden_state);

    wire signed [2*OUTPUT_ELEMENT_WIDTH-1:0] i_times_c_tilde;
    assign i_times_c_tilde = $signed(sigmoid_i_out) * $signed(tanh_c_out);

    wire signed [HIDDEN_STATE_WIDTH-1:0] f_term =
        f_times_c_prev[OUTPUT_ELEMENT_WIDTH+HIDDEN_STATE_WIDTH-1:OUTPUT_ELEMENT_WIDTH];

    assign hidden_state_output = f_term + i_times_c_tilde;

    // Output-gate peephole sees the updated cell, not the candidate pre-activation.
    wire signed [OUTPUT_ELEMENT_WIDTH-1:0] activation_o_input =
        pe_input_o + hidden_state_output * w_co;

    sigmoid_lut #(
        .LUT_BIT_WIDTH(8),
        .LUT_PRECISION(16),
        .INPUT_BIT_WIDTH(4),
        .SIGMOID_OUTPUT_WIDTH(OUTPUT_ELEMENT_WIDTH)
    ) sig_o (
        .index(activation_o_input[7:4]),
        .sigmoid_output(sigmoid_o_out)
    );

    wire signed [OUTPUT_ELEMENT_WIDTH-1:0] tanh_ct_out;
    tanh_lut #(
        .LUT_BIT_WIDTH(8),
        .LUT_PRECISION(16),
        .INPUT_BIT_WIDTH(4),
        .TANH_OUTPUT_WIDTH(OUTPUT_ELEMENT_WIDTH)
    ) tanh_ct_inst (
        .index(hidden_state_output[11:8]),
        .tanh_output(tanh_ct_out)
    );

    wire signed [2*OUTPUT_ELEMENT_WIDTH-1:0] h_t_full;
    assign h_t_full = $signed(sigmoid_o_out) * $signed(tanh_ct_out);
    assign output_state = h_t_full[OUTPUT_ELEMENT_WIDTH-1:0];

endmodule
