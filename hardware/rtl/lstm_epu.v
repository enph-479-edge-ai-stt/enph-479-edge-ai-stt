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
//  Fixed point (shared/specs/weight_image.md, "Activations and cell"):
//    pe_*     9-bit signed at 2^-5 (+-8), already rounded and saturated
//             from the accumulator by lstm_tile; offset binary indexes
//             the 512-entry tables
//    i, f, o  unsigned 8-bit at 2^-8 (sigmoid_lut)
//    c_tilde  signed 8-bit at 2^-7 (tanh_lut)
//    c        signed 16-bit at 2^-11 (+-16)
//    h        signed 8-bit at 2^-7
//
//    c_t = sat16(round(f * c_prev >> 8) + round(i * c_tilde >> 4))
//    h_t = round(o * tanh(round_sat9(c_t >> 6)) >> 8)
//
//  Every round is half up: add half an output step, then shift right
//  arithmetically. h never needs saturating, |o * tanh| <= 255 * 127
//  gives |h_t| <= 127.
//
//  clk, rst, and enable are unused. The datapath has no registers;
//  context_memory is what stores c_t and h_t.
//
// Ports:
//  old_hidden_state     - c_{t-1}, 16-bit cell (not h)
//  pe_input             - {pe_c, pe_o, pe_f, pe_i}, 9 bits each
//  hidden_state_output  - c_t
//  output_state         - h_t
//=====================================================================

module lstm_epu (
    input wire clk,
    input wire rst,
    input wire enable,

    input wire signed [15:0] old_hidden_state,
    input wire signed [35:0] pe_input,

    output wire signed [15:0] hidden_state_output,
    output wire signed [7:0] output_state
);

    wire signed [8:0] pe_i = pe_input[8:0];
    wire signed [8:0] pe_f = pe_input[17:9];
    wire signed [8:0] pe_o = pe_input[26:18];
    wire signed [8:0] pe_c = pe_input[35:27];

    wire [7:0] sigmoid_i;
    wire [7:0] sigmoid_f;
    wire [7:0] sigmoid_o;
    wire signed [7:0] tanh_c;

    sigmoid_lut sig_i (.index({~pe_i[8], pe_i[7:0]}), .sigmoid_output(sigmoid_i));
    sigmoid_lut sig_f (.index({~pe_f[8], pe_f[7:0]}), .sigmoid_output(sigmoid_f));
    sigmoid_lut sig_o (.index({~pe_o[8], pe_o[7:0]}), .sigmoid_output(sigmoid_o));
    tanh_lut tanh_cand (.index({~pe_c[8], pe_c[7:0]}), .tanh_output(tanh_c));

    // 0..255 as a non-negative signed operand.
    wire signed [8:0] gate_i = {1'b0, sigmoid_i};
    wire signed [8:0] gate_f = {1'b0, sigmoid_f};
    wire signed [8:0] gate_o = {1'b0, sigmoid_o};

    // f * c_prev is at 2^-19; back to 2^-11. |f * c_prev| < 2^23.
    wire signed [24:0] f_times_c = gate_f * old_hidden_state;
    wire signed [16:0] f_term = (f_times_c + 25'sd128) >>> 8;

    // i * c_tilde is at 2^-15; to 2^-11. |i * c_tilde| <= 32385.
    wire signed [16:0] i_times_g = gate_i * tanh_c;
    wire signed [12:0] i_term = (i_times_g + 17'sd8) >>> 4;

    wire signed [17:0] c_sum = f_term + i_term;
    assign hidden_state_output = (c_sum > 18'sd32767) ? 16'sh7fff
                               : (c_sum < -18'sd32768) ? 16'sh8000
                               : c_sum[15:0];

    // tanh(c_t): the cell from 2^-11 to the table's 2^-5.
    wire signed [8:0] c_index;
    wire signed [7:0] tanh_ct;

    round_sat #(
        .IN_WIDTH(16),
        .OUT_WIDTH(9),
        .SHIFT_AM(6),
        .SHIFT_LM(6)
    ) cell_slice (
        .lm(1'b0),
        .din(hidden_state_output),
        .dout(c_index)
    );

    tanh_lut tanh_cell (.index({~c_index[8], c_index[7:0]}), .tanh_output(tanh_ct));

    // o * tanh(c_t) is at 2^-15; the high byte, rounded, is h at 2^-7.
    wire signed [16:0] h_round = gate_o * tanh_ct + 17'sd128;
    assign output_state = h_round[15:8];

endmodule
