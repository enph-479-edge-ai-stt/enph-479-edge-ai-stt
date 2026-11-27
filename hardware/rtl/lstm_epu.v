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
//  The arithmetic is three blocks, lstm_iterm, lstm_cell and lstm_h, so
//  lstm_tile can register between them. This module wires the three with
//  no registers: its outputs are the function of the current inputs.
//  clk, rst, and enable are unused.
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
// Ports:
//  old_hidden_state     - c_{t-1}, 16-bit cell (not h)
//  pe_input             - {pe_c, pe_o, pe_f, pe_i}, 9 bits each
//  hidden_state_output  - c_t
//  output_state         - h_t
//=====================================================================

// i*tanh(g), and the f and o gates. None of this reads the old cell.
module lstm_iterm (
    input wire signed [8:0] pe_i,
    input wire signed [8:0] pe_f,
    input wire signed [8:0] pe_o,
    input wire signed [8:0] pe_c,
    output wire signed [8:0] gate_i,
    output wire signed [8:0] gate_f,
    output wire signed [8:0] gate_o,
    output wire signed [7:0] tanh_c,
    output wire signed [12:0] i_term
);
    wire [7:0] sigmoid_i;
    wire [7:0] sigmoid_f;
    wire [7:0] sigmoid_o;

    sigmoid_lut sig_i (.index({~pe_i[8], pe_i[7:0]}), .sigmoid_output(sigmoid_i));
    sigmoid_lut sig_f (.index({~pe_f[8], pe_f[7:0]}), .sigmoid_output(sigmoid_f));
    sigmoid_lut sig_o (.index({~pe_o[8], pe_o[7:0]}), .sigmoid_output(sigmoid_o));
    tanh_lut tanh_cand (.index({~pe_c[8], pe_c[7:0]}), .tanh_output(tanh_c));

    // 0..255 as a non-negative signed operand.
    assign gate_i = {1'b0, sigmoid_i};
    assign gate_f = {1'b0, sigmoid_f};
    assign gate_o = {1'b0, sigmoid_o};

    // i * c_tilde is at 2^-15; to 2^-11. |i * c_tilde| <= 32385.
    wire signed [16:0] i_times_g = gate_i * tanh_c;
    assign i_term = (i_times_g + 17'sd8) >>> 4;
endmodule

// f * c_prev + i_term, saturated to the 16-bit cell.
module lstm_cell (
    input wire signed [15:0] c_prev,
    input wire signed [8:0] gate_f,
    input wire signed [12:0] i_term,
    output wire signed [15:0] c_t
);
    // f * c_prev is at 2^-19; back to 2^-11. |f * c_prev| < 2^23.
    wire signed [24:0] f_times_c = gate_f * c_prev;
    wire signed [16:0] f_term = (f_times_c + 25'sd128) >>> 8;
    wire signed [17:0] c_sum = f_term + i_term;

    assign c_t = (c_sum > 18'sd32767) ? 16'sh7fff
               : (c_sum < -18'sd32768) ? 16'sh8000
               : c_sum[15:0];
endmodule

// h = o * tanh(c). c is the registered cell; gate_o is the same neuron's
// output gate, delayed to match it.
module lstm_h (
    input wire signed [15:0] c_t,
    input wire signed [8:0] gate_o,
    output wire signed [7:0] h_t
);
    wire signed [8:0] c_index;
    wire signed [7:0] tanh_ct;

    // tanh(c_t): the cell from 2^-11 to the table's 2^-5.
    round_sat #(
        .IN_WIDTH(16),
        .OUT_WIDTH(9),
        .SHIFT_AM(6),
        .SHIFT_LM(6)
    ) cell_slice (
        .lm(1'b0),
        .din(c_t),
        .dout(c_index)
    );

    tanh_lut tanh_cell (.index({~c_index[8], c_index[7:0]}), .tanh_output(tanh_ct));

    // o * tanh(c_t) is at 2^-15; the high byte, rounded, is h at 2^-7.
    wire signed [16:0] h_round = gate_o * tanh_ct + 17'sd128;
    assign h_t = h_round[15:8];
endmodule

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

    wire signed [8:0] gate_i;
    wire signed [8:0] gate_f;
    wire signed [8:0] gate_o;
    wire signed [7:0] tanh_c;
    wire signed [12:0] i_term;

    lstm_iterm terms (
        .pe_i(pe_i),
        .pe_f(pe_f),
        .pe_o(pe_o),
        .pe_c(pe_c),
        .gate_i(gate_i),
        .gate_f(gate_f),
        .gate_o(gate_o),
        .tanh_c(tanh_c),
        .i_term(i_term)
    );

    lstm_cell u_cell (
        .c_prev(old_hidden_state),
        .gate_f(gate_f),
        .i_term(i_term),
        .c_t(hidden_state_output)
    );

    lstm_h hidden (
        .c_t(hidden_state_output),
        .gate_o(gate_o),
        .h_t(output_state)
    );
endmodule
