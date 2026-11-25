`timescale 1ns / 1ps

//=====================================================================
// Module: round_sat.v
// Description:
//  Rescales a signed fixed-point value with an arithmetic right shift,
//  rounding half up, then saturates it to OUT_WIDTH bits:
//
//    dout = clamp((din + 2^(s-1)) >>> s, -2^(OUT_WIDTH-1), 2^(OUT_WIDTH-1) - 1)
//
//  s is SHIFT_LM while lm is high and SHIFT_AM otherwise. The acoustic
//  model and the character LM hold the same real value at different
//  accumulator scales (shared/specs/weight_image.md), so a slice that
//  feeds a table or leaves the fabric depends on the model. With
//  SHIFT_AM == SHIFT_LM, lm has no effect. Combinational. Both shifts
//  must be at least 1.
//=====================================================================

module round_sat #(
    parameter IN_WIDTH = 24,
    parameter OUT_WIDTH = 9,
    parameter SHIFT_AM = 9,
    parameter SHIFT_LM = 6
) (
    input wire lm,
    input wire signed [IN_WIDTH-1:0] din,
    output wire signed [OUT_WIDTH-1:0] dout
);

    localparam signed [IN_WIDTH:0] HALF_AM = 1 <<< (SHIFT_AM - 1);
    localparam signed [IN_WIDTH:0] HALF_LM = 1 <<< (SHIFT_LM - 1);
    localparam signed [IN_WIDTH:0] MAX = (1 <<< (OUT_WIDTH - 1)) - 1;
    localparam signed [IN_WIDTH:0] MIN = -(1 <<< (OUT_WIDTH - 1));

    // One extra bit so the rounding add cannot wrap.
    wire signed [IN_WIDTH:0] rounded = din + (lm ? HALF_LM : HALF_AM);
    wire signed [IN_WIDTH:0] shifted = lm ? (rounded >>> SHIFT_LM) : (rounded >>> SHIFT_AM);

    assign dout = (shifted > MAX) ? MAX[OUT_WIDTH-1:0]
                : (shifted < MIN) ? MIN[OUT_WIDTH-1:0]
                : shifted[OUT_WIDTH-1:0];

endmodule
