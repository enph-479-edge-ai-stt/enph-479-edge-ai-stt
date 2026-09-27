`timescale 1ps / 1ps

//=====================================================================
// Module: tanh_lut.v
// Description:
//  Combinational tanh activation lookup table. Maps a quantized index
//  to a signed 8-bit tanh value, so the datapath never needs a real
//  tanh. LUT_PRECISION entries cover the input range symmetrically
//  about zero: index LUT_PRECISION/2 is tanh(0) = 0, and the entries
//  at either end saturate at -127 / +127. Purely combinational, so the
//  result is available in the same cycle as the index.
//
// Parameters:
//  LUT_BIT_WIDTH       - bit width of a stored LUT entry (signed)
//  LUT_PRECISION       - number of LUT entries, must be 2**INPUT_BIT_WIDTH
//  INPUT_BIT_WIDTH     - bit width of the index port
//  TANH_OUTPUT_WIDTH   - bit width of the output value (signed)
// Ports:
//  index           - quantized input, selects the LUT entry
//  tanh_output     - signed tanh value for that entry
//=====================================================================

module tanh_lut #(
    parameter LUT_BIT_WIDTH = 8,
    parameter LUT_PRECISION = 16,
    parameter INPUT_BIT_WIDTH = 4,
    parameter TANH_OUTPUT_WIDTH = 8
) (
    input wire [INPUT_BIT_WIDTH-1:0] index,
    output reg signed [TANH_OUTPUT_WIDTH-1:0] tanh_output
);

    reg signed [LUT_BIT_WIDTH-1:0] lut [0:LUT_PRECISION-1];

    initial begin
        lut[0]  = -8'sd127; lut[1]  = -8'sd127; lut[2]  = -8'sd127; lut[3]  = -8'sd127;
        lut[4]  = -8'sd127; lut[5]  = -8'sd126; lut[6]  = -8'sd122; lut[7]  = -8'sd97;
        lut[8]  =  8'sd0;   lut[9]  =  8'sd97;  lut[10] =  8'sd122; lut[11] =  8'sd126;
        lut[12] =  8'sd127; lut[13] =  8'sd127; lut[14] =  8'sd127; lut[15] =  8'sd127;
    end

    always @(*) begin
        tanh_output = lut[index];
    end
endmodule
