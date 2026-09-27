`timescale 1ps / 1ps

//=====================================================================
// Module: sigmoid_lut.v
// Description:
//  Combinational sigmoid activation lookup table. Maps a quantized
//  index to an unsigned 8-bit sigmoid value, so the datapath never
//  needs a real sigmoid. LUT_PRECISION entries cover the input range
//  symmetrically about zero: index LUT_PRECISION/2 is sigmoid(0), held
//  as 128 (0.5 in 8-bit unsigned scale), and the entries at either end
//  saturate at 0 / 255. Purely combinational, so the result is
//  available in the same cycle as the index.
//
// Parameters:
//  LUT_BIT_WIDTH           - bit width of a stored LUT entry (unsigned)
//  LUT_PRECISION           - number of LUT entries, must be 2**INPUT_BIT_WIDTH
//  INPUT_BIT_WIDTH         - bit width of the index port
//  SIGMOID_OUTPUT_WIDTH    - bit width of the output value (unsigned)
// Ports:
//  index               - quantized input, selects the LUT entry
//  sigmoid_output      - unsigned sigmoid value for that entry
//=====================================================================

module sigmoid_lut #(
    parameter LUT_BIT_WIDTH = 8,
    parameter LUT_PRECISION = 16,
    parameter INPUT_BIT_WIDTH = 4,
    parameter SIGMOID_OUTPUT_WIDTH = 8
) (
    input wire [INPUT_BIT_WIDTH-1:0] index,
    output reg [SIGMOID_OUTPUT_WIDTH-1:0] sigmoid_output
);

    reg [LUT_BIT_WIDTH-1:0] lut [0:LUT_PRECISION-1];

    initial begin
        lut[0]  = 8'd0;   lut[1]  = 8'd0;   lut[2]  = 8'd1;   lut[3]  = 8'd2;
        lut[4]  = 8'd5;   lut[5]  = 8'd12;  lut[6]  = 8'd30;  lut[7]  = 8'd69;
        lut[8]  = 8'd128; lut[9]  = 8'd186; lut[10] = 8'd225; lut[11] = 8'd243;
        lut[12] = 8'd250; lut[13] = 8'd253; lut[14] = 8'd254; lut[15] = 8'd255;
    end

    always @(*) begin
        sigmoid_output = lut[index];
    end
endmodule
