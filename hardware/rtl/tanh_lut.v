`timescale 1ps / 1ps

module sigmoid_lut #(
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
