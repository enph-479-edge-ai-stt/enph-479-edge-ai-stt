`timescale 1ps / 1ps

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
        sigmoid_output = lut[index]
    end
endmodule
