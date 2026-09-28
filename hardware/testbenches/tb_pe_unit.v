module tb_pe_unit;
localparam X_ELEMENT_WIDTH = 8,
localparam WEIGHT_ELEMENT_WIDTH = 6,     
localparam OUTPUT_ELEMENT_WIDTH = 24,

reg clk = 0;
reg rst;
reg enable;

reg load_bias;
reg [OUTPUT_ELEMENT_WIDTH-1:0] bias;

reg signed [X_ELEMENT_WIDTH-1:0] x_input;
reg signed [WEIGHT_ELEMENT_WIDTH-1:0] weight;

wire signed [OUTPUT_ELEMENT_WIDTH-1:0] pe_out;

pe_unit #(
    .X_ELEMENT_WIDTH        (X_ELEMENT_WIDTH),
    .WEIGHT_ELEMENT_WIDTH   (WEIGHT_ELEMENT_WIDTH),
    .OUTPUT_ELEMENT_WIDTH   (OUTPUT_ELEMENT_WIDTH)
) dut (
    .clk(clk),
    .rst(rst),
    .enable(enable),

    .load_bias(load_bias),
    .bias(bias),

    .x_input(x_input),
    .weight(weight),

    .pe_out(pe_out)
);

always #5 clk = ~clk;

endmodule