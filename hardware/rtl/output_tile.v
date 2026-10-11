`timescale 1ns / 1ps

//=====================================================================
// Module: output_tile.v
// Description:
//  Fully connected projection from the last LSTM hidden vector to one
//  score per vocabulary symbol. This is not an LSTM. It does not read
//  or write context. The acoustic model and the character LM each have
//  a block of rows in the same weight memory; the controller selects
//  the block by the address it presents.
//
//  One processing element per symbol. Every element sees the same
//  h_element. Weight and bias reads are synchronous: the address on
//  cycle N is valid on cycle N+1. load_bias and enable must not be
//  high together; load_bias wins inside pe_unit and would skip the MAC.
//
//  scores takes accumulator bits [SCORE_LSB +: SCORE_WIDTH]. SCORE_LSB
//  stays 0 until the fixed-point scale is frozen. No softmax here.
//
// Parameters:
//  HIDDEN       - hidden units streamed into the dot product
//  VOCAB        - symbols, and the number of processing elements
//  N_JOBS       - weight blocks stored back to back (AM then LM)
//=====================================================================

module output_tile #(
    parameter HIDDEN = 256,
    parameter VOCAB = 30,
    parameter N_JOBS = 2,
    parameter X_WIDTH = 8,
    parameter W_WIDTH = 6,
    parameter BIAS_WIDTH = 16,
    parameter ACC_WIDTH = 24,
    parameter SCORE_WIDTH = 8,
    parameter SCORE_LSB = 0,
    parameter WEIGHT_DEPTH = HIDDEN * N_JOBS,
    parameter WEIGHT_AW = $clog2(WEIGHT_DEPTH),
    parameter BIAS_AW = (N_JOBS <= 1) ? 1 : $clog2(N_JOBS)
) (
    input wire clk,
    input wire rst,
    input wire enable,
    input wire load_bias,

    input wire signed [X_WIDTH-1:0] h_element,

    input wire [WEIGHT_AW-1:0] weight_addr,
    input wire weight_we,
    input wire [VOCAB*W_WIDTH-1:0] weight_din,

    input wire [BIAS_AW-1:0] bias_addr,
    input wire bias_we,
    input wire signed [VOCAB*BIAS_WIDTH-1:0] bias_din,

    output wire signed [VOCAB*ACC_WIDTH-1:0] acc,
    output wire signed [VOCAB*SCORE_WIDTH-1:0] scores
);

    wire [VOCAB*W_WIDTH-1:0] weight_row;
    reg signed [VOCAB*BIAS_WIDTH-1:0] bias_mem [0:N_JOBS-1];
    reg signed [VOCAB*BIAS_WIDTH-1:0] bias_q;

    weight_bram #(
        .WEIGHT_COUNT(WEIGHT_DEPTH),
        .WEIGHT_ELEMENT_WIDTH(W_WIDTH),
        .NEURON_COUNT(VOCAB),
        .ADDR_WIDTH(WEIGHT_AW)
    ) proj_weights (
        .clk(clk),
        .write_enable(weight_we),
        .address(weight_addr),
        .data_in(weight_din),
        .data_out(weight_row)
    );

    always @(posedge clk) begin
        if (bias_we)
            bias_mem[bias_addr] <= bias_din;
        bias_q <= bias_mem[bias_addr];
    end

    genvar g;
    generate
        for (g = 0; g < VOCAB; g = g + 1) begin: proj_pe
            wire signed [BIAS_WIDTH-1:0] bias_lane;
            wire signed [ACC_WIDTH-1:0] bias_ext;
            wire signed [ACC_WIDTH-1:0] acc_lane;

            assign bias_lane = bias_q[g*BIAS_WIDTH +: BIAS_WIDTH];
            assign bias_ext = bias_lane;
            assign acc[g*ACC_WIDTH +: ACC_WIDTH] = acc_lane;
            assign scores[g*SCORE_WIDTH +: SCORE_WIDTH] = acc_lane[SCORE_LSB +: SCORE_WIDTH];

            pe_unit #(
                .X_ELEMENT_WIDTH(X_WIDTH),
                .WEIGHT_ELEMENT_WIDTH(W_WIDTH),
                .OUTPUT_ELEMENT_WIDTH(ACC_WIDTH)
            ) pe (
                .clk(clk),
                .rst(rst),
                .enable(enable),
                .load_bias(load_bias),
                .bias(bias_ext),
                .x_input(h_element),
                .weight(weight_row[g*W_WIDTH +: W_WIDTH]),
                .pe_out(acc_lane)
            );
        end
    endgenerate

endmodule
