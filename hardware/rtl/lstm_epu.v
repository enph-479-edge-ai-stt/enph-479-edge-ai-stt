`timescale 1ns / 1ps

//=====================================================================
// Module: lstm_epu.v
// Description:
//  LSTM (EPU) Extra Processing Unit. This module manages LSTM operations 
//  of applying activation, and peepholing to the output of the PE Buffers.
//
// Parameters:
//  OUTPUT_ELEMENT_WIDTH    - bit width of PE output
//
// Ports:
//  clk             - clock
//  rst             - reset signal
//  enable          - when high, run all processing elements in the array
//  load_bias       - when high, load the bias into the processing elements
//  data_input      - input data to all the processing elements
//  weight_input0   - input weights for the first array of processing elements
//  weight_input1   - input weights for the second array of processing elements
//  bias_input0     - input biases for the first array of processing elements
//  bias_input1     - input biases for the second array of processing elements
//  pe_out0         - concatenated output of the first array of processing elements
//  pe_out1         - concatenated output of the second array of processing elements
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


    wire signed [OUTPUT_ELEMENT_WIDTH-1:0] activation_i_input = pe_input_i * peephole_weight[PEEPHOLE_WEIGHT_WIDTH-1:PEEPHOLE_WEIGHT_WIDTH-OUTPUT_ELEMENT_WIDTH];
    wire signed [OUTPUT_ELEMENT_WIDTH-1:0] activation_f_input,
    wire signed [OUTPUT_ELEMENT_WIDTH-1:0] activation_o_input,
    wire signed [OUTPUT_ELEMENT_WIDTH-1:0] activation_i_input


    always @(posedge clk) begin
        if (enabled)
            hidden_state_output <= pe_out + x_input * weight;

    end
    
endmodule
