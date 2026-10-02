`timescale 1ns / 1ps

//=====================================================================
// Module: pe_array.v
// Description:
//  Processing Element (PE) Array. This module contains 2 arrays 
//  of 256 PE units each. Both arrays are connected to the same input data.
//  The 2 arrays work in parallel to compute the outputs that are stored
//  in the 4 buffers.
//
// Parameters:
//  OUTPUT_ELEMENT_WIDTH    - bit width of PE output
//  parameter PE_ARRAY_SIZE - size of singular PE array
//  parameter DATA_ELEMENT_WIDTH = 8 - data element width
//  parameter WEIGHT_INPUT_WIDTH = 6 - weight input width
//  parameter BIAS_INPUT_WIDTH = 16 - bias input width
//  parameter OUTPUT_ELEMENT_WIDTH = 16 - output element width

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

module pe_array #(
    
    parameter PE_ARRAY_SIZE = 256,
    parameter DATA_ELEMENT_WIDTH = 8,
    parameter WEIGHT_INPUT_WIDTH = 6,
    parameter BIAS_INPUT_WIDTH = 16,
    parameter OUTPUT_ELEMENT_WIDTH = 16
) (
    
    input wire clk,
    input wire rst,
    input wire enable,
    input wire load_bias,
    
    
    input wire signed [DATA_ELEMENT_WIDTH-1:0] data_input,
    input wire signed [WEIGHT_INPUT_WIDTH * PE_ARRAY_SIZE-1:0] weight_input0,
    input wire signed [WEIGHT_INPUT_WIDTH * PE_ARRAY_SIZE-1:0] weight_input1,
    input wire signed [BIAS_INPUT_WIDTH * PE_ARRAY_SIZE-1:0] bias_input0,
    input wire signed [BIAS_INPUT_WIDTH * PE_ARRAY_SIZE-1:0] bias_input1,

    output wire signed [OUTPUT_ELEMENT_WIDTH * PE_ARRAY_SIZE-1:0] pe_out0,
    output wire signed [OUTPUT_ELEMENT_WIDTH * PE_ARRAY_SIZE-1:0] pe_out1
);

    genvar i;
    
    generate
        for (i = 0; i < PE_ARRAY_SIZE; i = i + 1) begin: pe_array_loop
            
            pe_unit #(
                //Common parameters

                .X_ELEMENT_WIDTH(DATA_ELEMENT_WIDTH),
                .WEIGHT_ELEMENT_WIDTH(WEIGHT_INPUT_WIDTH),
                .OUTPUT_ELEMENT_WIDTH(OUTPUT_ELEMENT_WIDTH)

            ) pe_unit_inst_0 (
                //Common inputs
                .clk(clk),  
                .rst(rst),
                .enable(enable),
                .load_bias(load_bias),
                .x_input(data_input),

                //Individual inputs
                .weight(weight_input0[i * WEIGHT_INPUT_WIDTH +: WEIGHT_INPUT_WIDTH]),
                .bias(bias_input0[i * BIAS_INPUT_WIDTH   +: BIAS_INPUT_WIDTH]),
                .pe_out(pe_out0[i * OUTPUT_ELEMENT_WIDTH +: OUTPUT_ELEMENT_WIDTH])            
                );
            
            pe_unit #(
                //Common parameters

                .X_ELEMENT_WIDTH(DATA_ELEMENT_WIDTH),
                .WEIGHT_ELEMENT_WIDTH(WEIGHT_INPUT_WIDTH),
                .OUTPUT_ELEMENT_WIDTH(OUTPUT_ELEMENT_WIDTH)

            ) pe_unit_inst_1 (
                //Common inputs
                .clk(clk),  
                .rst(rst),
                .enable(enable),
                .load_bias(load_bias),
                .x_input(data_input),

                //Individual inputs
                .weight(weight_input1[i * WEIGHT_INPUT_WIDTH +: WEIGHT_INPUT_WIDTH]),
                .bias(bias_input1[i * BIAS_INPUT_WIDTH   +: BIAS_INPUT_WIDTH]),
                .pe_out(pe_out1[i * OUTPUT_ELEMENT_WIDTH +: OUTPUT_ELEMENT_WIDTH])            
                );
        
        
        end
    endgenerate


endmodule
