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

    // Activation inputs
    // i = input gate, f = forget gate, o = output gate
    // take the last hidden state and multiply it by the respective peephole weight
    // assume peehole weight is cocatenated for i then f then o MSB to LSB
    wire signed [OUTPUT_ELEMENT_WIDTH-1:0] activation_i_input = pe_input_i + old_hidden_state * peephole_weight[PEEPHOLE_WEIGHT_WIDTH-OUTPUT_ELEMENT_WIDTH*0-1:PEEPHOLE_WEIGHT_WIDTH-1*OUTPUT_ELEMENT_WIDTH];
    wire signed [OUTPUT_ELEMENT_WIDTH-1:0] activation_f_input = pe_input_f + old_hidden_state * peephole_weight[PEEPHOLE_WEIGHT_WIDTH-OUTPUT_ELEMENT_WIDTH*1-1:PEEPHOLE_WEIGHT_WIDTH-2*OUTPUT_ELEMENT_WIDTH];
    wire signed [OUTPUT_ELEMENT_WIDTH-1:0] activation_o_input = pe_input_o + pe_input_c *peephole_weight[PEEPHOLE_WEIGHT_WIDTH-OUTPUT_ELEMENT_WIDTH*2-1:PEEPHOLE_WEIGHT_WIDTH-3*OUTPUT_ELEMENT_WIDTH];

    // Wires to hold the outputs from the sigmoid LUTs
    wire [OUTPUT_ELEMENT_WIDTH-1:0] sigmoid_i_out;
    wire [OUTPUT_ELEMENT_WIDTH-1:0] sigmoid_f_out;
    wire [OUTPUT_ELEMENT_WIDTH-1:0] sigmoid_o_out;
    wire [OUTPUT_ELEMENT_WIDTH-1:0] tanh_c_out;

    // Instantiate Sigmoid LUT for input gate (i)
    sigmoid_lut #(
        .LUT_BIT_WIDTH(8),
        .LUT_PRECISION(16),
        .INPUT_BIT_WIDTH(4), 
        .SIGMOID_OUTPUT_WIDTH(OUTPUT_ELEMENT_WIDTH)
    ) sig_i (
        // Note: Slicing the upper 4 bits as an example. Adjust [7:4] based on your fixed-point radix.
        .index(activation_i_input[7:4]), 
        .sigmoid_output(sigmoid_i_out)
    );

    // Instantiate Sigmoid LUT for forget gate (f)
    sigmoid_lut #(
        .LUT_BIT_WIDTH(8),
        .LUT_PRECISION(16),
        .INPUT_BIT_WIDTH(4),
        .SIGMOID_OUTPUT_WIDTH(OUTPUT_ELEMENT_WIDTH)
    ) sig_f (
        .index(activation_f_input[7:4]),
        .sigmoid_output(sigmoid_f_out)
    );

    // Instantiate Sigmoid LUT for output gate (o)
    sigmoid_lut #(
        .LUT_BIT_WIDTH(8),
        .LUT_PRECISION(16),
        .INPUT_BIT_WIDTH(4),
        .SIGMOID_OUTPUT_WIDTH(OUTPUT_ELEMENT_WIDTH)
    ) sig_o (
        .index(activation_o_input[7:4]),
        .sigmoid_output(sigmoid_o_out)
    );

    // Instantiate Tanh LUT for cell state (c)
    tanh_lut #(
        .LUT_BIT_WIDTH(8),
        .LUT_PRECISION(16),
        .INPUT_BIT_WIDTH(4),
        .TANH_OUTPUT_WIDTH(OUTPUT_ELEMENT_WIDTH)
    ) tanh_c (
        .index(pe_input_c[7:4]),
        .tanh_output(tanh_c_out)
    );

    //Calculate f_t * c_{t-1} 
    // sigmoid_f_out (8-bit) * old_hidden_state (16-bit) yields a 24-bit result
    wire signed [OUTPUT_ELEMENT_WIDTH+HIDDEN_STATE_WIDTH-1:0] f_times_c_prev;
    assign f_times_c_prev = $signed(sigmoid_f_out) * $signed(old_hidden_state);

    // Calculate i_t * c_tilde_t
    // sigmoid_i_out (8-bit) * tanh_c_out (8-bit) yields a 16-bit result
    wire signed [2*OUTPUT_ELEMENT_WIDTH-1:0] i_times_c_tilde;
    assign i_times_c_tilde = $signed(sigmoid_i_out) * $signed(tanh_c_out);

    // To add the two products, you must align their decimal points based on your fixed-point radix.
    // Example: Truncating the 24-bit wire down to 16 bits to match i_times_c_tilde
    // NOTE: Adjust the bit slice [23:8] below according to your specific fixed-point scaling factor
    assign hidden_state_output = f_times_c_prev[23:8] + i_times_c_tilde;

    // Instantiate second Tanh LUT for tanh(c_t)
    wire signed [OUTPUT_ELEMENT_WIDTH-1:0] tanh_ct_out;
    tanh_lut #(
        .LUT_BIT_WIDTH(8),
        .LUT_PRECISION(16),
        .INPUT_BIT_WIDTH(4),
        .TANH_OUTPUT_WIDTH(OUTPUT_ELEMENT_WIDTH)
    ) tanh_ct_inst (
        // NOTE: Adjust the index slicing based on your 16-bit c_t fixed-point representation
        .index(hidden_state_output[11:8]), 
        .tanh_output(tanh_ct_out)
    );

    //Calculate final output state (h_t = o_t * tanh(c_t))
    // sigmoid_o_out (8-bit) * tanh_ct_out (8-bit) yields a 16-bit result
    wire signed [2*OUTPUT_ELEMENT_WIDTH-1:0] h_t_full;
    assign h_t_full = $signed(sigmoid_o_out) * $signed(tanh_ct_out);
    
    // Adjust [7:0] based on your fixed-point radix
    assign output_state = h_t_full[7:0];


    
endmodule
