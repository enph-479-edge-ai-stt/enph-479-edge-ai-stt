`timescale 1ns / 1ps

//=====================================================================
// Module: weight_bram.v
// Description:
//  Weight storage memory inferred as BRAM. Stores one weight matrix
//  as packed rows. Each row (address) holds all NEURON_COUNT neurons'
//  weights for one input position, packed side by side. A synchronous
//  read/write gives one full row per access. Weights are loaded in via
//  the data_in port and can be read out on the data_out port.
//
// Parameters:
//  WEIGHT_COUNT            - number of rows (input weight entries)
//  WEIGHT_ELEMENT_WIDTH    - bit width of a sigle weight (6-bit quantized)
//  NEURON_COUNT            - number of neurons
//  ADDR_WIDTH              - address width, sized to WEIGHT_COUNT ROWS
// Ports:
//  clk             - clock
//  write_enable    - when high, write data into addressed row
//  address         - address of row to read/write
//  data_in         - full packed row to write in
//  data_out        - full packed row to read out
//=====================================================================

module weight_bram #(
    parameter WEIGHT_COUNT = 123,
    parameter WEIGHT_ELEMENT_WIDTH = 6,
    parameter NEURON_COUNT = 256,
    parameter ADDR_WIDTH = $clog2(WEIGHT_COUNT)
) (
    input wire clk,
    input wire write_enable,

    input wire [ADDR_WIDTH-1:0] address,
    input wire [NEURON_COUNT*WEIGHT_ELEMENT_WIDTH-1:0] data_in,
    output reg [NEURON_COUNT*WEIGHT_ELEMENT_WIDTH-1:0] data_out
);
    reg [NEURON_COUNT*WEIGHT_ELEMENT_WIDTH-1:0] weight_bram_memory [0:WEIGHT_COUNT-1];

    always @(posedge clk) begin
        if (write_enable)
            weight_bram_memory[address] <= data_in;
        data_out <= weight_bram_memory[address];
    end
endmodule