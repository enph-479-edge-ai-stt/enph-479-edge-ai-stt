`timescale 1ns / 1ps

`timescale 1ns / 1ps

//=====================================================================
// Module: context_memory.v
// Description:
//   Context (state) memory inferred as BRAM. Stores the LSTM state
//   (cell state c_t and hidden state h_t) for each neuron, across
//   layers and beam hypotheses. Written by the EPU each time step and
//   read back at the start of the next step. One entry per neuron-state,
//   packed as {c_t (16-bit), h_t (8-bit)} = 24 bits.
//
// Parameters:
//   STATE_WIDTH  - bits per entry (16-bit c_t + 8-bit h_t = 24)
//   DEPTH        - number of state entries (beams x layers x neurons)
//   ADDR_WIDTH   - address width, sized to index DEPTH entries
//
// Ports:
//   clk          - clock
//   write_enable - when high, write data_in into the addressed entry
//   address      - which state entry to read/write
//   data_in      - state value to write ({c_t, h_t})
//   data_out     - state value read out
//=====================================================================

module context_memory #(
    parameter STATE_WIDTH = 24,
    parameter DEPTH = 66304,
    parameter ADDR_WIDTH = $clog(DEPTH)
) (
    input wire clk,
    input wire write_enable,

    input wire [ADDR_WIDTH-1:0] address,
    input wire [STATE_WIDTH-1:0] data_in,
    output reg [STATE_WIDTH-1:0] data_out
);
    reg [STATE_WIDTH-1:0] context_memory_array [0:DEPTH-1];

    always @(posedge clk) begin
        if (write_enable)
            context_memory_array[address] <= data_in;
        data_out <= context_memory_array[address];
    end
endmodule