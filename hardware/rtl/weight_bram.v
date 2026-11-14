`timescale 1ns / 1ps

//=====================================================================
// Module: weight_bram.v
// Description:
//  Weight storage memory inferred as BRAM and UltraRAM. Stores one weight matrix
//  as packed rows. Each row (address) holds all NEURON_COUNT neurons'
//  weights for one input position, packed side by side. A synchronous
//  read/write gives one full row per access. Weights are loaded in via
//  the data_in port and can be read out on the data_out port.
//
// Parameters:
//  WEIGHT_COUNT            - number of rows (input weight entries)
//  WEIGHT_ELEMENT_WIDTH    - bit width of a single weight (6-bit quantized)
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
    parameter ADDR_WIDTH = $clog2(WEIGHT_COUNT),
    parameter URAM_DEPTH = 4096
) (
    input wire clk,
    input wire write_enable,

    input wire [ADDR_WIDTH-1:0] address,
    input wire [NEURON_COUNT*WEIGHT_ELEMENT_WIDTH-1:0] data_in,
    output wire [NEURON_COUNT*WEIGHT_ELEMENT_WIDTH-1:0] data_out
);
    localparam ROW_W = NEURON_COUNT * WEIGHT_ELEMENT_WIDTH;
    localparam LO_DEPTH = (WEIGHT_COUNT < URAM_DEPTH) ? WEIGHT_COUNT : URAM_DEPTH;
    localparam HI_DEPTH = WEIGHT_COUNT - LO_DEPTH;
    localparam LO_AW = (LO_DEPTH <= 1) ? 1 : $clog2(LO_DEPTH);

    // Rows below URAM_DEPTH live in UltraRAM, the rest in block RAM.
    // Both reads are registered, so the latency is still one cycle.
    wire in_hi = (address >= LO_DEPTH);
    reg sel_hi;
    reg [ROW_W-1:0] lo_q;

    (* ram_style = "ultra" *)
    reg [ROW_W-1:0] mem_lo [0:LO_DEPTH-1];

    always @(posedge clk) begin
        if (write_enable && !in_hi)
            mem_lo[address[LO_AW-1:0]] <= data_in;
        lo_q <= mem_lo[address[LO_AW-1:0]];
        sel_hi <= in_hi;
    end

    generate
        if (HI_DEPTH > 0) begin : hi
            wire [ADDR_WIDTH-1:0] hi_addr = address - LO_DEPTH;
            reg [ROW_W-1:0] hi_q;

            (* ram_style = "block" *)
            reg [ROW_W-1:0] mem_hi [0:HI_DEPTH-1];

            always @(posedge clk) begin
                if (write_enable && in_hi)
                    mem_hi[hi_addr] <= data_in;
                hi_q <= mem_hi[hi_addr];
            end

            assign data_out = sel_hi ? hi_q : lo_q;
        end else begin : lo_only
            assign data_out = lo_q;
        end
    endgenerate
endmodule