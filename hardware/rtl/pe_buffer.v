`timescale 1ns / 1ps

//=====================================================================
// Module: pe_buffer.v
// Description:
//  Four gate memories between the PE array and the LSTM EPU.
//  pe_in0 and pe_in1 each carry PE_COUNT lanes. capture writes every
//  lane in one cycle: gate_pair 0 stores them as i and f, gate_pair 1
//  stores them as o and c. addr then reads one neuron from all four
//  memories. pe_i, pe_f, pe_o, and pe_c are that 16-bit read. After the
//  capture the PE array can be reused; the lanes live here.
//
// Parameters:
//  GATE_WIDTH   - bits stored per lane
//  PE_COUNT     - neurons captured from each PE array
//
// Ports:
//  clk          - clock
//  rst          - reset, clears every lane
//  capture      - when high, copy both PE buses into the selected pair
//  gate_pair    - 0 = i from pe_in0 and f from pe_in1; 1 = o and c
//  pe_in0       - packed array-0 result, lane 0 in the LSBs
//  pe_in1       - packed array-1 result, lane 0 in the LSBs
//  addr         - neuron read by pe_i, pe_f, pe_o, pe_c
//  pe_i/f/o/c   - gate pre-activations for addr, combinational
//=====================================================================

module pe_buffer #(
    parameter GATE_WIDTH = 16,
    parameter PE_COUNT = 256,
    parameter ADDR_WIDTH = (PE_COUNT <= 1) ? 1 : $clog2(PE_COUNT)
) (
    input wire clk,
    input wire rst,
    input wire capture,
    input wire gate_pair,

    input wire signed [PE_COUNT*GATE_WIDTH-1:0] pe_in0,
    input wire signed [PE_COUNT*GATE_WIDTH-1:0] pe_in1,
    input wire [ADDR_WIDTH-1:0] addr,

    output wire signed [GATE_WIDTH-1:0] pe_i,
    output wire signed [GATE_WIDTH-1:0] pe_f,
    output wire signed [GATE_WIDTH-1:0] pe_o,
    output wire signed [GATE_WIDTH-1:0] pe_c
);

    reg signed [GATE_WIDTH-1:0] mem_i [0:PE_COUNT-1];
    reg signed [GATE_WIDTH-1:0] mem_f [0:PE_COUNT-1];
    reg signed [GATE_WIDTH-1:0] mem_o [0:PE_COUNT-1];
    reg signed [GATE_WIDTH-1:0] mem_c [0:PE_COUNT-1];

    integer n;

    always @(posedge clk) begin
        if (rst) begin
            for (n = 0; n < PE_COUNT; n = n + 1) begin
                mem_i[n] <= {GATE_WIDTH{1'b0}};
                mem_f[n] <= {GATE_WIDTH{1'b0}};
                mem_o[n] <= {GATE_WIDTH{1'b0}};
                mem_c[n] <= {GATE_WIDTH{1'b0}};
            end
        end else if (capture) begin
            for (n = 0; n < PE_COUNT; n = n + 1) begin
                if (!gate_pair) begin
                    mem_i[n] <= pe_in0[n*GATE_WIDTH +: GATE_WIDTH];
                    mem_f[n] <= pe_in1[n*GATE_WIDTH +: GATE_WIDTH];
                end else begin
                    mem_o[n] <= pe_in0[n*GATE_WIDTH +: GATE_WIDTH];
                    mem_c[n] <= pe_in1[n*GATE_WIDTH +: GATE_WIDTH];
                end
            end
        end
    end

    assign pe_i = mem_i[addr];
    assign pe_f = mem_f[addr];
    assign pe_o = mem_o[addr];
    assign pe_c = mem_c[addr];

endmodule
