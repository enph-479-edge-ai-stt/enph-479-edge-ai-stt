`timescale 1ns / 1ps

//=====================================================================
// Module: lstm_tile.v
// Description:
//  Datapath for one LSTM tile, shared by the acoustic model and the
//  character LM. No state machine lives here. The controller drives
//  every enable and address. Both networks stay in the memories; a
//  model switch is a different weight address and context address.
//
//  Weight, bias, and context reads are synchronous. The address on
//  cycle N is valid at the data output on cycle N+1. The gate-buffer
//  read is combinational. The EPU arithmetic is combinational too
//  (lstm_iterm, lstm_cell, lstm_h), and this tile registers it in
//  three stages, and the i-term multiply is one stage after the gates:
//    1. c_prev and the gate values (mux and sigmoid, not the multiply)
//    2. i_term from those gates, and c_prev delayed to match it
//    3. c_t, and the o gate delayed to match it
//    4. {c_t, h_t}, which is what c_t and h_t ports present
//  context_we stores that pair on the cycle it is already valid.
//
//  h used as a multiply input is registered twice after the context
//  read. h_near sits on the RAM output. h_mac fans out to the PE array
//  and to the output layer. The controller presents that context address
//  two cycles before the matching weight address.
//
//  Per layer the controller is expected to:
//    1. Present the i/f bias address, then pulse load_bias.
//    2. MAC d_in (sel_in = 0) across the input vector.
//    3. MAC h_{t-1} (sel_in = 1) from the context word.
//    4. Pulse capture_gates with gate_pair = 0. That copies both
//       accumulator buses into memories i and f.
//    5. Repeat 1-4 for gates o and c with gate_pair = 1.
//    6. For neuron_addr = 0 .. PE_COUNT-1: present the source context
//       address, hold neuron_addr while that word is valid, then pulse
//       context_we once the registered {c_t, h_t} is valid.
//
//  d_in is shifted left by x_shift before it reaches the PEs, which is
//  the same as shifting every product (shared/specs/weight_image.md:
//  layer 0's x-phase is shifted 7 - x_frac). One shifter here instead of
//  one per PE; the PEs take PE_X_WIDTH-bit data to hold the result.
//  h (sel_in = 1) is never shifted.
//
//  Each accumulator is rescaled to the EPU's table input before it is
//  captured: rounded half up at the model's gate shift and saturated to
//  9 bits at 2^-5 (+-8). The shift is lstm_frac + 7 - 5 from the
//  manifest, 9 for the AM (2^-14) and 6 for the CM (2^-11); lm picks
//  it and must hold through capture_gates. Formats in
//  shared/specs/weight_image.md ("Activations and cell").
//
//  Context entries are {c_t, h_t}. c_t is the top C_WIDTH bits.
//=====================================================================

module lstm_tile #(
    parameter PE_COUNT = 256,
    parameter X_WIDTH = 8,
    parameter PE_X_WIDTH = 16,
    parameter W_WIDTH = 6,
    parameter BIAS_WIDTH = 24,
    parameter ACC_WIDTH = 24,
    parameter C_WIDTH = 16,
    parameter H_WIDTH = 8,
    parameter AM_GATE_SHIFT = 9,
    parameter LM_GATE_SHIFT = 6,
    parameter WEIGHT_DEPTH = 4402,
    parameter CONTEXT_DEPTH = 66304,
    parameter N_LSTM_LAYERS = 5,
    parameter BIAS_DEPTH = N_LSTM_LAYERS * 2,
    parameter WEIGHT_AW = $clog2(WEIGHT_DEPTH),
    parameter CONTEXT_AW = $clog2(CONTEXT_DEPTH),
    parameter BIAS_AW = $clog2(BIAS_DEPTH),
    parameter NEURON_AW = (PE_COUNT <= 1) ? 1 : $clog2(PE_COUNT)
) (
    input wire clk,
    input wire rst,

    input wire pe_enable,
    input wire load_bias,
    input wire sel_in,
    input wire signed [X_WIDTH-1:0] d_in,
    input wire [2:0] x_shift,
    input wire lm,

    input wire [WEIGHT_AW-1:0] weight_addr,
    input wire weight0_we,
    input wire weight1_we,
    input wire [PE_COUNT*W_WIDTH-1:0] weight0_din,
    input wire [PE_COUNT*W_WIDTH-1:0] weight1_din,

    input wire [BIAS_AW-1:0] bias_addr,
    input wire bias_we,
    input wire signed [PE_COUNT*BIAS_WIDTH-1:0] bias0_din,
    input wire signed [PE_COUNT*BIAS_WIDTH-1:0] bias1_din,

    input wire capture_gates,
    input wire gate_pair,
    input wire [NEURON_AW-1:0] neuron_addr,

    input wire [CONTEXT_AW-1:0] context_addr,
    input wire context_we,
    input wire context_use_ext,
    input wire [C_WIDTH+H_WIDTH-1:0] context_ext_din,

    output wire signed [C_WIDTH-1:0] c_t,
    output wire signed [H_WIDTH-1:0] h_t,
    output wire [C_WIDTH+H_WIDTH-1:0] context_dout,
    output wire signed [8:0] pe_i,
    output wire signed [8:0] pe_f,
    output wire signed [8:0] pe_o,
    output wire signed [8:0] pe_c,
    output wire signed [H_WIDTH-1:0] h_mac
);

    localparam STATE_WIDTH = C_WIDTH + H_WIDTH;
    localparam GATE_WIDTH = 9;

    wire [PE_COUNT*W_WIDTH-1:0] weight0;
    wire [PE_COUNT*W_WIDTH-1:0] weight1;
    wire signed [PE_COUNT*ACC_WIDTH-1:0] pe_out0;
    wire signed [PE_COUNT*ACC_WIDTH-1:0] pe_out1;
    wire signed [PE_COUNT*GATE_WIDTH-1:0] gate_bus0;
    wire signed [PE_COUNT*GATE_WIDTH-1:0] gate_bus1;
    wire signed [X_WIDTH-1:0] h_from_context;
    wire signed [PE_X_WIDTH-1:0] x_shifted;
    wire signed [PE_X_WIDTH-1:0] pe_data;
    wire signed [C_WIDTH-1:0] c_prev;
    wire [STATE_WIDTH-1:0] context_din;
    wire signed [GATE_WIDTH-1:0] term_i;
    wire signed [GATE_WIDTH-1:0] term_f;
    wire signed [GATE_WIDTH-1:0] term_o;
    wire signed [7:0] tanh_c;
    wire signed [12:0] i_term_c;
    wire signed [16:0] i_times_g;
    wire signed [C_WIDTH-1:0] c_t_c;
    wire signed [H_WIDTH-1:0] h_c;

    // h_near is the only load on the context-RAM cascade. KEEP keeps it
    // from folding into h_mac, which has to drive every multiplier.
    (* KEEP = "true" *) reg signed [H_WIDTH-1:0] h_near;
    reg signed [H_WIDTH-1:0] h_fan;
    (* KEEP = "true" *) reg signed [C_WIDTH-1:0] c_prev_r;
    reg signed [C_WIDTH-1:0] c_prev_d;
    (* KEEP = "true" *) reg signed [GATE_WIDTH-1:0] gate_i_r;
    (* KEEP = "true" *) reg signed [GATE_WIDTH-1:0] gate_f_r;
    (* KEEP = "true" *) reg signed [GATE_WIDTH-1:0] gate_o_r;
    (* KEEP = "true" *) reg signed [7:0] tanh_c_r;
    reg signed [12:0] i_term_r;
    reg signed [C_WIDTH-1:0] c_t_r;
    reg signed [GATE_WIDTH-1:0] gate_o_d;
    reg signed [C_WIDTH-1:0] c_t_q;
    reg signed [H_WIDTH-1:0] h_t_q;

    (* ram_style = "distributed" *)
    reg signed [PE_COUNT*BIAS_WIDTH-1:0] bias0_mem [0:BIAS_DEPTH-1];
    (* ram_style = "distributed" *)
    reg signed [PE_COUNT*BIAS_WIDTH-1:0] bias1_mem [0:BIAS_DEPTH-1];
    reg signed [PE_COUNT*BIAS_WIDTH-1:0] bias0_q;
    reg signed [PE_COUNT*BIAS_WIDTH-1:0] bias1_q;

    assign h_from_context = context_dout[H_WIDTH-1:0];
    assign h_mac = h_fan;
    assign c_prev = context_dout[STATE_WIDTH-1:H_WIDTH];
    // d_in is sign-extended to PE_X_WIDTH before the shift.
    assign x_shifted = d_in <<< x_shift;
    assign pe_data = sel_in ? h_fan : x_shifted;
    assign i_times_g = gate_i_r * tanh_c_r;
    assign i_term_c = (i_times_g + 17'sd8) >>> 4;
    assign c_t = c_t_q;
    assign h_t = h_t_q;
    assign context_din = context_use_ext ? context_ext_din : {c_t, h_t};

    weight_bram #(
        .WEIGHT_COUNT(WEIGHT_DEPTH),
        .WEIGHT_ELEMENT_WIDTH(W_WIDTH),
        .NEURON_COUNT(PE_COUNT),
        .ADDR_WIDTH(WEIGHT_AW)
    ) weight0_bram (
        .clk(clk),
        .write_enable(weight0_we),
        .address(weight_addr),
        .data_in(weight0_din),
        .data_out(weight0)
    );

    weight_bram #(
        .WEIGHT_COUNT(WEIGHT_DEPTH),
        .WEIGHT_ELEMENT_WIDTH(W_WIDTH),
        .NEURON_COUNT(PE_COUNT),
        .ADDR_WIDTH(WEIGHT_AW)
    ) weight1_bram (
        .clk(clk),
        .write_enable(weight1_we),
        .address(weight_addr),
        .data_in(weight1_din),
        .data_out(weight1)
    );

    always @(posedge clk) begin
        if (bias_we) begin
            bias0_mem[bias_addr] <= bias0_din;
            bias1_mem[bias_addr] <= bias1_din;
        end
        bias0_q <= bias0_mem[bias_addr];
        bias1_q <= bias1_mem[bias_addr];
    end

    always @(posedge clk) begin
        if (rst) begin
            h_near <= 0;
            h_fan <= 0;
        end else begin
            h_near <= h_from_context;
            h_fan <= h_near;
        end
    end

    pe_array #(
        .PE_ARRAY_SIZE(PE_COUNT),
        .DATA_ELEMENT_WIDTH(PE_X_WIDTH),
        .WEIGHT_INPUT_WIDTH(W_WIDTH),
        .OUTPUT_ELEMENT_WIDTH(ACC_WIDTH)
    ) pe_arrays (
        .clk(clk),
        .rst(rst),
        .enable(pe_enable),
        .load_bias(load_bias),
        .data_input(pe_data),
        .weight_input0(weight0),
        .weight_input1(weight1),
        .bias_input0(bias0_q),
        .bias_input1(bias1_q),
        .pe_out0(pe_out0),
        .pe_out1(pe_out1)
    );

    genvar g;
    generate
        for (g = 0; g < PE_COUNT; g = g + 1) begin: gate_slice
            wire signed [ACC_WIDTH-1:0] acc0;
            wire signed [ACC_WIDTH-1:0] acc1;
            assign acc0 = pe_out0[g*ACC_WIDTH +: ACC_WIDTH];
            assign acc1 = pe_out1[g*ACC_WIDTH +: ACC_WIDTH];

            round_sat #(
                .IN_WIDTH(ACC_WIDTH),
                .OUT_WIDTH(GATE_WIDTH),
                .SHIFT_AM(AM_GATE_SHIFT),
                .SHIFT_LM(LM_GATE_SHIFT)
            ) slice0 (
                .lm(lm),
                .din(acc0),
                .dout(gate_bus0[g*GATE_WIDTH +: GATE_WIDTH])
            );

            round_sat #(
                .IN_WIDTH(ACC_WIDTH),
                .OUT_WIDTH(GATE_WIDTH),
                .SHIFT_AM(AM_GATE_SHIFT),
                .SHIFT_LM(LM_GATE_SHIFT)
            ) slice1 (
                .lm(lm),
                .din(acc1),
                .dout(gate_bus1[g*GATE_WIDTH +: GATE_WIDTH])
            );
        end
    endgenerate

    pe_buffer #(
        .GATE_WIDTH(GATE_WIDTH),
        .PE_COUNT(PE_COUNT),
        .ADDR_WIDTH(NEURON_AW)
    ) gates (
        .clk(clk),
        .rst(rst),
        .capture(capture_gates),
        .gate_pair(gate_pair),
        .pe_in0(gate_bus0),
        .pe_in1(gate_bus1),
        .addr(neuron_addr),
        .pe_i(pe_i),
        .pe_f(pe_f),
        .pe_o(pe_o),
        .pe_c(pe_c)
    );

    // Gates are registered off the mux. The i-term multiply uses those
    // registers on the next cycle, with c_prev delayed to stay aligned.
    lstm_iterm terms (
        .pe_i(pe_i),
        .pe_f(pe_f),
        .pe_o(pe_o),
        .pe_c(pe_c),
        .gate_i(term_i),
        .gate_f(term_f),
        .gate_o(term_o),
        .tanh_c(tanh_c)
    );

    always @(posedge clk) begin
        if (rst) begin
            c_prev_r <= 0;
            gate_i_r <= 0;
            gate_f_r <= 0;
            gate_o_r <= 0;
            tanh_c_r <= 0;
        end else begin
            c_prev_r <= c_prev;
            gate_i_r <= term_i;
            gate_f_r <= term_f;
            gate_o_r <= term_o;
            tanh_c_r <= tanh_c;
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            i_term_r <= 0;
            c_prev_d <= 0;
        end else begin
            i_term_r <= i_term_c;
            c_prev_d <= c_prev_r;
        end
    end

    lstm_cell u_cell (
        .c_prev(c_prev_d),
        .gate_f(gate_f_r),
        .i_term(i_term_r),
        .c_t(c_t_c)
    );

    always @(posedge clk) begin
        if (rst) begin
            c_t_r <= 0;
            gate_o_d <= 0;
        end else begin
            c_t_r <= c_t_c;
            gate_o_d <= gate_o_r;
        end
    end

    lstm_h hidden (
        .c_t(c_t_r),
        .gate_o(gate_o_d),
        .h_t(h_c)
    );

    always @(posedge clk) begin
        if (rst) begin
            c_t_q <= 0;
            h_t_q <= 0;
        end else begin
            c_t_q <= c_t_r;
            h_t_q <= h_c;
        end
    end

    context_memory #(
        .STATE_WIDTH(STATE_WIDTH),
        .DEPTH(CONTEXT_DEPTH),
        .ADDR_WIDTH(CONTEXT_AW)
    ) context_store (
        .clk(clk),
        .write_enable(context_we),
        .address(context_addr),
        .data_in(context_din),
        .data_out(context_dout)
    );

endmodule
