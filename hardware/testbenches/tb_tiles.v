`timescale 1ns / 1ps

// Smoke test for the gate buffer, the EPU peephole order, a tiny LSTM
// tile, and a tiny output projection. Run from hardware/rtl:
//   iverilog -g2012 -o /tmp/tb_tiles.vvp \
//     ../testbenches/tb_tiles.v pe_buffer.v pe_unit.v pe_array.v \
//     lstm_epu.v sigmoid_lut.v tanh_lut.v weight_bram.v \
//     context_memory.v lstm_tile.v output_tile.v
//   vvp /tmp/tb_tiles.vvp

module tb_tiles;
    reg clk;
    reg rst;
    integer errors;

    reg capture;
    reg gate_pair;
    reg signed [4*16-1:0] pe_in0;
    reg signed [4*16-1:0] pe_in1;
    reg [1:0] addr;
    wire signed [15:0] pe_i, pe_f, pe_o, pe_c;

    pe_buffer #(.GATE_WIDTH(16), .PE_COUNT(4), .ADDR_WIDTH(2)) buf_dut (
        .clk(clk), .rst(rst), .capture(capture), .gate_pair(gate_pair),
        .pe_in0(pe_in0), .pe_in1(pe_in1), .addr(addr),
        .pe_i(pe_i), .pe_f(pe_f), .pe_o(pe_o), .pe_c(pe_c)
    );

    // Output peephole must use c_t, so pe_c = 0 still changes h when w_co != 0.
    reg signed [15:0] c_prev;
    reg signed [63:0] pe_input;
    reg signed [23:0] peep;
    wire signed [15:0] c_t;
    wire signed [7:0] h_t;

    lstm_epu epu_dut (
        .clk(clk), .rst(rst), .enable(1'b1),
        .old_hidden_state(c_prev),
        .pe_input(pe_input),
        .peephole_weight(peep),
        .hidden_state_output(c_t),
        .output_state(h_t)
    );

    reg pe_enable, load_bias, sel_in, capture_gates, tile_pair;
    reg context_we, context_use_ext, bias_we, peep_we, w0_we, w1_we;
    reg signed [7:0] d_in;
    reg [2:0] weight_addr;
    reg [11:0] w0_din, w1_din;
    reg [1:0] bias_addr;
    reg signed [2*16-1:0] bias0_din, bias1_din;
    reg [0:0] neuron_addr;
    reg [1:0] peep_addr;
    reg signed [23:0] peep_din;
    reg [2:0] context_addr;
    reg [23:0] context_ext_din;
    wire signed [15:0] tile_c;
    wire signed [7:0] tile_h;
    wire [23:0] context_dout;
    wire signed [15:0] t_i, t_f, t_o, t_c;

    lstm_tile #(
        .PE_COUNT(2),
        .WEIGHT_DEPTH(8),
        .CONTEXT_DEPTH(8),
        .N_LSTM_LAYERS(2),
        .ACC_LSB(0)
    ) tile (
        .clk(clk), .rst(rst),
        .pe_enable(pe_enable), .load_bias(load_bias), .sel_in(sel_in), .d_in(d_in),
        .weight_addr(weight_addr), .weight0_we(w0_we), .weight1_we(w1_we),
        .weight0_din(w0_din), .weight1_din(w1_din),
        .bias_addr(bias_addr), .bias_we(bias_we), .bias0_din(bias0_din), .bias1_din(bias1_din),
        .capture_gates(capture_gates), .gate_pair(tile_pair), .neuron_addr(neuron_addr),
        .peep_addr(peep_addr), .peep_we(peep_we), .peep_din(peep_din),
        .context_addr(context_addr), .context_we(context_we),
        .context_use_ext(context_use_ext), .context_ext_din(context_ext_din),
        .c_t(tile_c), .h_t(tile_h), .context_dout(context_dout),
        .pe_i(t_i), .pe_f(t_f), .pe_o(t_o), .pe_c(t_c)
    );

    reg out_en, out_load, out_wwe, out_bwe;
    reg signed [7:0] h_element;
    reg out_waddr;
    reg [2*6-1:0] out_wdin;
    reg [0:0] out_baddr;
    reg signed [2*16-1:0] out_bdin;
    wire signed [2*24-1:0] acc;
    wire signed [2*8-1:0] scores;

    output_tile #(
        .HIDDEN(2), .VOCAB(2), .N_JOBS(1)
    ) proj (
        .clk(clk), .rst(rst), .enable(out_en), .load_bias(out_load),
        .h_element(h_element),
        .weight_addr(out_waddr), .weight_we(out_wwe), .weight_din(out_wdin),
        .bias_addr(out_baddr), .bias_we(out_bwe), .bias_din(out_bdin),
        .acc(acc), .scores(scores)
    );

    always #5 clk = ~clk;

    task check;
        input integer cond;
        input [255:0] msg;
        begin
            if (!cond) begin
                $display("FAIL: %0s", msg);
                errors = errors + 1;
            end
        end
    endtask

    initial begin
        errors = 0;
        clk = 0;
        rst = 1;
        capture = 0;
        gate_pair = 0;
        pe_in0 = 0;
        pe_in1 = 0;
        addr = 0;
        c_prev = 0;
        pe_input = 0;
        peep = 0;
        pe_enable = 0;
        load_bias = 0;
        sel_in = 0;
        capture_gates = 0;
        tile_pair = 0;
        context_we = 0;
        context_use_ext = 0;
        bias_we = 0;
        peep_we = 0;
        w0_we = 0;
        w1_we = 0;
        d_in = 0;
        weight_addr = 0;
        w0_din = 0;
        w1_din = 0;
        bias_addr = 0;
        bias0_din = 0;
        bias1_din = 0;
        neuron_addr = 0;
        peep_addr = 0;
        peep_din = 0;
        context_addr = 0;
        context_ext_din = 0;
        out_en = 0;
        out_load = 0;
        out_wwe = 0;
        out_bwe = 0;
        h_element = 0;
        out_waddr = 0;
        out_wdin = 0;
        out_baddr = 0;
        out_bdin = 0;

        repeat (2) @(posedge clk);
        rst = 0;
        @(posedge clk);

        // Capture i/f, then o/c. Both pairs must still be readable.
        pe_in0 = {16'h0003, 16'h0002, 16'h0001, 16'h0000};
        pe_in1 = {16'h0013, 16'h0012, 16'h0011, 16'h0010};
        gate_pair = 0;
        capture = 1;
        @(posedge clk);
        capture = 0;
        pe_in0 = {16'h0023, 16'h0022, 16'h0021, 16'h0020};
        pe_in1 = {16'h0033, 16'h0032, 16'h0031, 16'h0030};
        gate_pair = 1;
        capture = 1;
        @(posedge clk);
        capture = 0;
        @(posedge clk);

        addr = 0;
        #1;
        check(pe_i === 16'h0000 && pe_f === 16'h0010, "lane0 i/f");
        check(pe_o === 16'h0020 && pe_c === 16'h0030, "lane0 o/c");
        addr = 3;
        #1;
        check(pe_i === 16'h0003 && pe_f === 16'h0013, "lane3 i/f kept after o/c capture");
        check(pe_o === 16'h0023 && pe_c === 16'h0033, "lane3 o/c");

        // pe_c = 0, pe_f = 0x80 -> sigmoid index 8, c_prev = 256, w_co = 1.
        // c_t comes from f * c_prev. o must follow c_t, so h must be nonzero.
        c_prev = 16'sd256;
        peep = 24'h000001;
        pe_input = {16'h0000, 16'h0000, 16'h0080, 16'h0000}; // {c, o, f, i}
        #1;
        check(c_t !== 16'sd0, "cell updated from forget gate");
        check(h_t !== 8'sd0, "output gate follows c_t when pe_c is 0");
        peep = 24'h000000;
        #1;
        check(h_t === 8'sd0, "clearing w_co drops the output-gate peephole");

        // Tile: one MAC of x=3 against a loaded bias and weight row.
        bias_addr = 0;
        bias0_din = {16'sd9, 16'sd7};
        bias1_din = {16'sd11, 16'sd5};
        bias_we = 1;
        @(posedge clk);
        bias_we = 0;
        @(posedge clk);

        weight_addr = 3'd1;
        w0_din = {6'sd4, 6'sd2};
        w1_din = {6'sd1, 6'sd3};
        w0_we = 1;
        w1_we = 1;
        @(posedge clk);
        w0_we = 0;
        w1_we = 0;
        @(posedge clk);

        load_bias = 1;
        pe_enable = 0;
        @(posedge clk);
        load_bias = 0;
        d_in = 8'sd3;
        sel_in = 0;
        pe_enable = 1;
        @(posedge clk);
        pe_enable = 0;
        tile_pair = 0;
        capture_gates = 1;
        @(posedge clk);
        capture_gates = 0;

        neuron_addr = 0;
        #1;
        // bias 7 + 3*2 = 13, bias 5 + 3*3 = 14
        check(t_i === 16'sd13, "neuron0 gate i mac");
        check(t_f === 16'sd14, "neuron0 gate f mac");
        neuron_addr = 1;
        #1;
        // bias 9 + 3*4 = 21, bias 11 + 3*1 = 14
        check(t_i === 16'sd21, "neuron1 gate i mac");
        check(t_f === 16'sd14, "neuron1 gate f mac");

        context_use_ext = 1;
        context_ext_din = {16'h00A5, 8'h3C};
        context_addr = 3'd2;
        context_we = 1;
        @(posedge clk);
        context_we = 0;
        context_use_ext = 0;
        @(posedge clk);
        #1;
        check(context_dout === {16'h00A5, 8'h3C}, "context ext write");

        // Output tile: bias 10, then h=4 * w=3 -> 22 for symbol 0.
        out_baddr = 0;
        out_bdin = {16'sd8, 16'sd10};
        out_bwe = 1;
        @(posedge clk);
        out_bwe = 0;
        out_waddr = 1'b1;
        out_wdin = {6'sd5, 6'sd3};
        out_wwe = 1;
        @(posedge clk);
        out_wwe = 0;
        @(posedge clk);
        out_load = 1;
        @(posedge clk);
        out_load = 0;
        h_element = 8'sd4;
        out_en = 1;
        @(posedge clk);
        out_en = 0;
        #1;
        check(acc[23:0] === 24'sd22, "proj symbol0 10+4*3");
        check(acc[47:24] === 24'sd28, "proj symbol1 8+4*5");
        check(scores[7:0] === 8'sd22, "score low byte");

        if (errors == 0)
            $display("ALL CHECKS PASSED");
        else
            $display("%0d CHECKS FAILED", errors);
        $finish;
    end
endmodule
