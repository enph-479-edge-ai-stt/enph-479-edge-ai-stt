`timescale 1ns / 1ps

// The EPU is combinational: set the inputs, wait, check c_t and h_t.
// Covers the LUT index mapping (every entry of both tables), unsigned
// sigmoid gates, the cell update, and that the gates see only their own
// pre-activation (no peepholes).
//
// Bit slices are placeholders until shared/specs fixes the cell and
// table formats: the gate index is pre-activation bits [7:4], tanh(c)
// indexes c_t[11:8], and h_t is the low byte of o * tanh(c). Checks that
// depend on those slices will change with them.

module tb_lstm_epu;
    reg signed [15:0] c_prev = 0;
    reg signed [15:0] pe_i = 0, pe_f = 0, pe_o = 0, pe_c = 0;
    wire signed [15:0] c_t;
    wire signed [7:0] h_t;

    // Gate values that land on the ends and middle of the tables.
    localparam signed [15:0] OPEN = 16'sh0070;    // index 15: sigmoid 255, tanh 127
    localparam signed [15:0] CLOSED = -16'sd128;  // index 0:  sigmoid 0,   tanh -127

    reg [7:0] sig_table [0:15];
    reg signed [7:0] tanh_table [0:15];

    integer errors = 0;
    integer sweep_errors;
    integer s;

    lstm_epu dut (
        .clk(1'b0), .rst(1'b0), .enable(1'b1),
        .old_hidden_state(c_prev),
        .pe_input({pe_c, pe_o, pe_f, pe_i}),
        .hidden_state_output(c_t),
        .output_state(h_t)
    );

    task check_c;
        input signed [15:0] expected;
        input [8*48-1:0] msg;
        begin
            #1;
            if (c_t !== expected) begin
                $display("FAIL: %0s: c_t %0d, expected %0d", msg, c_t, expected);
                errors = errors + 1;
            end else
                $display("pass: %0s", msg);
        end
    endtask

    task check_h;
        input signed [7:0] expected;
        input [8*48-1:0] msg;
        begin
            #1;
            if (h_t !== expected) begin
                $display("FAIL: %0s: h_t %0d, expected %0d", msg, h_t, expected);
                errors = errors + 1;
            end else
                $display("pass: %0s", msg);
        end
    endtask

    task set_gates;
        input signed [15:0] i, f, o, c;
        begin
            pe_i = i;
            pe_f = f;
            pe_o = o;
            pe_c = c;
        end
    endtask

    initial begin
        sig_table[0]  = 0;   sig_table[1]  = 0;   sig_table[2]  = 1;   sig_table[3]  = 2;
        sig_table[4]  = 5;   sig_table[5]  = 12;  sig_table[6]  = 30;  sig_table[7]  = 69;
        sig_table[8]  = 128; sig_table[9]  = 186; sig_table[10] = 225; sig_table[11] = 243;
        sig_table[12] = 250; sig_table[13] = 253; sig_table[14] = 254; sig_table[15] = 255;
        tanh_table[0]  = -127; tanh_table[1]  = -127; tanh_table[2]  = -127; tanh_table[3]  = -127;
        tanh_table[4]  = -127; tanh_table[5]  = -126; tanh_table[6]  = -122; tanh_table[7]  = -97;
        tanh_table[8]  = 0;    tanh_table[9]  = 97;   tanh_table[10] = 122;  tanh_table[11] = 126;
        tanh_table[12] = 127;  tanh_table[13] = 127;  tanh_table[14] = 127;  tanh_table[15] = 127;

        // Zero pre-activation is the middle of the table: f = 0.5, tanh = 0.
        c_prev = 16'sd544;
        set_gates(0, 0, 0, 0);
        check_c(16'sd272, "zero gates: f = 0.5 halves c_prev");

        // Every sigmoid entry: f closed, tanh(c) = 127, so c_t = i * 127.
        c_prev = 0;
        sweep_errors = errors;
        for (s = -8; s < 8; s = s + 1) begin
            set_gates(s * 16, CLOSED, 0, OPEN);
            #1;
            if (c_t !== sig_table[s + 8] * 127) begin
                $display("FAIL: sigmoid slice %0d: c_t %0d, expected %0d",
                    s, c_t, sig_table[s + 8] * 127);
                errors = errors + 1;
            end
        end
        if (errors == sweep_errors)
            $display("pass: sigmoid index mapping, all 16 entries");

        // Every tanh entry: i = 255, f closed, so c_t = 255 * tanh(c).
        sweep_errors = errors;
        for (s = -8; s < 8; s = s + 1) begin
            set_gates(OPEN, CLOSED, 0, s * 16);
            #1;
            if (c_t !== 255 * tanh_table[s + 8]) begin
                $display("FAIL: tanh slice %0d: c_t %0d, expected %0d",
                    s, c_t, 255 * tanh_table[s + 8]);
                errors = errors + 1;
            end
        end
        if (errors == sweep_errors)
            $display("pass: tanh index mapping, all 16 entries");

        // f = 255 must keep c_prev's sign (a signed sigmoid would read -1).
        set_gates(CLOSED, OPEN, 0, 0);
        c_prev = 16'sd1000;
        check_c(16'sd996, "open forget gate keeps +c_prev (255/256)");
        c_prev = -16'sd1000;
        check_c(-16'sd997, "open forget gate keeps -c_prev (floor)");

        set_gates(0, CLOSED, 0, 0);
        c_prev = 16'sd1000;
        check_c(16'sd0, "closed forget gate drops c_prev");

        // Both terms together: 128 * 400 >> 8 + 255 * -127.
        set_gates(OPEN, 0, 0, CLOSED);
        c_prev = 16'sd400;
        check_c(16'sd200 - 16'sd32385, "c_t = f * c_prev + i * tanh(c)");

        // h: closed output gate gives 0 whatever the cell holds.
        set_gates(OPEN, CLOSED, CLOSED, OPEN);
        c_prev = 0;
        check_h(8'sd0, "closed output gate gives h = 0");

        // c_t = 255 * 768 >> 8 = 765, c_t[11:8] = 2 -> tanh 122,
        // o = 255: 255 * 122 = 0x7986, low byte 0x86.
        set_gates(CLOSED, OPEN, OPEN, 0);
        c_prev = 16'sd768;
        check_c(16'sd765, "cell for the h check");
        check_h(8'sh86, "h = low byte of o * tanh(c_t)");

        // No peepholes: the cell must not leak into any gate. With all
        // gates 0, c_t = 272 and c_t[11:8] = 1 -> tanh 97. o stays
        // sigmoid(0) = 128, so h = low byte of 128 * 97 = 0x3080. A
        // c_prev * w or c_t * w term would move i, f or o off 128.
        c_prev = 16'sd544;
        set_gates(0, 0, 0, 0);
        check_c(16'sd272, "i and f ignore c_prev");
        check_h(8'sh80, "o ignores c_t");
        c_prev = -16'sd30000;
        check_c(-16'sd15000, "i and f ignore a large negative c_prev");

        if (errors == 0)
            $display("ALL CHECKS PASSED");
        else
            $display("%0d CHECKS FAILED", errors);
        $finish;
    end
endmodule
