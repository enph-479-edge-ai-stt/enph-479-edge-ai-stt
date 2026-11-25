`timescale 1ns / 1ps

// The EPU is combinational: set the inputs, wait, check c_t and h_t
// against a model of the formats in shared/specs/weight_image.md
// ("Activations and cell"), written here on integers, with the tables
// computed from their formulas in real arithmetic (not copied from the
// LUT files):
//
//   c_t = sat16(round(f * c_prev >> 8) + round(i * tanh(g) >> 4))
//   h_t = round(o * tanh(round_sat9(c_t >> 6)) >> 8)
//
// Covers every value of each gate through the EPU, every c_prev with the
// forget gate fully open (which walks tanh(c_t) over the whole cell
// range), both saturation corners of the cell, a few hand-worked values,
// and random inputs. Random gates also show there are no peepholes: the
// model's gates see only their own pre-activation.

module tb_lstm_epu;
    reg signed [15:0] c_prev = 0;
    reg signed [8:0] pe_i = 0, pe_f = 0, pe_o = 0, pe_c = 0;
    wire signed [15:0] c_t;
    wire signed [7:0] h_t;

    integer errors = 0;
    integer seed = 479;
    integer fails, k, exp_c, exp_h;

    lstm_epu dut (
        .clk(1'b0), .rst(1'b0), .enable(1'b1),
        .old_hidden_state(c_prev),
        .pe_input({pe_c, pe_o, pe_f, pe_i}),
        .hidden_state_output(c_t),
        .output_state(h_t)
    );

    function integer clamp;
        input integer v, lo, hi;
        clamp = (v < lo) ? lo : ((v > hi) ? hi : v);
    endfunction

    // Round half up: add half a step, then floor.
    function integer rsh;
        input integer v, k;
        rsh = (v + (1 << (k - 1))) >>> k;
    endfunction

    // Table entries for a value g at 2^-5, g in -256..255.
    function integer sig_of;
        input integer g;
        sig_of = clamp($rtoi($floor(256.0 / (1.0 + $exp(-g / 32.0)) + 0.5)), 0, 255);
    endfunction

    function integer tanh_of;
        input integer g;
        tanh_of = clamp($rtoi($floor(128.0 * $tanh(g / 32.0) + 0.5)), -127, 127);
    endfunction

    task model;
        integer c;
        begin
            c = rsh(sig_of(pe_f) * c_prev, 8) + rsh(sig_of(pe_i) * tanh_of(pe_c), 4);
            exp_c = clamp(c, -32768, 32767);
            exp_h = clamp(rsh(sig_of(pe_o) * tanh_of(clamp(rsh(exp_c, 6), -256, 255)), 8),
                -128, 127);
        end
    endtask

    task check;
        begin
            #1;
            model;
            if (c_t !== exp_c || h_t !== exp_h) begin
                if (fails < 10)
                    $display("  i %0d f %0d o %0d g %0d c_prev %0d: c_t %0d h_t %0d, expected %0d %0d",
                        pe_i, pe_f, pe_o, pe_c, c_prev, c_t, h_t, exp_c, exp_h);
                fails = fails + 1;
            end
        end
    endtask

    task section;
        input [8*56-1:0] msg;
        begin
            errors = errors + fails;
            if (fails == 0)
                $display("pass: %0s", msg);
            else
                $display("FAIL: %0s (%0d mismatches)", msg, fails);
            fails = 0;
        end
    endtask

    task expect_exact;
        input integer c_want, h_want;
        input [8*56-1:0] msg;
        begin
            #1;
            if (c_t !== c_want || h_t !== h_want) begin
                $display("FAIL: %0s: c_t %0d h_t %0d, expected %0d %0d",
                    msg, c_t, h_t, c_want, h_want);
                errors = errors + 1;
            end else
                $display("pass: %0s", msg);
        end
    endtask

    initial begin
        fails = 0;

        // Each gate's every value with the others pinned, so a mistake in
        // one table, one index or one product shows in its own sweep.
        pe_f = -256; pe_o = 255; pe_c = 255; c_prev = 0;
        for (k = -256; k < 256; k = k + 1) begin
            pe_i = k;
            check;
        end
        section("input gate, all 512 values");

        pe_i = 255; pe_f = -256; pe_o = 255;
        for (k = -256; k < 256; k = k + 1) begin
            pe_c = k;
            check;
        end
        section("candidate gate, all 512 values");

        pe_i = -256; pe_c = 0; pe_o = 255; c_prev = 16'sd20000;
        for (k = -256; k < 256; k = k + 1) begin
            pe_f = k;
            check;
        end
        section("forget gate, all 512 values");

        pe_i = 255; pe_f = 0; pe_c = 128; c_prev = 16'sd3000;
        for (k = -256; k < 256; k = k + 1) begin
            pe_o = k;
            check;
        end
        section("output gate, all 512 values");

        // f = 255, i = 0: c_t = round(255 * c_prev / 256), so this walks
        // the cell slice and tanh(c_t) over the whole cell range.
        pe_i = -256; pe_f = 255; pe_o = 255; pe_c = 0;
        for (k = -32768; k < 32768; k = k + 1) begin
            c_prev = k;
            check;
        end
        section("every c_prev through the open forget gate");

        for (k = 0; k < 50000; k = k + 1) begin
            pe_i = $random(seed);
            pe_f = $random(seed);
            pe_o = $random(seed);
            pe_c = $random(seed);
            c_prev = $random(seed);
            check;
        end
        section("50000 random inputs");

        // Hand-worked values, independent of the model above.
        // Zero gates: f = i = o = 128, tanh(0) = 0. c_t = round(128 * 544 / 256)
        // = 272 (0.1328); index round(272 / 64) = 4 (0.125), tanh 16;
        // h = round(128 * 16 / 256) = 8.
        pe_i = 0; pe_f = 0; pe_o = 0; pe_c = 0; c_prev = 16'sd544;
        expect_exact(272, 8, "zero gates halve the cell");

        // Saturation: f = i = 255, tanh(g) = 127, c_prev at the top:
        // 32640 + 2024 = 34664 clips to 32767. tanh(16) = 127, h = 127.
        pe_i = 255; pe_f = 255; pe_o = 255; pe_c = 255; c_prev = 16'sd32767;
        expect_exact(32767, 127, "cell saturates at +32767");
        // And at the bottom: -32640 - 2024 clips to -32768, h = -127.
        pe_c = -256; c_prev = -16'sd32768;
        expect_exact(-32768, -127, "cell saturates at -32768");

        // Closed output gate: sigmoid(-8) = 0, so h = 0 whatever c is.
        pe_o = -256; c_prev = 16'sd32767; pe_c = 255;
        expect_exact(32767, 0, "closed output gate gives h = 0");

        // Closed forget gate drops the cell: f = 0, i = 0.
        pe_i = -256; pe_f = -256; pe_o = 0; c_prev = 16'sd1000;
        expect_exact(0, 0, "closed forget and input gates clear the cell");

        if (errors == 0)
            $display("ALL CHECKS PASSED");
        else
            $display("%0d CHECKS FAILED", errors);
        $finish;
    end
endmodule
