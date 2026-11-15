`timescale 1ns / 1ps

// round_sat in the three shapes the RTL uses, against
// clamp(floor((x + 2^(s-1)) / 2^s), min, max) computed on integers here:
//   - the cell into the tanh table: 16 -> 9 bits, shift 6, every input
//   - a gate: 24 -> 9 bits, shift 9 (AM) or 6 (LM)
//   - a score: 24 -> 8 bits, shift 11 (AM) or 8 (LM)
// The 24-bit shapes take the values on and next to every rounding step
// out to past both saturation edges, the two ends of the range, and
// random values of every magnitude.

module tb_round_sat;
    reg lm = 0;
    reg signed [15:0] c_in = 0;
    reg signed [23:0] acc = 0;
    wire signed [8:0] c_out;
    wire signed [8:0] gate_out;
    wire signed [7:0] score_out;

    integer errors = 0;
    integer seed = 512;
    integer fails, x, k, s, m;

    round_sat #(.IN_WIDTH(16), .OUT_WIDTH(9), .SHIFT_AM(6), .SHIFT_LM(6))
        cell_slice (.lm(1'b0), .din(c_in), .dout(c_out));
    round_sat #(.IN_WIDTH(24), .OUT_WIDTH(9), .SHIFT_AM(9), .SHIFT_LM(6))
        gate_slice (.lm(lm), .din(acc), .dout(gate_out));
    round_sat #(.IN_WIDTH(24), .OUT_WIDTH(8), .SHIFT_AM(11), .SHIFT_LM(8))
        score_slice (.lm(lm), .din(acc), .dout(score_out));

    function integer want;
        input integer v, shift, bits;
        integer r;
        begin
            r = (v + (1 << (shift - 1))) >>> shift;
            if (r > (1 << (bits - 1)) - 1)
                r = (1 << (bits - 1)) - 1;
            if (r < -(1 << (bits - 1)))
                r = -(1 << (bits - 1));
            want = r;
        end
    endfunction

    task check_acc;
        input integer v;
        integer g, sc;
        begin
            acc = v;
            #1;
            g = want($signed(acc), lm ? 6 : 9, 9);
            sc = want($signed(acc), lm ? 8 : 11, 8);
            if (gate_out !== g || score_out !== sc) begin
                if (fails < 10)
                    $display("  lm %0d acc %0d: gate %0d expected %0d, score %0d expected %0d",
                        lm, $signed(acc), gate_out, g, score_out, sc);
                fails = fails + 1;
            end
        end
    endtask

    initial begin
        fails = 0;
        for (x = -32768; x < 32768; x = x + 1) begin
            c_in = x;
            #1;
            if (c_out !== want(x, 6, 9)) begin
                if (fails < 10)
                    $display("  c %0d: got %0d expected %0d", x, c_out, want(x, 6, 9));
                fails = fails + 1;
            end
        end
        errors = errors + fails;
        if (fails == 0)
            $display("pass: cell slice, all 65536 inputs");

        for (m = 0; m < 2; m = m + 1) begin
            lm = m;
            fails = 0;
            // Each shift the slices use, out past +-256 steps (beyond
            // both saturation edges): the step, its midpoint, and one
            // below the midpoint.
            for (s = 6; s <= 11; s = s + 1)
                for (k = -300; k <= 300; k = k + 1) begin
                    check_acc(k * (1 << s) + (1 << (s - 1)) - 1);
                    check_acc(k * (1 << s) + (1 << (s - 1)));
                    check_acc(k * (1 << s));
                end
            check_acc(-(1 << 23));
            check_acc((1 << 23) - 1);
            for (k = 0; k < 20000; k = k + 1)
                check_acc($random(seed) % (1 << (k % 23 + 1)));
            errors = errors + fails;
            if (fails == 0)
                $display("pass: gate and score slices, lm = %0d", m);
        end

        if (errors == 0)
            $display("ALL CHECKS PASSED");
        else
            $display("%0d CHECKS FAILED", errors);
        $finish;
    end
endmodule
