`timescale 1ns / 1ps

// Every entry of the 512-entry table against the formula in tanh_lut.v,
// computed here in real arithmetic: index n is v = (n - 256) / 32 and holds
// clamp(round(128 * tanh(v)), -127, 127). Then the shape: monotonic, 0 at
// index 256, odd about it (entry 256 + k is minus entry 256 - k), and
// -127 / 127 at the two ends.

module tb_tanh_lut;
    reg [8:0] index = 0;
    wire signed [7:0] tanh_output;

    integer errors = 0;
    integer fails, n, expected, prev, mirror;

    tanh_lut dut (.index(index), .tanh_output(tanh_output));

    initial begin
        fails = 0;
        for (n = 0; n < 512; n = n + 1) begin
            expected = $rtoi($floor(128.0 * $tanh((n - 256) / 32.0) + 0.5));
            if (expected > 127)
                expected = 127;
            if (expected < -127)
                expected = -127;
            index = n;
            #1;
            if (tanh_output !== expected) begin
                $display("FAIL: index %0d: got %0d, expected %0d", n, tanh_output, expected);
                fails = fails + 1;
            end
        end
        errors = errors + fails;
        if (fails == 0)
            $display("pass: all 512 entries match the formula");

        fails = 0;
        index = 0;
        #1;
        prev = tanh_output;
        for (n = 1; n < 512; n = n + 1) begin
            index = n;
            #1;
            if (tanh_output < prev) begin
                $display("FAIL: not monotonic at index %0d", n);
                fails = fails + 1;
            end
            prev = tanh_output;
        end
        errors = errors + fails;
        if (fails == 0)
            $display("pass: monotonic non-decreasing");

        fails = 0;
        for (n = 1; n < 256; n = n + 1) begin
            index = 256 - n;
            #1;
            mirror = tanh_output;
            index = 256 + n;
            #1;
            if (tanh_output !== -mirror) begin
                $display("FAIL: not odd at +-%0d: %0d vs %0d", n, tanh_output, mirror);
                fails = fails + 1;
            end
        end
        errors = errors + fails;
        if (fails == 0)
            $display("pass: odd about index 256");

        index = 256; #1;
        if (tanh_output !== 8'sd0) begin
            $display("FAIL: tanh(0) = %0d, expected 0", tanh_output);
            errors = errors + 1;
        end
        index = 0; #1;
        if (tanh_output !== -8'sd127) begin
            $display("FAIL: tanh(-8) = %0d, expected -127", tanh_output);
            errors = errors + 1;
        end
        index = 511; #1;
        if (tanh_output !== 8'sd127) begin
            $display("FAIL: tanh(8 - 1/32) = %0d, expected 127", tanh_output);
            errors = errors + 1;
        end

        if (errors == 0) $display("RESULT: all tanh_lut tests passed");
        else             $display("RESULT: %0d failures", errors);
        $finish;
    end
endmodule
