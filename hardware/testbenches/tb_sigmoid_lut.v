`timescale 1ns / 1ps

// Every entry of the 512-entry table against the formula in sigmoid_lut.v,
// computed here in real arithmetic: index n is v = (n - 256) / 32 and holds
// clamp(round(256 / (1 + exp(-v))), 0, 255). Then the shape: monotonic,
// 128 at index 256, 0 and 255 at the two ends.

module tb_sigmoid_lut;
    reg [8:0] index = 0;
    wire [7:0] sigmoid_output;

    integer errors = 0;
    integer fails, n, expected, prev;

    sigmoid_lut dut (.index(index), .sigmoid_output(sigmoid_output));

    initial begin
        fails = 0;
        for (n = 0; n < 512; n = n + 1) begin
            expected = $rtoi($floor(256.0 / (1.0 + $exp(-(n - 256) / 32.0)) + 0.5));
            if (expected > 255)
                expected = 255;
            index = n;
            #1;
            if (sigmoid_output !== expected) begin
                $display("FAIL: index %0d: got %0d, expected %0d", n, sigmoid_output, expected);
                fails = fails + 1;
            end
        end
        errors = errors + fails;
        if (fails == 0)
            $display("pass: all 512 entries match the formula");

        fails = 0;
        index = 0;
        #1;
        prev = sigmoid_output;
        for (n = 1; n < 512; n = n + 1) begin
            index = n;
            #1;
            if (sigmoid_output < prev) begin
                $display("FAIL: not monotonic at index %0d", n);
                fails = fails + 1;
            end
            prev = sigmoid_output;
        end
        errors = errors + fails;
        if (fails == 0)
            $display("pass: monotonic non-decreasing");

        index = 256; #1;
        if (sigmoid_output !== 8'd128) begin
            $display("FAIL: sigmoid(0) = %0d, expected 128", sigmoid_output);
            errors = errors + 1;
        end
        index = 0; #1;
        if (sigmoid_output !== 8'd0) begin
            $display("FAIL: sigmoid(-8) = %0d, expected 0", sigmoid_output);
            errors = errors + 1;
        end
        index = 511; #1;
        if (sigmoid_output !== 8'd255) begin
            $display("FAIL: sigmoid(8 - 1/32) = %0d, expected 255", sigmoid_output);
            errors = errors + 1;
        end

        if (errors == 0) $display("RESULT: all sigmoid_lut tests passed");
        else             $display("RESULT: %0d failures", errors);
        $finish;
    end
endmodule
