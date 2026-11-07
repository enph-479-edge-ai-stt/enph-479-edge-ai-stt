`timescale 1ns / 1ps

// Gate buffer: reset, capture of each gate pair across every lane, the
// other pair left alone, no write without capture, and overwrite.
// Reads are combinational, so each check sweeps addr.

module tb_pe_buffer;
    localparam N = 8;
    localparam G = 16;

    reg clk = 0;
    reg rst = 1;
    reg capture = 0;
    reg gate_pair = 0;
    reg [N*G-1:0] pe_in0 = 0;
    reg [N*G-1:0] pe_in1 = 0;
    reg [2:0] addr = 0;
    wire signed [G-1:0] pe_i, pe_f, pe_o, pe_c;

    // What each memory should hold.
    reg [N*G-1:0] exp_i, exp_f, exp_o, exp_c;

    integer errors = 0;
    integer seed = 26;
    integer k;

    pe_buffer #(.GATE_WIDTH(G), .PE_COUNT(N)) dut (
        .clk(clk), .rst(rst), .capture(capture), .gate_pair(gate_pair),
        .pe_in0(pe_in0), .pe_in1(pe_in1), .addr(addr),
        .pe_i(pe_i), .pe_f(pe_f), .pe_o(pe_o), .pe_c(pe_c)
    );

    always #5 clk = ~clk;

    task check_all;
        input [8*48-1:0] msg;
        integer bad;
        begin
            bad = 0;
            for (k = 0; k < N; k = k + 1) begin
                addr = k;
                #1;
                if (pe_i !== exp_i[k*G +: G] || pe_f !== exp_f[k*G +: G]
                    || pe_o !== exp_o[k*G +: G] || pe_c !== exp_c[k*G +: G]) begin
                    if (bad == 0)
                        $display("  lane %0d: i %h f %h o %h c %h", k, pe_i, pe_f, pe_o, pe_c);
                    bad = bad + 1;
                end
            end
            if (bad == 0)
                $display("pass: %0s", msg);
            else begin
                $display("FAIL: %0s (%0d lanes wrong)", msg, bad);
                errors = errors + 1;
            end
        end
    endtask

    task randomize_inputs;
        begin
            for (k = 0; k < N; k = k + 1) begin
                pe_in0[k*G +: G] = $random(seed);
                pe_in1[k*G +: G] = $random(seed);
            end
        end
    endtask

    task capture_pair;
        input p;
        begin
            @(negedge clk);
            randomize_inputs;
            gate_pair = p;
            capture = 1;
            if (!p) begin
                exp_i = pe_in0;
                exp_f = pe_in1;
            end else begin
                exp_o = pe_in0;
                exp_c = pe_in1;
            end
            @(negedge clk);
            capture = 0;
            randomize_inputs;
        end
    endtask

    initial begin
        exp_i = 0; exp_f = 0; exp_o = 0; exp_c = 0;
        repeat (2) @(negedge clk);
        rst = 0;
        check_all("reset clears all four memories");

        capture_pair(0);
        check_all("pair 0 fills i and f, o and c stay 0");

        capture_pair(1);
        check_all("pair 1 fills o and c, i and f kept");

        // Inputs keep changing with capture low.
        repeat (3) begin
            @(negedge clk);
            randomize_inputs;
            gate_pair = $random(seed);
        end
        check_all("no capture, no change");

        capture_pair(0);
        check_all("recapture of pair 0 leaves o and c");

        @(negedge clk);
        rst = 1;
        exp_i = 0; exp_f = 0; exp_o = 0; exp_c = 0;
        @(negedge clk);
        rst = 0;
        check_all("reset after captures");

        if (errors == 0)
            $display("ALL CHECKS PASSED");
        else
            $display("%0d CHECKS FAILED", errors);
        $finish;
    end
endmodule
