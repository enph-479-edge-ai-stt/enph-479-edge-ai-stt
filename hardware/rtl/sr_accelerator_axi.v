`timescale 1ns / 1ps

//=====================================================================
// Module: sr_accelerator_axi.v
// Description:
//  AXI wrapper around sr_accelerator_top, so the Zynq's ARM cores can
//  drive it through an AXI DMA:
//    s_axil  AXI-Lite slave, the control registers
//    s_axis  AXI-Stream slave, weights and frames in (DMA MM2S)
//    m_axis  AXI-Stream master, scores out (DMA S2MM)
//  The register map and stream formats are
//  shared/specs/fabric_interface.md.
//
//  Loading. LOAD_TARGET picks the memory the input stream fills and
//  LOAD_ADDR its first row; writing either restarts the loader. A row
//  arrives as 32-bit words, least significant word first. When its last
//  word is in, the row is written and the address steps on. The stream
//  stalls (tready low) for that write cycle and while a job runs, since
//  the accelerator only takes host writes while idle. The frame target
//  takes 4 features per word and writes them one byte per cycle;
//  bytes past AM_IN are dropped.
//
//  Running. JOB, BEAM and LM_CHAR describe the job; writing 1 to CTRL
//  starts it, and they are captured then. A start that arrives while a
//  row is being written waits for the write to finish. STATUS.done
//  stays set until the next start. An AM or LM job then sends its VOCAB
//  int8 scores on m_axis, 4 per word with symbol 0 in the low byte of
//  the first word, tlast on the last word. A start is ignored while a
//  job is running or while those words still wait to be taken.
//=====================================================================

module sr_accelerator_axi #(
    parameter PE_COUNT = 256,
    parameter AM_LAYERS = 3,
    parameter LM_LAYERS = 2,
    parameter AM_IN = 123,
    parameter LM_IN = 30,
    parameter VOCAB = 30,
    parameter N_BEAMS = 128,
    parameter WEIGHT_DEPTH =
        ((AM_IN + PE_COUNT) * 2)
        + ((AM_LAYERS > 1) ? (AM_LAYERS - 1) : 0) * (PE_COUNT * 4)
        + ((LM_IN + PE_COUNT) * 2)
        + ((LM_LAYERS > 1) ? (LM_LAYERS - 1) : 0) * (PE_COUNT * 4),
    parameter CONTEXT_DEPTH = (AM_LAYERS * PE_COUNT) + (N_BEAMS * LM_LAYERS * PE_COUNT)
) (
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axil:s_axis:m_axis, ASSOCIATED_RESET rst_n" *)
    input wire clk,
    input wire rst_n,

    input wire [5:0] s_axil_awaddr,
    input wire s_axil_awvalid,
    output wire s_axil_awready,
    input wire [31:0] s_axil_wdata,
    input wire [3:0] s_axil_wstrb,
    input wire s_axil_wvalid,
    output wire s_axil_wready,
    output wire [1:0] s_axil_bresp,
    output reg s_axil_bvalid,
    input wire s_axil_bready,
    input wire [5:0] s_axil_araddr,
    input wire s_axil_arvalid,
    output wire s_axil_arready,
    output reg [31:0] s_axil_rdata,
    output wire [1:0] s_axil_rresp,
    output reg s_axil_rvalid,
    input wire s_axil_rready,

    input wire [31:0] s_axis_tdata,
    input wire s_axis_tvalid,
    output wire s_axis_tready,

    output wire [31:0] m_axis_tdata,
    output wire [3:0] m_axis_tkeep,
    output wire m_axis_tvalid,
    input wire m_axis_tready,
    output wire m_axis_tlast
);

    // Widths the accelerator is built with (sr_accelerator_top defaults).
    localparam W_W = 6;
    localparam B_W = 24;
    localparam S_W = 8;

    localparam WEIGHT_AW = $clog2(WEIGHT_DEPTH);
    localparam CONTEXT_AW = $clog2(CONTEXT_DEPTH);
    localparam BIAS_AW = $clog2((AM_LAYERS + LM_LAYERS) * 2);
    localparam FRAME_AW = (AM_IN <= 1) ? 1 : $clog2(AM_IN);
    localparam BEAM_W = (N_BEAMS <= 1) ? 1 : $clog2(N_BEAMS);
    localparam CHAR_W = (VOCAB <= 1) ? 1 : $clog2(VOCAB);
    localparam OUT_AW = $clog2(PE_COUNT * 2);

    // Registers, by word address (byte address / 4).
    localparam R_CTRL = 4'd0;
    localparam R_JOB = 4'd1;
    localparam R_BEAM = 4'd2;
    localparam R_CHAR = 4'd3;
    localparam R_STATUS = 4'd4;
    localparam R_LOAD_TARGET = 4'd5;
    localparam R_LOAD_ADDR = 4'd6;
    localparam R_LOAD_COUNT = 4'd7;
    localparam R_CTX_ADDR = 4'd8;
    localparam R_CTX_DATA = 4'd9;

    localparam T_FRAME = 3'd0;
    localparam T_WEIGHT = 3'd1;
    localparam T_BIAS = 3'd2;
    localparam T_OUT_WEIGHT = 3'd3;
    localparam T_OUT_BIAS = 3'd4;

    // A row of each target, in bits and in 32-bit words.
    localparam WT_BITS = 2 * PE_COUNT * W_W;
    localparam BS_BITS = 2 * PE_COUNT * B_W;
    localparam OW_BITS = VOCAB * W_W;
    localparam OB_BITS = VOCAB * B_W;
    localparam WT_WORDS = (WT_BITS + 31) / 32;
    localparam BS_WORDS = (BS_BITS + 31) / 32;
    localparam OW_WORDS = (OW_BITS + 31) / 32;
    localparam OB_WORDS = (OB_BITS + 31) / 32;
    localparam MAX_A = (WT_WORDS > BS_WORDS) ? WT_WORDS : BS_WORDS;
    localparam MAX_B = (OW_WORDS > OB_WORDS) ? OW_WORDS : OB_WORDS;
    localparam MAX_C = (MAX_A > MAX_B) ? MAX_A : MAX_B;
    localparam MAX_WORDS = (MAX_C > 2) ? MAX_C : 2;
    localparam BUF_BITS = MAX_WORDS * 32;
    localparam WC_W = $clog2(MAX_WORDS + 1);
    localparam OUT_WORDS = (VOCAB * S_W + 31) / 32;
    localparam OI_W = (OUT_WORDS <= 1) ? 1 : $clog2(OUT_WORDS);

    // Control registers.
    reg [1:0] job_r;
    reg [BEAM_W-1:0] beam_r;
    reg [CHAR_W-1:0] char_r;
    reg [CONTEXT_AW-1:0] ctx_addr_r;

    // The running job, captured at start. start_req holds a start until
    // the loader's row write is done; start_r is what the accelerator sees.
    reg start_req;
    reg start_r;
    reg [1:0] run_job;
    reg [BEAM_W-1:0] run_beam;
    reg [CHAR_W-1:0] run_char;
    reg job_done;
    reg job_error;

    // Loader.
    reg [2:0] target;
    reg [31:0] load_addr;
    reg [31:0] row_addr;
    reg [31:0] load_count;
    reg [WC_W-1:0] word_cnt;
    reg [BUF_BITS-1:0] row_buf;
    reg writing;
    reg [1:0] byte_k;

    // Score output.
    reg [OUT_WORDS*32-1:0] out_buf;
    reg [OI_W-1:0] out_idx;
    reg out_pending;

    wire accel_busy;
    wire accel_done;
    wire accel_error;
    wire signed [VOCAB*S_W-1:0] scores;
    wire [23:0] context_dout;

    wire [WC_W-1:0] words_per_row =
        (target == T_WEIGHT) ? WT_WORDS[WC_W-1:0] :
        (target == T_BIAS) ? BS_WORDS[WC_W-1:0] :
        (target == T_OUT_WEIGHT) ? OW_WORDS[WC_W-1:0] :
        (target == T_OUT_BIAS) ? OB_WORDS[WC_W-1:0] :
        {{(WC_W-1){1'b0}}, 1'b1};
    wire target_ok = (target <= T_OUT_BIAS);

    // ---------------- AXI-Lite ----------------
    // One write or read in flight; both address and data must be valid.
    wire wr_fire = s_axil_awvalid && s_axil_wvalid && !s_axil_bvalid;
    wire rd_fire = s_axil_arvalid && !s_axil_rvalid;
    assign s_axil_awready = wr_fire;
    assign s_axil_wready = wr_fire;
    assign s_axil_bresp = 2'b00;
    assign s_axil_arready = rd_fire;
    assign s_axil_rresp = 2'b00;
    wire [3:0] wr_reg = s_axil_awaddr[5:2];
    wire [3:0] rd_reg = s_axil_araddr[5:2];

    // ---------------- streams ----------------
    wire in_fire = s_axis_tvalid && s_axis_tready;
    assign s_axis_tready = target_ok && !writing && !start_req && !start_r && !accel_busy;

    assign m_axis_tdata = out_buf[out_idx*32 +: 32];
    assign m_axis_tkeep = 4'hF;
    assign m_axis_tvalid = out_pending;
    assign m_axis_tlast = (out_idx == OUT_WORDS - 1);
    wire out_fire = m_axis_tvalid && m_axis_tready;

    wire job_busy = start_req || start_r;
    wire can_start = !job_busy && !out_pending;
    wire job_finished = start_r && accel_done;
    wire load_restart = wr_fire && (wr_reg == R_LOAD_TARGET || wr_reg == R_LOAD_ADDR);

    always @(posedge clk) begin
        if (!rst_n) begin
            s_axil_bvalid <= 0;
            s_axil_rvalid <= 0;
            s_axil_rdata <= 0;
            job_r <= 0;
            beam_r <= 0;
            char_r <= 0;
            ctx_addr_r <= 0;
            start_req <= 0;
            start_r <= 0;
            run_job <= 0;
            run_beam <= 0;
            run_char <= 0;
            job_done <= 0;
            job_error <= 0;
            target <= T_WEIGHT;
            load_addr <= 0;
            row_addr <= 0;
            load_count <= 0;
            word_cnt <= 0;
            row_buf <= 0;
            writing <= 0;
            byte_k <= 0;
            out_buf <= 0;
            out_idx <= 0;
            out_pending <= 0;
        end else begin
            // Register writes.
            if (wr_fire) begin
                s_axil_bvalid <= 1;
                case (wr_reg)
                    R_CTRL: begin
                        if (s_axil_wdata[0] && can_start) begin
                            start_req <= 1;
                            run_job <= job_r;
                            run_beam <= beam_r;
                            run_char <= char_r;
                            job_done <= 0;
                            job_error <= 0;
                        end
                    end
                    R_JOB: job_r <= s_axil_wdata[1:0];
                    R_BEAM: beam_r <= s_axil_wdata[BEAM_W-1:0];
                    R_CHAR: char_r <= s_axil_wdata[CHAR_W-1:0];
                    R_LOAD_TARGET: target <= s_axil_wdata[2:0];
                    R_LOAD_ADDR: load_addr <= s_axil_wdata;
                    R_CTX_ADDR: ctx_addr_r <= s_axil_wdata[CONTEXT_AW-1:0];
                    default: begin
                    end
                endcase
            end else if (s_axil_bvalid && s_axil_bready) begin
                s_axil_bvalid <= 0;
            end

            // Register reads.
            if (rd_fire) begin
                s_axil_rvalid <= 1;
                case (rd_reg)
                    R_CTRL: s_axil_rdata <= {31'd0, job_busy};
                    R_JOB: s_axil_rdata <= {30'd0, job_r};
                    R_BEAM: s_axil_rdata <= beam_r;
                    R_CHAR: s_axil_rdata <= char_r;
                    R_STATUS: s_axil_rdata <= {28'd0, out_pending, job_error, job_done, job_busy};
                    R_LOAD_TARGET: s_axil_rdata <= {29'd0, target};
                    R_LOAD_ADDR: s_axil_rdata <= load_addr;
                    R_LOAD_COUNT: s_axil_rdata <= load_count;
                    R_CTX_ADDR: s_axil_rdata <= ctx_addr_r;
                    R_CTX_DATA: s_axil_rdata <= {8'd0, context_dout};
                    default: s_axil_rdata <= 0;
                endcase
            end else if (s_axil_rvalid && s_axil_rready) begin
                s_axil_rvalid <= 0;
            end

            // A held start goes to the accelerator once no row is being
            // written (start_req keeps the stream from starting another).
            if (start_req && !writing) begin
                start_req <= 0;
                start_r <= 1;
            end

            // The accelerator holds done until start falls; drop start then,
            // and queue the scores of an AM or LM job.
            if (job_finished) begin
                start_r <= 0;
                job_done <= 1;
                job_error <= accel_error;
                if (run_job == 2'd0 || run_job == 2'd1) begin
                    out_buf <= $unsigned(scores);
                    out_idx <= 0;
                    out_pending <= 1;
                end
            end else if (out_fire) begin
                if (m_axis_tlast)
                    out_pending <= 0;
                else
                    out_idx <= out_idx + 1'b1;
            end

            // Loader. A restart clears any partial row; its new target and
            // address take effect from the register write above.
            if (load_restart) begin
                row_addr <= (wr_reg == R_LOAD_ADDR) ? s_axil_wdata : load_addr;
                load_count <= 0;
                word_cnt <= 0;
                writing <= 0;
            end else if (writing) begin
                row_addr <= row_addr + 1;
                if (target == T_FRAME) begin
                    if (row_addr < AM_IN)
                        load_count <= load_count + 1;
                    byte_k <= byte_k + 1'b1;
                    if (byte_k == 2'd3)
                        writing <= 0;
                end else begin
                    load_count <= load_count + 1;
                    writing <= 0;
                end
            end else if (in_fire) begin
                row_buf <= {s_axis_tdata, row_buf[BUF_BITS-1:32]};
                if (word_cnt == words_per_row - 1'b1) begin
                    word_cnt <= 0;
                    writing <= 1;
                    byte_k <= 0;
                end else begin
                    word_cnt <= word_cnt + 1'b1;
                end
            end
        end
    end

    // A row's first word is the lowest of its last N words in row_buf.
    wire [WT_BITS-1:0] wt_row = row_buf[BUF_BITS-WT_WORDS*32 +: WT_BITS];
    wire [BS_BITS-1:0] bs_row = row_buf[BUF_BITS-BS_WORDS*32 +: BS_BITS];
    wire [OW_BITS-1:0] ow_row = row_buf[BUF_BITS-OW_WORDS*32 +: OW_BITS];
    wire [OB_BITS-1:0] ob_row = row_buf[BUF_BITS-OB_WORDS*32 +: OB_BITS];
    wire [7:0] frame_byte = row_buf[BUF_BITS-32 + 8*byte_k +: 8];

    sr_accelerator_top #(
        .PE_COUNT(PE_COUNT),
        .AM_LAYERS(AM_LAYERS),
        .LM_LAYERS(LM_LAYERS),
        .AM_IN(AM_IN),
        .LM_IN(LM_IN),
        .VOCAB(VOCAB),
        .N_BEAMS(N_BEAMS),
        .WEIGHT_DEPTH(WEIGHT_DEPTH),
        .CONTEXT_DEPTH(CONTEXT_DEPTH)
    ) accel (
        .clk(clk),
        .rst(!rst_n),
        .start(start_r),
        .job(run_job),
        .beam_id(run_beam),
        .lm_char(run_char),
        .frame_we(writing && target == T_FRAME && row_addr < AM_IN),
        .frame_addr(row_addr[FRAME_AW-1:0]),
        .frame_data(frame_byte),
        .host_context_addr(ctx_addr_r),
        .host_weight0_we(writing && target == T_WEIGHT),
        .host_weight1_we(writing && target == T_WEIGHT),
        .host_weight_addr(row_addr[WEIGHT_AW-1:0]),
        .host_weight0_din(wt_row[PE_COUNT*W_W-1:0]),
        .host_weight1_din(wt_row[WT_BITS-1:PE_COUNT*W_W]),
        .host_bias_we(writing && target == T_BIAS),
        .host_bias_addr(row_addr[BIAS_AW-1:0]),
        .host_bias0_din(bs_row[PE_COUNT*B_W-1:0]),
        .host_bias1_din(bs_row[BS_BITS-1:PE_COUNT*B_W]),
        .host_out_weight_we(writing && target == T_OUT_WEIGHT),
        .host_out_weight_addr(row_addr[OUT_AW-1:0]),
        .host_out_weight_din(ow_row),
        .host_out_bias_we(writing && target == T_OUT_BIAS),
        .host_out_bias_addr(row_addr[0]),
        .host_out_bias_din(ob_row),
        .busy(accel_busy),
        .done(accel_done),
        .error(accel_error),
        .state_out(),
        .score_valid(),
        .score_idx(),
        .score(),
        .scores(scores),
        .context_dout(context_dout)
    );

endmodule
