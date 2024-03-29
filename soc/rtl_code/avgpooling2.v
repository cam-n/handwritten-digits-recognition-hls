// ==============================================================
// RTL generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC
// Version: 2019.1
// Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

`timescale 1 ns / 1 ps 

module avgpooling2 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        in_r_address0,
        in_r_ce0,
        in_r_q0,
        in_r_address1,
        in_r_ce1,
        in_r_q1,
        out_r_address0,
        out_r_ce0,
        out_r_we0,
        out_r_d0
);

parameter    ap_ST_fsm_state1 = 25'd1;
parameter    ap_ST_fsm_state2 = 25'd2;
parameter    ap_ST_fsm_state3 = 25'd4;
parameter    ap_ST_fsm_state4 = 25'd8;
parameter    ap_ST_fsm_state5 = 25'd16;
parameter    ap_ST_fsm_state6 = 25'd32;
parameter    ap_ST_fsm_state7 = 25'd64;
parameter    ap_ST_fsm_state8 = 25'd128;
parameter    ap_ST_fsm_state9 = 25'd256;
parameter    ap_ST_fsm_state10 = 25'd512;
parameter    ap_ST_fsm_state11 = 25'd1024;
parameter    ap_ST_fsm_state12 = 25'd2048;
parameter    ap_ST_fsm_state13 = 25'd4096;
parameter    ap_ST_fsm_state14 = 25'd8192;
parameter    ap_ST_fsm_state15 = 25'd16384;
parameter    ap_ST_fsm_state16 = 25'd32768;
parameter    ap_ST_fsm_state17 = 25'd65536;
parameter    ap_ST_fsm_state18 = 25'd131072;
parameter    ap_ST_fsm_state19 = 25'd262144;
parameter    ap_ST_fsm_state20 = 25'd524288;
parameter    ap_ST_fsm_state21 = 25'd1048576;
parameter    ap_ST_fsm_state22 = 25'd2097152;
parameter    ap_ST_fsm_state23 = 25'd4194304;
parameter    ap_ST_fsm_state24 = 25'd8388608;
parameter    ap_ST_fsm_state25 = 25'd16777216;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
output  [10:0] in_r_address0;
output   in_r_ce0;
input  [31:0] in_r_q0;
output  [10:0] in_r_address1;
output   in_r_ce1;
input  [31:0] in_r_q1;
output  [8:0] out_r_address0;
output   out_r_ce0;
output   out_r_we0;
output  [31:0] out_r_d0;

reg ap_done;
reg ap_idle;
reg ap_ready;
reg[10:0] in_r_address0;
reg in_r_ce0;
reg[10:0] in_r_address1;
reg in_r_ce1;
reg out_r_ce0;
reg out_r_we0;

(* fsm_encoding = "none" *) reg   [24:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
reg   [31:0] reg_149;
wire    ap_CS_fsm_state5;
wire    ap_CS_fsm_state10;
wire    ap_CS_fsm_state15;
wire   [31:0] grp_fu_140_p2;
reg   [31:0] reg_156;
wire    ap_CS_fsm_state20;
wire   [4:0] n_channel_fu_168_p2;
reg   [4:0] n_channel_reg_441;
wire    ap_CS_fsm_state2;
wire   [8:0] add_ln81_fu_202_p2;
reg   [8:0] add_ln81_reg_446;
wire   [0:0] icmp_ln78_fu_162_p2;
wire   [7:0] add_ln81_1_fu_220_p2;
reg   [7:0] add_ln81_1_reg_452;
wire   [11:0] add_ln81_3_fu_261_p2;
reg   [11:0] add_ln81_3_reg_460;
wire    ap_CS_fsm_state3;
wire   [0:0] icmp_ln79_fu_226_p2;
wire   [11:0] add_ln81_5_fu_302_p2;
reg   [11:0] add_ln81_5_reg_466;
wire   [9:0] add_ln81_7_fu_339_p2;
reg   [9:0] add_ln81_7_reg_472;
wire    ap_CS_fsm_state4;
wire   [0:0] icmp_ln80_fu_345_p2;
wire   [11:0] add_ln81_10_fu_385_p2;
reg   [11:0] add_ln81_10_reg_490;
wire   [11:0] add_ln81_11_fu_390_p2;
reg   [11:0] add_ln81_11_reg_495;
wire   [9:0] add_ln81_12_fu_409_p2;
reg   [9:0] add_ln81_12_reg_500;
wire   [3:0] j_fu_414_p2;
reg   [3:0] j_reg_505;
wire   [3:0] i_fu_420_p2;
reg   [31:0] in_load_1_reg_515;
wire    ap_CS_fsm_state9;
wire    ap_CS_fsm_state14;
wire   [31:0] grp_fu_144_p2;
reg   [31:0] tmp_2_reg_530;
wire    ap_CS_fsm_state24;
reg   [4:0] n_channel_0_reg_106;
reg   [3:0] i_0_reg_117;
reg   [3:0] j_0_reg_129;
wire    ap_CS_fsm_state25;
wire   [63:0] zext_ln81_11_fu_360_p1;
wire   [63:0] zext_ln81_12_fu_370_p1;
wire   [63:0] zext_ln81_14_fu_426_p1;
wire   [63:0] zext_ln81_15_fu_430_p1;
wire   [63:0] zext_ln81_17_fu_434_p1;
reg   [31:0] grp_fu_140_p0;
reg   [31:0] grp_fu_140_p1;
wire    ap_CS_fsm_state6;
wire    ap_CS_fsm_state11;
wire    ap_CS_fsm_state16;
wire    ap_CS_fsm_state21;
wire   [7:0] tmp_8_fu_178_p3;
wire   [5:0] tmp_9_fu_190_p3;
wire   [8:0] zext_ln81_2_fu_198_p1;
wire   [8:0] zext_ln81_1_fu_186_p1;
wire   [6:0] tmp_10_fu_208_p3;
wire   [7:0] zext_ln81_fu_174_p1;
wire   [7:0] zext_ln81_3_fu_216_p1;
wire   [8:0] zext_ln81_4_fu_232_p1;
wire   [8:0] add_ln81_2_fu_236_p2;
wire   [9:0] tmp_11_fu_249_p3;
wire   [11:0] p_shl6_cast_fu_241_p3;
wire   [11:0] zext_ln81_5_fu_257_p1;
wire   [3:0] or_ln81_fu_267_p2;
wire   [8:0] zext_ln81_6_fu_273_p1;
wire   [8:0] add_ln81_4_fu_277_p2;
wire   [9:0] tmp_12_fu_290_p3;
wire   [11:0] p_shl4_cast_fu_282_p3;
wire   [11:0] zext_ln81_7_fu_298_p1;
wire   [2:0] tmp_13_fu_308_p4;
wire   [7:0] zext_ln81_8_fu_318_p1;
wire   [7:0] add_ln81_6_fu_322_p2;
wire   [9:0] p_shl3_cast_fu_331_p3;
wire   [9:0] zext_ln81_9_fu_327_p1;
wire   [11:0] zext_ln81_10_fu_351_p1;
wire   [11:0] add_ln81_8_fu_355_p2;
wire   [11:0] add_ln81_9_fu_365_p2;
wire   [3:0] or_ln81_1_fu_375_p2;
wire   [11:0] zext_ln81_13_fu_381_p1;
wire   [2:0] tmp_14_fu_395_p4;
wire   [9:0] zext_ln81_16_fu_405_p1;
reg   [24:0] ap_NS_fsm;

// power-on initialization
initial begin
#0 ap_CS_fsm = 25'd1;
end

Prediction_fadd_3bkb #(
    .ID( 1 ),
    .NUM_STAGE( 5 ),
    .din0_WIDTH( 32 ),
    .din1_WIDTH( 32 ),
    .dout_WIDTH( 32 ))
Prediction_fadd_3bkb_U26(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(grp_fu_140_p0),
    .din1(grp_fu_140_p1),
    .ce(1'b1),
    .dout(grp_fu_140_p2)
);

Prediction_fmul_3cud #(
    .ID( 1 ),
    .NUM_STAGE( 4 ),
    .din0_WIDTH( 32 ),
    .din1_WIDTH( 32 ),
    .dout_WIDTH( 32 ))
Prediction_fmul_3cud_U27(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(reg_156),
    .din1(32'd1048576000),
    .ce(1'b1),
    .dout(grp_fu_144_p2)
);

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_fsm <= ap_ST_fsm_state1;
    end else begin
        ap_CS_fsm <= ap_NS_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (((icmp_ln78_fu_162_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state2))) begin
        i_0_reg_117 <= 4'd0;
    end else if (((icmp_ln80_fu_345_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state4))) begin
        i_0_reg_117 <= i_fu_420_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_state3) & (icmp_ln79_fu_226_p2 == 1'd1))) begin
        j_0_reg_129 <= 4'd0;
    end else if ((1'b1 == ap_CS_fsm_state25)) begin
        j_0_reg_129 <= j_reg_505;
    end
end

always @ (posedge ap_clk) begin
    if (((icmp_ln79_fu_226_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state3))) begin
        n_channel_0_reg_106 <= n_channel_reg_441;
    end else if (((ap_start == 1'b1) & (1'b1 == ap_CS_fsm_state1))) begin
        n_channel_0_reg_106 <= 5'd0;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state15)) begin
        reg_149 <= in_r_q1;
    end else if (((1'b1 == ap_CS_fsm_state10) | (1'b1 == ap_CS_fsm_state5))) begin
        reg_149 <= in_r_q0;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_state4) & (icmp_ln80_fu_345_p2 == 1'd1))) begin
        add_ln81_10_reg_490[11 : 1] <= add_ln81_10_fu_385_p2[11 : 1];
        add_ln81_11_reg_495[11 : 1] <= add_ln81_11_fu_390_p2[11 : 1];
        add_ln81_12_reg_500 <= add_ln81_12_fu_409_p2;
        j_reg_505 <= j_fu_414_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((icmp_ln78_fu_162_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state2))) begin
        add_ln81_1_reg_452 <= add_ln81_1_fu_220_p2;
        add_ln81_reg_446[8 : 1] <= add_ln81_fu_202_p2[8 : 1];
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_state3) & (icmp_ln79_fu_226_p2 == 1'd1))) begin
        add_ln81_3_reg_460[11 : 1] <= add_ln81_3_fu_261_p2[11 : 1];
        add_ln81_5_reg_466[11 : 2] <= add_ln81_5_fu_302_p2[11 : 2];
        add_ln81_7_reg_472 <= add_ln81_7_fu_339_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state5)) begin
        in_load_1_reg_515 <= in_r_q1;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state2)) begin
        n_channel_reg_441 <= n_channel_fu_168_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_state20) | (1'b1 == ap_CS_fsm_state15) | (1'b1 == ap_CS_fsm_state10))) begin
        reg_156 <= grp_fu_140_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state24)) begin
        tmp_2_reg_530 <= grp_fu_144_p2;
    end
end

always @ (*) begin
    if ((((icmp_ln78_fu_162_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state2)) | ((ap_start == 1'b0) & (1'b1 == ap_CS_fsm_state1)))) begin
        ap_done = 1'b1;
    end else begin
        ap_done = 1'b0;
    end
end

always @ (*) begin
    if (((ap_start == 1'b0) & (1'b1 == ap_CS_fsm_state1))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln78_fu_162_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state2))) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state16) | (1'b1 == ap_CS_fsm_state11))) begin
        grp_fu_140_p0 = reg_156;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        grp_fu_140_p0 = reg_149;
    end else begin
        grp_fu_140_p0 = 'bx;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state16) | (1'b1 == ap_CS_fsm_state11))) begin
        grp_fu_140_p1 = reg_149;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        grp_fu_140_p1 = in_load_1_reg_515;
    end else begin
        grp_fu_140_p1 = 'bx;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state9)) begin
        in_r_address0 = zext_ln81_14_fu_426_p1;
    end else if ((1'b1 == ap_CS_fsm_state4)) begin
        in_r_address0 = zext_ln81_11_fu_360_p1;
    end else begin
        in_r_address0 = 'bx;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        in_r_address1 = zext_ln81_15_fu_430_p1;
    end else if ((1'b1 == ap_CS_fsm_state4)) begin
        in_r_address1 = zext_ln81_12_fu_370_p1;
    end else begin
        in_r_address1 = 'bx;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state9) | (1'b1 == ap_CS_fsm_state4))) begin
        in_r_ce0 = 1'b1;
    end else begin
        in_r_ce0 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state14) | (1'b1 == ap_CS_fsm_state4))) begin
        in_r_ce1 = 1'b1;
    end else begin
        in_r_ce1 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state25)) begin
        out_r_ce0 = 1'b1;
    end else begin
        out_r_ce0 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state25)) begin
        out_r_we0 = 1'b1;
    end else begin
        out_r_we0 = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            if (((ap_start == 1'b1) & (1'b1 == ap_CS_fsm_state1))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end
        end
        ap_ST_fsm_state2 : begin
            if (((icmp_ln78_fu_162_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state2))) begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end
        end
        ap_ST_fsm_state3 : begin
            if (((icmp_ln79_fu_226_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state3))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state4;
            end
        end
        ap_ST_fsm_state4 : begin
            if (((icmp_ln80_fu_345_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state4))) begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state5;
            end
        end
        ap_ST_fsm_state5 : begin
            ap_NS_fsm = ap_ST_fsm_state6;
        end
        ap_ST_fsm_state6 : begin
            ap_NS_fsm = ap_ST_fsm_state7;
        end
        ap_ST_fsm_state7 : begin
            ap_NS_fsm = ap_ST_fsm_state8;
        end
        ap_ST_fsm_state8 : begin
            ap_NS_fsm = ap_ST_fsm_state9;
        end
        ap_ST_fsm_state9 : begin
            ap_NS_fsm = ap_ST_fsm_state10;
        end
        ap_ST_fsm_state10 : begin
            ap_NS_fsm = ap_ST_fsm_state11;
        end
        ap_ST_fsm_state11 : begin
            ap_NS_fsm = ap_ST_fsm_state12;
        end
        ap_ST_fsm_state12 : begin
            ap_NS_fsm = ap_ST_fsm_state13;
        end
        ap_ST_fsm_state13 : begin
            ap_NS_fsm = ap_ST_fsm_state14;
        end
        ap_ST_fsm_state14 : begin
            ap_NS_fsm = ap_ST_fsm_state15;
        end
        ap_ST_fsm_state15 : begin
            ap_NS_fsm = ap_ST_fsm_state16;
        end
        ap_ST_fsm_state16 : begin
            ap_NS_fsm = ap_ST_fsm_state17;
        end
        ap_ST_fsm_state17 : begin
            ap_NS_fsm = ap_ST_fsm_state18;
        end
        ap_ST_fsm_state18 : begin
            ap_NS_fsm = ap_ST_fsm_state19;
        end
        ap_ST_fsm_state19 : begin
            ap_NS_fsm = ap_ST_fsm_state20;
        end
        ap_ST_fsm_state20 : begin
            ap_NS_fsm = ap_ST_fsm_state21;
        end
        ap_ST_fsm_state21 : begin
            ap_NS_fsm = ap_ST_fsm_state22;
        end
        ap_ST_fsm_state22 : begin
            ap_NS_fsm = ap_ST_fsm_state23;
        end
        ap_ST_fsm_state23 : begin
            ap_NS_fsm = ap_ST_fsm_state24;
        end
        ap_ST_fsm_state24 : begin
            ap_NS_fsm = ap_ST_fsm_state25;
        end
        ap_ST_fsm_state25 : begin
            ap_NS_fsm = ap_ST_fsm_state4;
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign add_ln81_10_fu_385_p2 = (add_ln81_3_reg_460 + zext_ln81_13_fu_381_p1);

assign add_ln81_11_fu_390_p2 = (add_ln81_5_reg_466 + zext_ln81_13_fu_381_p1);

assign add_ln81_12_fu_409_p2 = (add_ln81_7_reg_472 + zext_ln81_16_fu_405_p1);

assign add_ln81_1_fu_220_p2 = (zext_ln81_fu_174_p1 + zext_ln81_3_fu_216_p1);

assign add_ln81_2_fu_236_p2 = (zext_ln81_4_fu_232_p1 + add_ln81_reg_446);

assign add_ln81_3_fu_261_p2 = (p_shl6_cast_fu_241_p3 + zext_ln81_5_fu_257_p1);

assign add_ln81_4_fu_277_p2 = (zext_ln81_6_fu_273_p1 + add_ln81_reg_446);

assign add_ln81_5_fu_302_p2 = (p_shl4_cast_fu_282_p3 + zext_ln81_7_fu_298_p1);

assign add_ln81_6_fu_322_p2 = (zext_ln81_8_fu_318_p1 + add_ln81_1_reg_452);

assign add_ln81_7_fu_339_p2 = (p_shl3_cast_fu_331_p3 + zext_ln81_9_fu_327_p1);

assign add_ln81_8_fu_355_p2 = (add_ln81_3_reg_460 + zext_ln81_10_fu_351_p1);

assign add_ln81_9_fu_365_p2 = (add_ln81_5_reg_466 + zext_ln81_10_fu_351_p1);

assign add_ln81_fu_202_p2 = (zext_ln81_2_fu_198_p1 + zext_ln81_1_fu_186_p1);

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state10 = ap_CS_fsm[32'd9];

assign ap_CS_fsm_state11 = ap_CS_fsm[32'd10];

assign ap_CS_fsm_state14 = ap_CS_fsm[32'd13];

assign ap_CS_fsm_state15 = ap_CS_fsm[32'd14];

assign ap_CS_fsm_state16 = ap_CS_fsm[32'd15];

assign ap_CS_fsm_state2 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_state20 = ap_CS_fsm[32'd19];

assign ap_CS_fsm_state21 = ap_CS_fsm[32'd20];

assign ap_CS_fsm_state24 = ap_CS_fsm[32'd23];

assign ap_CS_fsm_state25 = ap_CS_fsm[32'd24];

assign ap_CS_fsm_state3 = ap_CS_fsm[32'd2];

assign ap_CS_fsm_state4 = ap_CS_fsm[32'd3];

assign ap_CS_fsm_state5 = ap_CS_fsm[32'd4];

assign ap_CS_fsm_state6 = ap_CS_fsm[32'd5];

assign ap_CS_fsm_state9 = ap_CS_fsm[32'd8];

assign i_fu_420_p2 = (i_0_reg_117 + 4'd2);

assign icmp_ln78_fu_162_p2 = ((n_channel_0_reg_106 == 5'd16) ? 1'b1 : 1'b0);

assign icmp_ln79_fu_226_p2 = ((i_0_reg_117 < 4'd10) ? 1'b1 : 1'b0);

assign icmp_ln80_fu_345_p2 = ((j_0_reg_129 < 4'd10) ? 1'b1 : 1'b0);

assign j_fu_414_p2 = (j_0_reg_129 + 4'd2);

assign n_channel_fu_168_p2 = (n_channel_0_reg_106 + 5'd1);

assign or_ln81_1_fu_375_p2 = (j_0_reg_129 | 4'd1);

assign or_ln81_fu_267_p2 = (i_0_reg_117 | 4'd1);

assign out_r_address0 = zext_ln81_17_fu_434_p1;

assign out_r_d0 = tmp_2_reg_530;

assign p_shl3_cast_fu_331_p3 = {{add_ln81_6_fu_322_p2}, {2'd0}};

assign p_shl4_cast_fu_282_p3 = {{add_ln81_4_fu_277_p2}, {3'd0}};

assign p_shl6_cast_fu_241_p3 = {{add_ln81_2_fu_236_p2}, {3'd0}};

assign tmp_10_fu_208_p3 = {{n_channel_0_reg_106}, {2'd0}};

assign tmp_11_fu_249_p3 = {{add_ln81_2_fu_236_p2}, {1'd0}};

assign tmp_12_fu_290_p3 = {{add_ln81_4_fu_277_p2}, {1'd0}};

assign tmp_13_fu_308_p4 = {{i_0_reg_117[3:1]}};

assign tmp_14_fu_395_p4 = {{j_0_reg_129[3:1]}};

assign tmp_8_fu_178_p3 = {{n_channel_0_reg_106}, {3'd0}};

assign tmp_9_fu_190_p3 = {{n_channel_0_reg_106}, {1'd0}};

assign zext_ln81_10_fu_351_p1 = j_0_reg_129;

assign zext_ln81_11_fu_360_p1 = add_ln81_8_fu_355_p2;

assign zext_ln81_12_fu_370_p1 = add_ln81_9_fu_365_p2;

assign zext_ln81_13_fu_381_p1 = or_ln81_1_fu_375_p2;

assign zext_ln81_14_fu_426_p1 = add_ln81_10_reg_490;

assign zext_ln81_15_fu_430_p1 = add_ln81_11_reg_495;

assign zext_ln81_16_fu_405_p1 = tmp_14_fu_395_p4;

assign zext_ln81_17_fu_434_p1 = add_ln81_12_reg_500;

assign zext_ln81_1_fu_186_p1 = tmp_8_fu_178_p3;

assign zext_ln81_2_fu_198_p1 = tmp_9_fu_190_p3;

assign zext_ln81_3_fu_216_p1 = tmp_10_fu_208_p3;

assign zext_ln81_4_fu_232_p1 = i_0_reg_117;

assign zext_ln81_5_fu_257_p1 = tmp_11_fu_249_p3;

assign zext_ln81_6_fu_273_p1 = or_ln81_fu_267_p2;

assign zext_ln81_7_fu_298_p1 = tmp_12_fu_290_p3;

assign zext_ln81_8_fu_318_p1 = tmp_13_fu_308_p4;

assign zext_ln81_9_fu_327_p1 = add_ln81_6_fu_322_p2;

assign zext_ln81_fu_174_p1 = n_channel_0_reg_106;

always @ (posedge ap_clk) begin
    add_ln81_reg_446[0] <= 1'b0;
    add_ln81_3_reg_460[0] <= 1'b0;
    add_ln81_5_reg_466[1:0] <= 2'b10;
    add_ln81_10_reg_490[0] <= 1'b1;
    add_ln81_11_reg_495[0] <= 1'b1;
end

endmodule //avgpooling2
