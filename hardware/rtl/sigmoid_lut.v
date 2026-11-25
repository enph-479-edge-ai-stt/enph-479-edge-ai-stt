`timescale 1ps / 1ps

//=====================================================================
// Module: sigmoid_lut.v
// Description:
//  Combinational sigmoid table: 512 entries, unsigned 8-bit out. The
//  index is a gate pre-activation at 2^-5, saturated to 9 bits and put
//  in offset binary (sign bit flipped), so index n is the real value
//  v = (n - 256) / 32, from -8 to 8 - 1/32. Entry n holds
//
//    clamp(round(256 / (1 + exp(-v))), 0, 255)
//
//  sigmoid(0) is 128 at index 256. The value is the gate at 2^-8.
//  Formats in shared/specs/weight_image.md ("Activations and cell").
//  None of the entries sits on a rounding tie, so half up and half even
//  give the same table.
//
// Ports:
//  index           - offset-binary gate value
//  sigmoid_output  - sigmoid of it, 0..255
//=====================================================================

module sigmoid_lut (
    input wire [8:0] index,
    output reg [7:0] sigmoid_output
);

    reg [7:0] lut [0:511];

    initial begin
        lut[0] = 8'd0;   lut[1] = 8'd0;   lut[2] = 8'd0;   lut[3] = 8'd0;   lut[4] = 8'd0;   lut[5] = 8'd0;   lut[6] = 8'd0;   lut[7] = 8'd0;
        lut[8] = 8'd0;   lut[9] = 8'd0;   lut[10] = 8'd0;  lut[11] = 8'd0;  lut[12] = 8'd0;  lut[13] = 8'd0;  lut[14] = 8'd0;  lut[15] = 8'd0;
        lut[16] = 8'd0;  lut[17] = 8'd0;  lut[18] = 8'd0;  lut[19] = 8'd0;  lut[20] = 8'd0;  lut[21] = 8'd0;  lut[22] = 8'd0;  lut[23] = 8'd0;
        lut[24] = 8'd0;  lut[25] = 8'd0;  lut[26] = 8'd0;  lut[27] = 8'd0;  lut[28] = 8'd0;  lut[29] = 8'd0;  lut[30] = 8'd0;  lut[31] = 8'd0;
        lut[32] = 8'd0;  lut[33] = 8'd0;  lut[34] = 8'd0;  lut[35] = 8'd0;  lut[36] = 8'd0;  lut[37] = 8'd0;  lut[38] = 8'd0;  lut[39] = 8'd0;
        lut[40] = 8'd0;  lut[41] = 8'd0;  lut[42] = 8'd0;  lut[43] = 8'd0;  lut[44] = 8'd0;  lut[45] = 8'd0;  lut[46] = 8'd0;  lut[47] = 8'd0;
        lut[48] = 8'd0;  lut[49] = 8'd0;  lut[50] = 8'd0;  lut[51] = 8'd0;  lut[52] = 8'd0;  lut[53] = 8'd0;  lut[54] = 8'd0;  lut[55] = 8'd0;
        lut[56] = 8'd0;  lut[57] = 8'd1;  lut[58] = 8'd1;  lut[59] = 8'd1;  lut[60] = 8'd1;  lut[61] = 8'd1;  lut[62] = 8'd1;  lut[63] = 8'd1;
        lut[64] = 8'd1;  lut[65] = 8'd1;  lut[66] = 8'd1;  lut[67] = 8'd1;  lut[68] = 8'd1;  lut[69] = 8'd1;  lut[70] = 8'd1;  lut[71] = 8'd1;
        lut[72] = 8'd1;  lut[73] = 8'd1;  lut[74] = 8'd1;  lut[75] = 8'd1;  lut[76] = 8'd1;  lut[77] = 8'd1;  lut[78] = 8'd1;  lut[79] = 8'd1;
        lut[80] = 8'd1;  lut[81] = 8'd1;  lut[82] = 8'd1;  lut[83] = 8'd1;  lut[84] = 8'd1;  lut[85] = 8'd1;  lut[86] = 8'd1;  lut[87] = 8'd1;
        lut[88] = 8'd1;  lut[89] = 8'd1;  lut[90] = 8'd1;  lut[91] = 8'd1;  lut[92] = 8'd2;  lut[93] = 8'd2;  lut[94] = 8'd2;  lut[95] = 8'd2;
        lut[96] = 8'd2;  lut[97] = 8'd2;  lut[98] = 8'd2;  lut[99] = 8'd2;  lut[100] = 8'd2; lut[101] = 8'd2; lut[102] = 8'd2; lut[103] = 8'd2;
        lut[104] = 8'd2; lut[105] = 8'd2; lut[106] = 8'd2; lut[107] = 8'd2; lut[108] = 8'd2; lut[109] = 8'd3; lut[110] = 8'd3; lut[111] = 8'd3;
        lut[112] = 8'd3; lut[113] = 8'd3; lut[114] = 8'd3; lut[115] = 8'd3; lut[116] = 8'd3; lut[117] = 8'd3; lut[118] = 8'd3; lut[119] = 8'd3;
        lut[120] = 8'd4; lut[121] = 8'd4; lut[122] = 8'd4; lut[123] = 8'd4; lut[124] = 8'd4; lut[125] = 8'd4; lut[126] = 8'd4; lut[127] = 8'd4;
        lut[128] = 8'd5; lut[129] = 8'd5; lut[130] = 8'd5; lut[131] = 8'd5; lut[132] = 8'd5; lut[133] = 8'd5; lut[134] = 8'd6; lut[135] = 8'd6;
        lut[136] = 8'd6; lut[137] = 8'd6; lut[138] = 8'd6; lut[139] = 8'd6; lut[140] = 8'd7; lut[141] = 8'd7; lut[142] = 8'd7; lut[143] = 8'd7;
        lut[144] = 8'd8; lut[145] = 8'd8; lut[146] = 8'd8; lut[147] = 8'd8; lut[148] = 8'd8; lut[149] = 8'd9; lut[150] = 8'd9; lut[151] = 8'd9;
        lut[152] = 8'd10; lut[153] = 8'd10; lut[154] = 8'd10; lut[155] = 8'd10; lut[156] = 8'd11; lut[157] = 8'd11; lut[158] = 8'd11; lut[159] = 8'd12;
        lut[160] = 8'd12; lut[161] = 8'd13; lut[162] = 8'd13; lut[163] = 8'd13; lut[164] = 8'd14; lut[165] = 8'd14; lut[166] = 8'd15; lut[167] = 8'd15;
        lut[168] = 8'd15; lut[169] = 8'd16; lut[170] = 8'd16; lut[171] = 8'd17; lut[172] = 8'd17; lut[173] = 8'd18; lut[174] = 8'd18; lut[175] = 8'd19;
        lut[176] = 8'd19; lut[177] = 8'd20; lut[178] = 8'd21; lut[179] = 8'd21; lut[180] = 8'd22; lut[181] = 8'd22; lut[182] = 8'd23; lut[183] = 8'd24;
        lut[184] = 8'd24; lut[185] = 8'd25; lut[186] = 8'd26; lut[187] = 8'd27; lut[188] = 8'd27; lut[189] = 8'd28; lut[190] = 8'd29; lut[191] = 8'd30;
        lut[192] = 8'd31; lut[193] = 8'd31; lut[194] = 8'd32; lut[195] = 8'd33; lut[196] = 8'd34; lut[197] = 8'd35; lut[198] = 8'd36; lut[199] = 8'd37;
        lut[200] = 8'd38; lut[201] = 8'd39; lut[202] = 8'd40; lut[203] = 8'd41; lut[204] = 8'd42; lut[205] = 8'd43; lut[206] = 8'd44; lut[207] = 8'd46;
        lut[208] = 8'd47; lut[209] = 8'd48; lut[210] = 8'd49; lut[211] = 8'd50; lut[212] = 8'd52; lut[213] = 8'd53; lut[214] = 8'd54; lut[215] = 8'd56;
        lut[216] = 8'd57; lut[217] = 8'd58; lut[218] = 8'd60; lut[219] = 8'd61; lut[220] = 8'd63; lut[221] = 8'd64; lut[222] = 8'd66; lut[223] = 8'd67;
        lut[224] = 8'd69; lut[225] = 8'd70; lut[226] = 8'd72; lut[227] = 8'd74; lut[228] = 8'd75; lut[229] = 8'd77; lut[230] = 8'd79; lut[231] = 8'd80;
        lut[232] = 8'd82; lut[233] = 8'd84; lut[234] = 8'd86; lut[235] = 8'd87; lut[236] = 8'd89; lut[237] = 8'd91; lut[238] = 8'd93; lut[239] = 8'd95;
        lut[240] = 8'd97; lut[241] = 8'd99; lut[242] = 8'd100; lut[243] = 8'd102; lut[244] = 8'd104; lut[245] = 8'd106; lut[246] = 8'd108; lut[247] = 8'd110;
        lut[248] = 8'd112; lut[249] = 8'd114; lut[250] = 8'd116; lut[251] = 8'd118; lut[252] = 8'd120; lut[253] = 8'd122; lut[254] = 8'd124; lut[255] = 8'd126;
        lut[256] = 8'd128; lut[257] = 8'd130; lut[258] = 8'd132; lut[259] = 8'd134; lut[260] = 8'd136; lut[261] = 8'd138; lut[262] = 8'd140; lut[263] = 8'd142;
        lut[264] = 8'd144; lut[265] = 8'd146; lut[266] = 8'd148; lut[267] = 8'd150; lut[268] = 8'd152; lut[269] = 8'd154; lut[270] = 8'd156; lut[271] = 8'd157;
        lut[272] = 8'd159; lut[273] = 8'd161; lut[274] = 8'd163; lut[275] = 8'd165; lut[276] = 8'd167; lut[277] = 8'd169; lut[278] = 8'd170; lut[279] = 8'd172;
        lut[280] = 8'd174; lut[281] = 8'd176; lut[282] = 8'd177; lut[283] = 8'd179; lut[284] = 8'd181; lut[285] = 8'd182; lut[286] = 8'd184; lut[287] = 8'd186;
        lut[288] = 8'd187; lut[289] = 8'd189; lut[290] = 8'd190; lut[291] = 8'd192; lut[292] = 8'd193; lut[293] = 8'd195; lut[294] = 8'd196; lut[295] = 8'd198;
        lut[296] = 8'd199; lut[297] = 8'd200; lut[298] = 8'd202; lut[299] = 8'd203; lut[300] = 8'd204; lut[301] = 8'd206; lut[302] = 8'd207; lut[303] = 8'd208;
        lut[304] = 8'd209; lut[305] = 8'd210; lut[306] = 8'd212; lut[307] = 8'd213; lut[308] = 8'd214; lut[309] = 8'd215; lut[310] = 8'd216; lut[311] = 8'd217;
        lut[312] = 8'd218; lut[313] = 8'd219; lut[314] = 8'd220; lut[315] = 8'd221; lut[316] = 8'd222; lut[317] = 8'd223; lut[318] = 8'd224; lut[319] = 8'd225;
        lut[320] = 8'd225; lut[321] = 8'd226; lut[322] = 8'd227; lut[323] = 8'd228; lut[324] = 8'd229; lut[325] = 8'd229; lut[326] = 8'd230; lut[327] = 8'd231;
        lut[328] = 8'd232; lut[329] = 8'd232; lut[330] = 8'd233; lut[331] = 8'd234; lut[332] = 8'd234; lut[333] = 8'd235; lut[334] = 8'd235; lut[335] = 8'd236;
        lut[336] = 8'd237; lut[337] = 8'd237; lut[338] = 8'd238; lut[339] = 8'd238; lut[340] = 8'd239; lut[341] = 8'd239; lut[342] = 8'd240; lut[343] = 8'd240;
        lut[344] = 8'd241; lut[345] = 8'd241; lut[346] = 8'd241; lut[347] = 8'd242; lut[348] = 8'd242; lut[349] = 8'd243; lut[350] = 8'd243; lut[351] = 8'd243;
        lut[352] = 8'd244; lut[353] = 8'd244; lut[354] = 8'd245; lut[355] = 8'd245; lut[356] = 8'd245; lut[357] = 8'd246; lut[358] = 8'd246; lut[359] = 8'd246;
        lut[360] = 8'd246; lut[361] = 8'd247; lut[362] = 8'd247; lut[363] = 8'd247; lut[364] = 8'd248; lut[365] = 8'd248; lut[366] = 8'd248; lut[367] = 8'd248;
        lut[368] = 8'd248; lut[369] = 8'd249; lut[370] = 8'd249; lut[371] = 8'd249; lut[372] = 8'd249; lut[373] = 8'd250; lut[374] = 8'd250; lut[375] = 8'd250;
        lut[376] = 8'd250; lut[377] = 8'd250; lut[378] = 8'd250; lut[379] = 8'd251; lut[380] = 8'd251; lut[381] = 8'd251; lut[382] = 8'd251; lut[383] = 8'd251;
        lut[384] = 8'd251; lut[385] = 8'd252; lut[386] = 8'd252; lut[387] = 8'd252; lut[388] = 8'd252; lut[389] = 8'd252; lut[390] = 8'd252; lut[391] = 8'd252;
        lut[392] = 8'd252; lut[393] = 8'd253; lut[394] = 8'd253; lut[395] = 8'd253; lut[396] = 8'd253; lut[397] = 8'd253; lut[398] = 8'd253; lut[399] = 8'd253;
        lut[400] = 8'd253; lut[401] = 8'd253; lut[402] = 8'd253; lut[403] = 8'd253; lut[404] = 8'd254; lut[405] = 8'd254; lut[406] = 8'd254; lut[407] = 8'd254;
        lut[408] = 8'd254; lut[409] = 8'd254; lut[410] = 8'd254; lut[411] = 8'd254; lut[412] = 8'd254; lut[413] = 8'd254; lut[414] = 8'd254; lut[415] = 8'd254;
        lut[416] = 8'd254; lut[417] = 8'd254; lut[418] = 8'd254; lut[419] = 8'd254; lut[420] = 8'd254; lut[421] = 8'd255; lut[422] = 8'd255; lut[423] = 8'd255;
        lut[424] = 8'd255; lut[425] = 8'd255; lut[426] = 8'd255; lut[427] = 8'd255; lut[428] = 8'd255; lut[429] = 8'd255; lut[430] = 8'd255; lut[431] = 8'd255;
        lut[432] = 8'd255; lut[433] = 8'd255; lut[434] = 8'd255; lut[435] = 8'd255; lut[436] = 8'd255; lut[437] = 8'd255; lut[438] = 8'd255; lut[439] = 8'd255;
        lut[440] = 8'd255; lut[441] = 8'd255; lut[442] = 8'd255; lut[443] = 8'd255; lut[444] = 8'd255; lut[445] = 8'd255; lut[446] = 8'd255; lut[447] = 8'd255;
        lut[448] = 8'd255; lut[449] = 8'd255; lut[450] = 8'd255; lut[451] = 8'd255; lut[452] = 8'd255; lut[453] = 8'd255; lut[454] = 8'd255; lut[455] = 8'd255;
        lut[456] = 8'd255; lut[457] = 8'd255; lut[458] = 8'd255; lut[459] = 8'd255; lut[460] = 8'd255; lut[461] = 8'd255; lut[462] = 8'd255; lut[463] = 8'd255;
        lut[464] = 8'd255; lut[465] = 8'd255; lut[466] = 8'd255; lut[467] = 8'd255; lut[468] = 8'd255; lut[469] = 8'd255; lut[470] = 8'd255; lut[471] = 8'd255;
        lut[472] = 8'd255; lut[473] = 8'd255; lut[474] = 8'd255; lut[475] = 8'd255; lut[476] = 8'd255; lut[477] = 8'd255; lut[478] = 8'd255; lut[479] = 8'd255;
        lut[480] = 8'd255; lut[481] = 8'd255; lut[482] = 8'd255; lut[483] = 8'd255; lut[484] = 8'd255; lut[485] = 8'd255; lut[486] = 8'd255; lut[487] = 8'd255;
        lut[488] = 8'd255; lut[489] = 8'd255; lut[490] = 8'd255; lut[491] = 8'd255; lut[492] = 8'd255; lut[493] = 8'd255; lut[494] = 8'd255; lut[495] = 8'd255;
        lut[496] = 8'd255; lut[497] = 8'd255; lut[498] = 8'd255; lut[499] = 8'd255; lut[500] = 8'd255; lut[501] = 8'd255; lut[502] = 8'd255; lut[503] = 8'd255;
        lut[504] = 8'd255; lut[505] = 8'd255; lut[506] = 8'd255; lut[507] = 8'd255; lut[508] = 8'd255; lut[509] = 8'd255; lut[510] = 8'd255; lut[511] = 8'd255;
    end

    always @(*) begin
        sigmoid_output = lut[index];
    end
endmodule
