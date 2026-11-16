`timescale 1ps / 1ps

//=====================================================================
// Module: tanh_lut.v
// Description:
//  Combinational tanh table: 512 entries, signed 8-bit out. The index is
//  a value at 2^-5, saturated to 9 bits and put in offset binary (sign
//  bit flipped), so index n is the real value v = (n - 256) / 32, from
//  -8 to 8 - 1/32. Entry n holds
//
//    clamp(round(128 * tanh(v)), -127, 127)
//
//  tanh(0) is 0 at index 256. The value is tanh at 2^-7. The EPU uses
//  it for the candidate gate and for tanh(c_t). Formats in
//  shared/specs/weight_image.md ("Activations and cell"). None of the
//  entries sits on a rounding tie, so half up and half even give the
//  same table.
//
// Ports:
//  index        - offset-binary input value
//  tanh_output  - tanh of it, -127..127
//=====================================================================

module tanh_lut (
    input wire [8:0] index,
    output reg signed [7:0] tanh_output
);

    reg signed [7:0] lut [0:511];

    initial begin
        lut[0] = -8'sd127; lut[1] = -8'sd127; lut[2] = -8'sd127; lut[3] = -8'sd127; lut[4] = -8'sd127; lut[5] = -8'sd127; lut[6] = -8'sd127; lut[7] = -8'sd127;
        lut[8] = -8'sd127; lut[9] = -8'sd127; lut[10] = -8'sd127; lut[11] = -8'sd127; lut[12] = -8'sd127; lut[13] = -8'sd127; lut[14] = -8'sd127; lut[15] = -8'sd127;
        lut[16] = -8'sd127; lut[17] = -8'sd127; lut[18] = -8'sd127; lut[19] = -8'sd127; lut[20] = -8'sd127; lut[21] = -8'sd127; lut[22] = -8'sd127; lut[23] = -8'sd127;
        lut[24] = -8'sd127; lut[25] = -8'sd127; lut[26] = -8'sd127; lut[27] = -8'sd127; lut[28] = -8'sd127; lut[29] = -8'sd127; lut[30] = -8'sd127; lut[31] = -8'sd127;
        lut[32] = -8'sd127; lut[33] = -8'sd127; lut[34] = -8'sd127; lut[35] = -8'sd127; lut[36] = -8'sd127; lut[37] = -8'sd127; lut[38] = -8'sd127; lut[39] = -8'sd127;
        lut[40] = -8'sd127; lut[41] = -8'sd127; lut[42] = -8'sd127; lut[43] = -8'sd127; lut[44] = -8'sd127; lut[45] = -8'sd127; lut[46] = -8'sd127; lut[47] = -8'sd127;
        lut[48] = -8'sd127; lut[49] = -8'sd127; lut[50] = -8'sd127; lut[51] = -8'sd127; lut[52] = -8'sd127; lut[53] = -8'sd127; lut[54] = -8'sd127; lut[55] = -8'sd127;
        lut[56] = -8'sd127; lut[57] = -8'sd127; lut[58] = -8'sd127; lut[59] = -8'sd127; lut[60] = -8'sd127; lut[61] = -8'sd127; lut[62] = -8'sd127; lut[63] = -8'sd127;
        lut[64] = -8'sd127; lut[65] = -8'sd127; lut[66] = -8'sd127; lut[67] = -8'sd127; lut[68] = -8'sd127; lut[69] = -8'sd127; lut[70] = -8'sd127; lut[71] = -8'sd127;
        lut[72] = -8'sd127; lut[73] = -8'sd127; lut[74] = -8'sd127; lut[75] = -8'sd127; lut[76] = -8'sd127; lut[77] = -8'sd127; lut[78] = -8'sd127; lut[79] = -8'sd127;
        lut[80] = -8'sd127; lut[81] = -8'sd127; lut[82] = -8'sd127; lut[83] = -8'sd127; lut[84] = -8'sd127; lut[85] = -8'sd127; lut[86] = -8'sd127; lut[87] = -8'sd127;
        lut[88] = -8'sd127; lut[89] = -8'sd127; lut[90] = -8'sd127; lut[91] = -8'sd127; lut[92] = -8'sd127; lut[93] = -8'sd127; lut[94] = -8'sd127; lut[95] = -8'sd127;
        lut[96] = -8'sd127; lut[97] = -8'sd127; lut[98] = -8'sd127; lut[99] = -8'sd127; lut[100] = -8'sd127; lut[101] = -8'sd127; lut[102] = -8'sd127; lut[103] = -8'sd127;
        lut[104] = -8'sd127; lut[105] = -8'sd127; lut[106] = -8'sd127; lut[107] = -8'sd127; lut[108] = -8'sd127; lut[109] = -8'sd127; lut[110] = -8'sd127; lut[111] = -8'sd127;
        lut[112] = -8'sd127; lut[113] = -8'sd127; lut[114] = -8'sd127; lut[115] = -8'sd127; lut[116] = -8'sd127; lut[117] = -8'sd127; lut[118] = -8'sd127; lut[119] = -8'sd127;
        lut[120] = -8'sd127; lut[121] = -8'sd127; lut[122] = -8'sd127; lut[123] = -8'sd127; lut[124] = -8'sd127; lut[125] = -8'sd127; lut[126] = -8'sd127; lut[127] = -8'sd127;
        lut[128] = -8'sd127; lut[129] = -8'sd127; lut[130] = -8'sd127; lut[131] = -8'sd127; lut[132] = -8'sd127; lut[133] = -8'sd127; lut[134] = -8'sd127; lut[135] = -8'sd127;
        lut[136] = -8'sd127; lut[137] = -8'sd127; lut[138] = -8'sd127; lut[139] = -8'sd127; lut[140] = -8'sd127; lut[141] = -8'sd127; lut[142] = -8'sd127; lut[143] = -8'sd127;
        lut[144] = -8'sd127; lut[145] = -8'sd127; lut[146] = -8'sd127; lut[147] = -8'sd127; lut[148] = -8'sd127; lut[149] = -8'sd127; lut[150] = -8'sd127; lut[151] = -8'sd127;
        lut[152] = -8'sd127; lut[153] = -8'sd127; lut[154] = -8'sd127; lut[155] = -8'sd127; lut[156] = -8'sd127; lut[157] = -8'sd127; lut[158] = -8'sd127; lut[159] = -8'sd127;
        lut[160] = -8'sd127; lut[161] = -8'sd127; lut[162] = -8'sd127; lut[163] = -8'sd127; lut[164] = -8'sd127; lut[165] = -8'sd127; lut[166] = -8'sd127; lut[167] = -8'sd127;
        lut[168] = -8'sd127; lut[169] = -8'sd127; lut[170] = -8'sd127; lut[171] = -8'sd127; lut[172] = -8'sd127; lut[173] = -8'sd127; lut[174] = -8'sd126; lut[175] = -8'sd126;
        lut[176] = -8'sd126; lut[177] = -8'sd126; lut[178] = -8'sd126; lut[179] = -8'sd126; lut[180] = -8'sd126; lut[181] = -8'sd126; lut[182] = -8'sd126; lut[183] = -8'sd125;
        lut[184] = -8'sd125; lut[185] = -8'sd125; lut[186] = -8'sd125; lut[187] = -8'sd125; lut[188] = -8'sd124; lut[189] = -8'sd124; lut[190] = -8'sd124; lut[191] = -8'sd124;
        lut[192] = -8'sd123; lut[193] = -8'sd123; lut[194] = -8'sd123; lut[195] = -8'sd122; lut[196] = -8'sd122; lut[197] = -8'sd122; lut[198] = -8'sd121; lut[199] = -8'sd121;
        lut[200] = -8'sd120; lut[201] = -8'sd120; lut[202] = -8'sd120; lut[203] = -8'sd119; lut[204] = -8'sd118; lut[205] = -8'sd118; lut[206] = -8'sd117; lut[207] = -8'sd117;
        lut[208] = -8'sd116; lut[209] = -8'sd115; lut[210] = -8'sd114; lut[211] = -8'sd113; lut[212] = -8'sd113; lut[213] = -8'sd112; lut[214] = -8'sd111; lut[215] = -8'sd110;
        lut[216] = -8'sd109; lut[217] = -8'sd107; lut[218] = -8'sd106; lut[219] = -8'sd105; lut[220] = -8'sd104; lut[221] = -8'sd102; lut[222] = -8'sd101; lut[223] = -8'sd99;
        lut[224] = -8'sd97; lut[225] = -8'sd96; lut[226] = -8'sd94; lut[227] = -8'sd92; lut[228] = -8'sd90; lut[229] = -8'sd88; lut[230] = -8'sd86; lut[231] = -8'sd84;
        lut[232] = -8'sd81; lut[233] = -8'sd79; lut[234] = -8'sd76; lut[235] = -8'sd74; lut[236] = -8'sd71; lut[237] = -8'sd68; lut[238] = -8'sd65; lut[239] = -8'sd62;
        lut[240] = -8'sd59; lut[241] = -8'sd56; lut[242] = -8'sd53; lut[243] = -8'sd49; lut[244] = -8'sd46; lut[245] = -8'sd42; lut[246] = -8'sd39; lut[247] = -8'sd35;
        lut[248] = -8'sd31; lut[249] = -8'sd28; lut[250] = -8'sd24; lut[251] = -8'sd20; lut[252] = -8'sd16; lut[253] = -8'sd12; lut[254] = -8'sd8; lut[255] = -8'sd4;
        lut[256] = 8'sd0; lut[257] = 8'sd4; lut[258] = 8'sd8; lut[259] = 8'sd12; lut[260] = 8'sd16; lut[261] = 8'sd20; lut[262] = 8'sd24; lut[263] = 8'sd28;
        lut[264] = 8'sd31; lut[265] = 8'sd35; lut[266] = 8'sd39; lut[267] = 8'sd42; lut[268] = 8'sd46; lut[269] = 8'sd49; lut[270] = 8'sd53; lut[271] = 8'sd56;
        lut[272] = 8'sd59; lut[273] = 8'sd62; lut[274] = 8'sd65; lut[275] = 8'sd68; lut[276] = 8'sd71; lut[277] = 8'sd74; lut[278] = 8'sd76; lut[279] = 8'sd79;
        lut[280] = 8'sd81; lut[281] = 8'sd84; lut[282] = 8'sd86; lut[283] = 8'sd88; lut[284] = 8'sd90; lut[285] = 8'sd92; lut[286] = 8'sd94; lut[287] = 8'sd96;
        lut[288] = 8'sd97; lut[289] = 8'sd99; lut[290] = 8'sd101; lut[291] = 8'sd102; lut[292] = 8'sd104; lut[293] = 8'sd105; lut[294] = 8'sd106; lut[295] = 8'sd107;
        lut[296] = 8'sd109; lut[297] = 8'sd110; lut[298] = 8'sd111; lut[299] = 8'sd112; lut[300] = 8'sd113; lut[301] = 8'sd113; lut[302] = 8'sd114; lut[303] = 8'sd115;
        lut[304] = 8'sd116; lut[305] = 8'sd117; lut[306] = 8'sd117; lut[307] = 8'sd118; lut[308] = 8'sd118; lut[309] = 8'sd119; lut[310] = 8'sd120; lut[311] = 8'sd120;
        lut[312] = 8'sd120; lut[313] = 8'sd121; lut[314] = 8'sd121; lut[315] = 8'sd122; lut[316] = 8'sd122; lut[317] = 8'sd122; lut[318] = 8'sd123; lut[319] = 8'sd123;
        lut[320] = 8'sd123; lut[321] = 8'sd124; lut[322] = 8'sd124; lut[323] = 8'sd124; lut[324] = 8'sd124; lut[325] = 8'sd125; lut[326] = 8'sd125; lut[327] = 8'sd125;
        lut[328] = 8'sd125; lut[329] = 8'sd125; lut[330] = 8'sd126; lut[331] = 8'sd126; lut[332] = 8'sd126; lut[333] = 8'sd126; lut[334] = 8'sd126; lut[335] = 8'sd126;
        lut[336] = 8'sd126; lut[337] = 8'sd126; lut[338] = 8'sd126; lut[339] = 8'sd127; lut[340] = 8'sd127; lut[341] = 8'sd127; lut[342] = 8'sd127; lut[343] = 8'sd127;
        lut[344] = 8'sd127; lut[345] = 8'sd127; lut[346] = 8'sd127; lut[347] = 8'sd127; lut[348] = 8'sd127; lut[349] = 8'sd127; lut[350] = 8'sd127; lut[351] = 8'sd127;
        lut[352] = 8'sd127; lut[353] = 8'sd127; lut[354] = 8'sd127; lut[355] = 8'sd127; lut[356] = 8'sd127; lut[357] = 8'sd127; lut[358] = 8'sd127; lut[359] = 8'sd127;
        lut[360] = 8'sd127; lut[361] = 8'sd127; lut[362] = 8'sd127; lut[363] = 8'sd127; lut[364] = 8'sd127; lut[365] = 8'sd127; lut[366] = 8'sd127; lut[367] = 8'sd127;
        lut[368] = 8'sd127; lut[369] = 8'sd127; lut[370] = 8'sd127; lut[371] = 8'sd127; lut[372] = 8'sd127; lut[373] = 8'sd127; lut[374] = 8'sd127; lut[375] = 8'sd127;
        lut[376] = 8'sd127; lut[377] = 8'sd127; lut[378] = 8'sd127; lut[379] = 8'sd127; lut[380] = 8'sd127; lut[381] = 8'sd127; lut[382] = 8'sd127; lut[383] = 8'sd127;
        lut[384] = 8'sd127; lut[385] = 8'sd127; lut[386] = 8'sd127; lut[387] = 8'sd127; lut[388] = 8'sd127; lut[389] = 8'sd127; lut[390] = 8'sd127; lut[391] = 8'sd127;
        lut[392] = 8'sd127; lut[393] = 8'sd127; lut[394] = 8'sd127; lut[395] = 8'sd127; lut[396] = 8'sd127; lut[397] = 8'sd127; lut[398] = 8'sd127; lut[399] = 8'sd127;
        lut[400] = 8'sd127; lut[401] = 8'sd127; lut[402] = 8'sd127; lut[403] = 8'sd127; lut[404] = 8'sd127; lut[405] = 8'sd127; lut[406] = 8'sd127; lut[407] = 8'sd127;
        lut[408] = 8'sd127; lut[409] = 8'sd127; lut[410] = 8'sd127; lut[411] = 8'sd127; lut[412] = 8'sd127; lut[413] = 8'sd127; lut[414] = 8'sd127; lut[415] = 8'sd127;
        lut[416] = 8'sd127; lut[417] = 8'sd127; lut[418] = 8'sd127; lut[419] = 8'sd127; lut[420] = 8'sd127; lut[421] = 8'sd127; lut[422] = 8'sd127; lut[423] = 8'sd127;
        lut[424] = 8'sd127; lut[425] = 8'sd127; lut[426] = 8'sd127; lut[427] = 8'sd127; lut[428] = 8'sd127; lut[429] = 8'sd127; lut[430] = 8'sd127; lut[431] = 8'sd127;
        lut[432] = 8'sd127; lut[433] = 8'sd127; lut[434] = 8'sd127; lut[435] = 8'sd127; lut[436] = 8'sd127; lut[437] = 8'sd127; lut[438] = 8'sd127; lut[439] = 8'sd127;
        lut[440] = 8'sd127; lut[441] = 8'sd127; lut[442] = 8'sd127; lut[443] = 8'sd127; lut[444] = 8'sd127; lut[445] = 8'sd127; lut[446] = 8'sd127; lut[447] = 8'sd127;
        lut[448] = 8'sd127; lut[449] = 8'sd127; lut[450] = 8'sd127; lut[451] = 8'sd127; lut[452] = 8'sd127; lut[453] = 8'sd127; lut[454] = 8'sd127; lut[455] = 8'sd127;
        lut[456] = 8'sd127; lut[457] = 8'sd127; lut[458] = 8'sd127; lut[459] = 8'sd127; lut[460] = 8'sd127; lut[461] = 8'sd127; lut[462] = 8'sd127; lut[463] = 8'sd127;
        lut[464] = 8'sd127; lut[465] = 8'sd127; lut[466] = 8'sd127; lut[467] = 8'sd127; lut[468] = 8'sd127; lut[469] = 8'sd127; lut[470] = 8'sd127; lut[471] = 8'sd127;
        lut[472] = 8'sd127; lut[473] = 8'sd127; lut[474] = 8'sd127; lut[475] = 8'sd127; lut[476] = 8'sd127; lut[477] = 8'sd127; lut[478] = 8'sd127; lut[479] = 8'sd127;
        lut[480] = 8'sd127; lut[481] = 8'sd127; lut[482] = 8'sd127; lut[483] = 8'sd127; lut[484] = 8'sd127; lut[485] = 8'sd127; lut[486] = 8'sd127; lut[487] = 8'sd127;
        lut[488] = 8'sd127; lut[489] = 8'sd127; lut[490] = 8'sd127; lut[491] = 8'sd127; lut[492] = 8'sd127; lut[493] = 8'sd127; lut[494] = 8'sd127; lut[495] = 8'sd127;
        lut[496] = 8'sd127; lut[497] = 8'sd127; lut[498] = 8'sd127; lut[499] = 8'sd127; lut[500] = 8'sd127; lut[501] = 8'sd127; lut[502] = 8'sd127; lut[503] = 8'sd127;
        lut[504] = 8'sd127; lut[505] = 8'sd127; lut[506] = 8'sd127; lut[507] = 8'sd127; lut[508] = 8'sd127; lut[509] = 8'sd127; lut[510] = 8'sd127; lut[511] = 8'sd127;
    end

    always @(*) begin
        tanh_output = lut[index];
    end
endmodule
