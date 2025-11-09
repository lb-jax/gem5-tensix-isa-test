// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"

// Globals
uint32_t unp_cfg_context          = 0;
uint32_t pack_sync_tile_dst_ptr   = 0;
uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB_matmul.h"
#include "params.h"

void run_kernel()
{
    TT_SETDMAREG(0,
        0x88,
        0,
        0x8);
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_matmul.h"
#include "params.h"

void run_kernel()
{
    //uint32[0] = 0xf
    TT_SETDMAREG(0,
        0xf,
        0,
        0x0);

    //uint32[1] = 0xff
    TT_SETDMAREG(0,
        0xff,
        0,
        0x2);

    //uint32[2] = 0xfff
    TT_SETDMAREG(0,
        0xfff,
        0,
        0x4);

    //uint32[3] = 0xff
    TT_SETDMAREG(0,
        0xff,
        0,
        0x6);

    //uint32[4] = 0xca3d
    TT_SETDMAREG(0,
        0xca3d,
        0,
        0x8);

    // =======================================================================
    // ALU :ADDDMAREG,SUBDMAREG,MULDMAREG,CMPDMAREG,BITWOPDMAREG
    // =======================================================================
    // ALU :ADDDMAREG,SUBDMAREG,MULDMAREG,CMPDMAREG,BITWOPDMAREG
    //uint32[16] = 0xf + 0xff = 0x10e
    TT_ADDDMAREG(0,
        0x10,
        0x0,
        0x1);

    //uint32[17] = 0xfff - 0xff = 0xf00
    TT_SUBDMAREG(0,
        0x11,
        0x1,
        0x2);

    //uint32[18] = 0xff * 0xfff = 0xfef01
    TT_MULDMAREG(0,
        0x12,
        0x2,
        0x3);

    // uint32[19] = (0xfff > 0xff)? = 1
    TT_CMPDMAREG(0,
        0x0,
        0x13,
        0x1,
        0x2);

    // uint32[20] = (0xff > 0xfff)? = 0
    TT_CMPDMAREG(0,
        0x0,
        0x14,
        0x2,
        0x1);

    // uint32[21] = (0xfff < 0xff)? = 0
    TT_CMPDMAREG(0,
        0x1,
        0x15,
        0x1,
        0x2);

    // uint32[22] = (0xff < 0xfff)? = 1
    TT_CMPDMAREG(0,
        0x1,
        0x16,
        0x2,
        0x1);

    // uint32[23] = (0xff == 0xfff)? = 0
    TT_CMPDMAREG(0,
        0x2,
        0x17,
        0x2,
        0x1);

    // uint32[24] = (0xff == 0xff)? = 1
    TT_CMPDMAREG(0,
        0x2,
        0x18,
        0x3,
        0x1);

    // uint32[25] = 0xfff & 0xca3d = 0xa3d
    TT_BITWOPDMAREG(0,
        0x0,
        0x19,
        0x4,
        0x2);

    // uint32[32] = 0xfff | 0xca3d = 0xcfff
    TT_BITWOPDMAREG(0,
        0x1,
        0x20,
        0x4,
        0x2);

    // uint32[33] = 0xfff ^ 0xca3d = 0xc5c2
    TT_BITWOPDMAREG(0,
        0x2,
        0x21,
        0x4,
        0x2);
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel()
{
    TT_SETDMAREG(0,
        0x88,
        0,
        0x8);
}

#endif
