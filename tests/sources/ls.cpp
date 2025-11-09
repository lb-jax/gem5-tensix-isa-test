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
    TT_SETDMAREG(0, 0x0, 0, 0x20);
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_matmul.h"
#include "params.h"

// =======================================================================
// 辅助宏定义
// =======================================================================
#define WAIT_FOR_DMA() \
    (void)TT_OP_DMANOP; \
    (void)TT_OP_DMANOP; \
    (void)TT_OP_DMANOP; \
    (void)TT_OP_DMANOP; \
    (void)TT_OP_DMANOP; \
    (void)TT_OP_DMANOP; \
    (void)TT_OP_DMANOP;

// =======================================================================
// 测试1：8-bit 写入和读取
// Size=3, DataReg=0, AddrReg=8, OffsetHalfReg=32
// =======================================================================
void test_8bit_storeind_loadind()
{
    // 初始化数据
    // GPR[0] = 0xDEADBEEF，我们只使用低8位 = 0xEF
    TT_SETDMAREG(0, 0xBEEF, 0, 0x0);
    TT_SETDMAREG(0, 0xDEAD, 0, 0x1);
    // 当前: GPR[0] = 0xDEADBEEF, 低8位 = 0xEF
    
    // 设置基地址寄存器
    // GPR[8] = 0x0，基地址 = 0x0 * 16 = 0x0000
    TT_SETDMAREG(0, 0x0, 0, 0x8);
    // 当前: GPR[8] = 0x0
    
    // 设置偏移量
    // GPR[16] 低16位 = 0x0000
    TT_SETDMAREG(0, 0x0, 0, 0x10);
    // 当前: GPR[16] = 0x0000, 偏移量 = 0x0000
    
    // 清除目标寄存器用于读取
    // GPR[32] 用于存储读取结果
    TT_SETDMAREG(0, 0x0, 0, 0x20);
    // 当前: GPR[32] = 0x0
    
    // ===== STOREIND: 写入8-bit数据到L1 =====
    // TT_STOREIND(1, 0, Size=3, OffsetHalfReg=32, OffsetIncrement=0, DataReg=0, AddrReg=8)
    // 操作: 将 GPR[0] 的低8位 (0xEF) 写入 L1[0x0000]
    TT_STOREIND(1, 0, 3, 32, 0, 0, 8);
    // 完成后: L1[0x0000] = 0xEF, GPR[16] = 0x0000 (OffsetIncrement=0，无增量)
    
    WAIT_FOR_DMA();
    // 等待7个cycle让写入完成
    // 此时: L1[0x0000] = 0xEF
    
    // ===== LOADIND: 从L1读取8-bit数据 =====
    // TT_LOADIND(1, 0, Size=3, OffsetHalfReg=32, OffsetIncrement=0, DataReg=32, AddrReg=8)
    // 操作: 从 L1[0x0000] 读8-bit数据到 GPR[32] 的低8位
    TT_LOADIND(3, 32, 0, 32, 8);
    // 完成后: GPR[32] 低8位 = 0xEF, GPR[16] = 0x0000
    
    WAIT_FOR_DMA();
    // 等待7个cycle让读取完成
    // 验证: GPR[32] 低8位应该 = 0xEF
}

// =======================================================================
// 测试2：16-bit 写入和读取
// Size=2, DataReg=1, AddrReg=8, OffsetHalfReg=32
// =======================================================================
void test_16bit_storeind_loadind()
{
    // 初始化数据
    // GPR[1] = 0xCAFEBABE，我们只使用低16位 = 0xBABE
    TT_SETDMAREG(0, 0xBABE, 0, 0x2);
    TT_SETDMAREG(0, 0xCAFE, 0, 0x3);
    // 当前: GPR[1] = 0xCAFEBABE, 低16位 = 0xBABE
    
    // 设置基地址寄存器
    // GPR[8] = 0x0，基地址 = 0x0000
    TT_SETDMAREG(0, 0x0, 0, 0x8);
    // 当前: GPR[8] = 0x0
    
    // 设置偏移量
    // GPR[16] 低16位 = 0x0010 (从前一个测试的偏移开始)
    TT_SETDMAREG(0, 0x10, 0, 0x10);
    // 当前: GPR[16] = 0x0010, 偏移量 = 0x0010
    
    // 清除目标寄存器用于读取
    // GPR[33] 用于存储读取结果
    TT_SETDMAREG(0, 0x0, 0, 0x21);
    // 当前: GPR[33] = 0x0
    
    // ===== STOREIND: 写入16-bit数据到L1 =====
    // TT_STOREIND(1, 0, Size=2, OffsetHalfReg=32, OffsetIncrement=0, DataReg=1, AddrReg=8)
    // 操作: 将 GPR[1] 的低16位 (0xBABE) 写入 L1[0x0000 + 0x10] = L1[0x0010]
    // 由于16-bit对齐，实际地址 = L1[0x0010]
    TT_STOREIND(1, 0, 2, 32, 0, 1, 8);
    // 完成后: L1[0x0010] = 0xBABE (16-bit), GPR[16] = 0x0010 (OffsetIncrement=0，无增量)
    
    WAIT_FOR_DMA();
    // 等待7个cycle让写入完成
    // 此时: L1[0x0010] = 0xBABE
    
    // ===== LOADIND: 从L1读取16-bit数据 =====
    // TT_LOADIND(1, 0, Size=2, OffsetHalfReg=32, OffsetIncrement=0, DataReg=33, AddrReg=8)
    // 操作: 从 L1[0x0010] 读16-bit数据到 GPR[33] 的低16位
    TT_LOADIND(2, 32, 0, 33, 8);
    // 完成后: GPR[33] 低16位 = 0xBABE, GPR[16] = 0x0010
    
    WAIT_FOR_DMA();
    // 等待7个cycle让读取完成
    // 验证: GPR[33] 低16位应该 = 0xBABE
}

// =======================================================================
// 测试3：32-bit 写入和读取
// Size=1, DataReg=2, AddrReg=8, OffsetHalfReg=32
// =======================================================================
void test_32bit_storeind_loadind()
{
    // 初始化数据
    // GPR[2] = 0x12345678
    TT_SETDMAREG(0, 0x5678, 0, 0x4);
    TT_SETDMAREG(0, 0x1234, 0, 0x5);
    // 当前: GPR[2] = 0x12345678
    
    // 设置基地址寄存器
    // GPR[8] = 0x0，基地址 = 0x0000
    TT_SETDMAREG(0, 0x0, 0, 0x8);
    // 当前: GPR[8] = 0x0
    
    // 设置偏移量
    // GPR[16] 低16位 = 0x0020
    TT_SETDMAREG(0, 0x20, 0, 0x10);
    // 当前: GPR[16] = 0x0020, 偏移量 = 0x0020
    
    // 清除目标寄存器用于读取
    // GPR[34] 用于存储读取结果
    TT_SETDMAREG(0, 0x0, 0, 0x22);
    // 当前: GPR[34] = 0x0
    
    // ===== STOREIND: 写入32-bit数据到L1 =====
    // TT_STOREIND(1, 0, Size=1, OffsetHalfReg=32, OffsetIncrement=0, DataReg=2, AddrReg=8)
    // 操作: 将 GPR[2] (0x12345678) 写入 L1[0x0000 + 0x20] = L1[0x0020]
    // 由于32-bit对齐，实际地址 = L1[0x0020]
    TT_STOREIND(1, 0, 1, 32, 0, 2, 8);
    // 完成后: L1[0x0020] = 0x12345678, GPR[16] = 0x0020
    
    WAIT_FOR_DMA();
    // 等待7个cycle让写入完成
    // 此时: L1[0x0020] = 0x12345678
    
    // ===== LOADIND: 从L1读取32-bit数据 =====
    // TT_LOADIND(1, 0, Size=1, OffsetHalfReg=32, OffsetIncrement=0, DataReg=34, AddrReg=8)
    // 操作: 从 L1[0x0020] 读32-bit数据到 GPR[34]
    TT_LOADIND(1, 32, 0, 34, 8);
    // 完成后: GPR[34] = 0x12345678, GPR[16] = 0x0020
    
    WAIT_FOR_DMA();
    // 等待7个cycle让读取完成
    // 验证: GPR[34] 应该 = 0x12345678
}

// =======================================================================
// 测试4：128-bit 写入和读取（4个连续GPR）
// Size=0, DataReg=3, AddrReg=8, OffsetHalfReg=32
// =======================================================================
void test_128bit_storeind_loadind()
{
    // 初始化4个连续的GPR数据 (GPR[3-6])
    // GPR[3] = 0xDEADBEEF
    TT_SETDMAREG(0, 0xBEEF, 0, 0x6);
    TT_SETDMAREG(0, 0xDEAD, 0, 0x7);
    // 当前: GPR[3] = 0xDEADBEEF
    
    // GPR[4] = 0xCAFEBABE
    TT_SETDMAREG(0, 0xBABE, 0, 0x8);
    TT_SETDMAREG(0, 0xCAFE, 0, 0x9);
    // 当前: GPR[4] = 0xCAFEBABE
    
    // GPR[5] = 0xABCDEF01
    TT_SETDMAREG(0, 0xEF01, 0, 0xA);
    TT_SETDMAREG(0, 0xABCD, 0, 0xB);
    // 当前: GPR[5] = 0xABCDEF01
    
    // GPR[6] = 0x13579BDF
    TT_SETDMAREG(0, 0x9BDF, 0, 0xC);
    TT_SETDMAREG(0, 0x1357, 0, 0xD);
    // 当前: GPR[6] = 0x13579BDF
    
    // 设置基地址寄存器
    // GPR[8] 保持 = 0x0，基地址 = 0x0000
    // (如果GPR[8]被修改了，需要重新设置)
    TT_SETDMAREG(0, 0x0, 0, 0x8);
    // 当前: GPR[8] = 0x0
    
    // 设置偏移量
    // GPR[16] 低16位 = 0x0030
    TT_SETDMAREG(0, 0x30, 0, 0x10);
    // 当前: GPR[16] = 0x0030, 偏移量 = 0x0030
    
    // 清除目标寄存器用于读取 (GPR[40-43])
    TT_SETDMAREG(0, 0x0, 0, 0x28);
    TT_SETDMAREG(0, 0x0, 0, 0x29);
    TT_SETDMAREG(0, 0x0, 0, 0x2A);
    TT_SETDMAREG(0, 0x0, 0, 0x2B);
    // 当前: GPR[40-43] = 0x0
    
    // ===== STOREIND: 写入128-bit数据到L1 =====
    // TT_STOREIND(1, 0, Size=0, OffsetHalfReg=32, OffsetIncrement=3, DataReg=3, AddrReg=8)
    // 操作: 将 GPR[3-6] (4个连续GPR) 写入 L1[0x0000 + 0x30] = L1[0x0030]
    // 128-bit对齐到16字节
    TT_STOREIND(1, 0, 0, 32, 3, 3, 8);
    // 完成后: L1[0x0030] = GPR[3], L1[0x0034] = GPR[4], L1[0x0038] = GPR[5], L1[0x003C] = GPR[6]
    //         GPR[16] = 0x0030 + 16 = 0x0040
    
    WAIT_FOR_DMA();
    // 等待7个cycle让写入完成
    // 此时: L1[0x0030-0x003F] 包含4个GPR数据
    
    // ===== LOADIND: 从L1读取128-bit数据 =====
    // TT_LOADIND(1, 0, Size=0, OffsetHalfReg=32, OffsetIncrement=0, DataReg=40, AddrReg=8)
    // 操作: 从 L1[0x0030] 读128-bit数据到 GPR[40-43]
    TT_LOADIND(0, 32, 0, 40, 8);
    // 完成后: GPR[40] = GPR[3] = 0xDEADBEEF
    //         GPR[41] = GPR[4] = 0xCAFEBABE
    //         GPR[42] = GPR[5] = 0xABCDEF01
    //         GPR[43] = GPR[6] = 0x13579BDF
    //         GPR[16] = 0x0030
    
    WAIT_FOR_DMA();
    // 等待7个cycle让读取完成
    // 验证: GPR[40-43] 应该分别 = GPR[3-6]
}

void run_kernel()
{
    // 执行所有4个测试
    
    // 测试1：8-bit
    test_8bit_storeind_loadind();
    
    // // 测试2：16-bit
    // test_16bit_storeind_loadind();
    
    // // 测试3：32-bit
    // test_32bit_storeind_loadind();
    
    // // 测试4：128-bit
    // test_128bit_storeind_loadind();
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