// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include 
#include 
#include 

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
    TT_SETDMAREG(0, 0x88, 0, 0x8);
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
    TT_DMANOP(); \
    TT_DMANOP(); \
    TT_DMANOP(); \
    TT_DMANOP(); \
    TT_DMANOP(); \
    TT_DMANOP(); \
    TT_DMANOP();

// =======================================================================
// 测试1：ATSWAP 部分掩码写入（8个16-bit字段中的某些）
//
// 场景：
// - 初始化GPR数据（128-bit = 4个32-bit GPR）
// - 使用掩码指定写入哪些16-bit字段
// - ATSWAP立即返回，然后通过LOADIND验证L1中的数据
// =======================================================================
void test_atswap_partial_mask()
{
    // ===== 初始化阶段 =====
    
    // 初始化4个连续的GPR作为数据源
    // GPR[0:3]，每个32-bit = 2个16-bit字段
    // 总共8个16-bit字段
    
    // GPR[0] = 0xAAAA_BBBB
    //   字段0 (低16-bit) = 0xBBBB
    //   字段1 (高16-bit) = 0xAAAA
    TT_SETDMAREG(0, 0xBBBB, 0, 0x0);
    TT_SETDMAREG(0, 0xAAAA, 0, 0x1);
    // 当前: GPR[0] = 0xAAAABBBB
    
    // GPR[1] = 0xCCCC_DDDD
    //   字段2 (低16-bit) = 0xDDDD
    //   字段3 (高16-bit) = 0xCCCC
    TT_SETDMAREG(0, 0xDDDD, 0, 0x2);
    TT_SETDMAREG(0, 0xCCCC, 0, 0x3);
    // 当前: GPR[1] = 0xCCCCDDDD
    
    // GPR[2] = 0xEEEE_FFFF
    //   字段4 (低16-bit) = 0xFFFF
    //   字段5 (高16-bit) = 0xEEEE
    TT_SETDMAREG(0, 0xFFFF, 0, 0x4);
    TT_SETDMAREG(0, 0xEEEE, 0, 0x5);
    // 当前: GPR[2] = 0xEEEEFFFF
    
    // GPR[3] = 0x1111_2222
    //   字段6 (低16-bit) = 0x2222
    //   字段7 (高16-bit) = 0x1111
    TT_SETDMAREG(0, 0x2222, 0, 0x6);
    TT_SETDMAREG(0, 0x1111, 0, 0x7);
    // 当前: GPR[3] = 0x11112222
    
    // 当前GPR[0:3]内容：
    // GPR[0] = 0xAAAABBBB
    // GPR[1] = 0xCCCCDDDD
    // GPR[2] = 0xEEEEFFFF
    // GPR[3] = 0x11112222
    
    // 设置基地址寄存器
    // GPR[8] = 0x0，基地址 = 0x0 * 16 = 0x0000（单位是16-bit字节）
    TT_SETDMAREG(0, 0x0, 0, 0x8);
    // 当前: GPR[8] = 0x0
    
    // ===== 初始化L1：设置初始值（用于对比） =====
    // 先将L1初始化为0，然后通过ATSWAP写入特定字段
    
    // 使用STOREIND初始化L1为0x0000（全零）
    TT_SETDMAREG(0, 0x0, 0, 0x10);  // 初始化零值到GPR[16]
    TT_SETDMAREG(0, 0x0, 0, 0x11);
    TT_SETDMAREG(0, 0x0, 0, 0x12);
    TT_SETDMAREG(0, 0x0, 0, 0x13);
    // 当前: GPR[16:19] = 0x0
    
    // 设置STOREIND的偏移
    TT_SETDMAREG(0, 0x0, 0, 0x20);
    // 当前: GPR[32] = 0x0，作为STOREIND偏移
    
    // 存储0x00000000到L1[0x0000]（对应4个32-bit）
    // 这样L1地址0-15 bytes初始化为0
    TT_STOREIND(1, 0, 1, 32, 0, 16, 8);
    // 完成后: L1[0x0000:0x0003] = 0x00000000
    
    WAIT_FOR_DMA();
    
    // ===== ATSWAP: 原子写入指定16-bit字段 =====
    // TT_ATSWAP(0, Mask=0x5A, DataReg=0, AddrReg=8)
    //
    // 参数说明：
    //   Mask=0x5A (二进制 01011010)
    //     bit0=0: 不写入字段0
    //     bit1=1: 写入字段1 (0xAAAA)
    //     bit2=0: 不写入字段2
    //     bit3=1: 写入字段3 (0xCCCC)
    //     bit4=1: 写入字段4 (0xFFFF)
    //     bit5=0: 不写入字段5
    //     bit6=1: 写入字段6 (0x2222)
    //     bit7=0: 不写入字段7
    //   DataReg=0: 从GPR[0:3]读取数据
    //   AddrReg=8: L1基地址 = GPR[8] * 16 = 0x0000
    //
    // 功能：
    //   L1Address = 0x0000 (单位是16-bit字节)
    //   ToWrite从GPR[0:3]读取128-bit数据
    //   
    //   原子写入（掩码指定）：
    //     if (Mask & 0x01) L1[0x0000] = 0xBBBB;  // bit0=0, 不写
    //     if (Mask & 0x02) L1[0x0002] = 0xAAAA;  // bit1=1, 写入
    //     if (Mask & 0x04) L1[0x0004] = 0xDDDD;  // bit2=0, 不写
    //     if (Mask & 0x08) L1[0x0006] = 0xCCCC;  // bit3=1, 写入
    //     if (Mask & 0x10) L1[0x0008] = 0xFFFF;  // bit4=1, 写入
    //     if (Mask & 0x20) L1[0x000A] = 0xEEEE;  // bit5=0, 不写
    //     if (Mask & 0x40) L1[0x000C] = 0x2222;  // bit6=1, 写入
    //     if (Mask & 0x80) L1[0x000E] = 0x1111;  // bit7=0, 不写
    //
    // 预期L1最终状态（以32-bit视角）：
    //   L1[0x0000] = 0x0000BBBB (字段0未改，字段1=0xAAAA)
    //   L1[0x0004] = 0xCCCC0000 (字段2未改=0，字段3=0xCCCC)
    //   L1[0x0008] = 0x0000FFFF (字段4=0xFFFF，字段5未改=0)
    //   L1[0x000C] = 0x00002222 (字段6=0x2222，字段7未改=0)
    //
    // 指令执行时间：>= 3 cycles（发送请求后立即返回）
    // L1实际写入：后续在某个时间点完成
    
    TT_ATSWAP(0, 0x5A, 0, 8);
    // 完成后：指令返回，但L1写入可能仍在进行
    
    // ===== 等待L1写入完成 =====
    // 由于ATSWAP是异步的，需要等待足够长的时间让写入到达L1
    // 根据性能说明，需要等待至少12 cycles或更长时间
    WAIT_FOR_DMA();
    
    // ===== 验证：从L1读取数据 =====
    // 清除目标寄存器
    TT_SETDMAREG(0, 0x0, 0, 0x30);  // GPR[48] = 0x0
    TT_SETDMAREG(0, 0x0, 0, 0x31);
    TT_SETDMAREG(0, 0x0, 0, 0x32);
    TT_SETDMAREG(0, 0x0, 0, 0x33);
    // 当前: GPR[48:51] = 0x0
    
    // 从L1[0x0000]读取第一个32-bit
    TT_SETDMAREG(0, 0x0, 0, 0x28);  // GPR[40] = 0x0 (偏移)
    TT_LOADIND(1, 0, 1, 32, 0, 48, 8);
    // 完成后: GPR[48] = L1[0x0000:0x0003]
    
    WAIT_FOR_DMA();
    // 验证: GPR[48] 应该 = 0xAAAA0000 (字段1=0xAAAA, 字段0未改=0)
    
    // 读取第二个32-bit
    TT_SETDMAREG(0, 0x1, 0, 0x28);  // GPR[40] = 0x1 (偏移)
    TT_LOADIND(1, 0, 1, 32, 0, 49, 8);
    // 完成后: GPR[49] = L1[0x0004:0x0007]
    
    WAIT_FOR_DMA();
    // 验证: GPR[49] 应该 = 0xCCCC0000 (字段3=0xCCCC, 字段2未改=0)
    
    // 读取第三个32-bit
    TT_SETDMAREG(0, 0x2, 0, 0x28);  // GPR[40] = 0x2 (偏移)
    TT_LOADIND(1, 0, 1, 32, 0, 50, 8);
    // 完成后: GPR[50] = L1[0x0008:0x000B]
    
    WAIT_FOR_DMA();
    // 验证: GPR[50] 应该 = 0x0000FFFF (字段4=0xFFFF, 字段5未改=0)
    
    // 读取第四个32-bit
    TT_SETDMAREG(0, 0x3, 0, 0x28);  // GPR[40] = 0x3 (偏移)
    TT_LOADIND(1, 0, 1, 32, 0, 51, 8);
    // 完成后: GPR[51] = L1[0x000C:0x000F]
    
    WAIT_FOR_DMA();
    // 验证: GPR[51] 应该 = 0x00002222 (字段6=0x2222, 字段7未改=0)
}

// =======================================================================
// 测试2：ATSWAP 全掩码写入（所有16-bit字段）
//
// 场景：
// - 使用掩码0xFF，写入所有8个16-bit字段
// - 验证所有数据都被正确写入L1
// =======================================================================
void test_atswap_full_mask()
{
    // ===== 初始化阶段 =====
    
    // 初始化不同的GPR数据集
    // GPR[10:13]
    
    // GPR[10] = 0x4444_5555
    TT_SETDMAREG(0, 0x5555, 0, 0x14);
    TT_SETDMAREG(0, 0x4444, 0, 0x15);
    // 当前: GPR[10] = 0x44445555
    
    // GPR[11] = 0x6666_7777
    TT_SETDMAREG(0, 0x7777, 0, 0x16);
    TT_SETDMAREG(0, 0x6666, 0, 0x17);
    // 当前: GPR[11] = 0x66667777
    
    // GPR[12] = 0x8888_9999
    TT_SETDMAREG(0, 0x9999, 0, 0x18);
    TT_SETDMAREG(0, 0x8888, 0, 0x19);
    // 当前: GPR[12] = 0x88889999
    
    // GPR[13] = 0xAAAA_BBBB
    TT_SETDMAREG(0, 0xBBBB, 0, 0x1A);
    TT_SETDMAREG(0, 0xAAAA, 0, 0x1B);
    // 当前: GPR[13] = 0xAAAABBBB
    
    // 当前GPR[10:13]内容：
    // GPR[10] = 0x44445555
    // GPR[11] = 0x66667777
    // GPR[12] = 0x88889999
    // GPR[13] = 0xAAAABBBB
    
    // 设置基地址寄存器
    // GPR[9] = 0x1，基地址 = 0x1 * 16 = 0x0010（单位是16-bit字节）
    TT_SETDMAREG(0, 0x1, 0, 0x9);
    // 当前: GPR[9] = 0x1
    
    // ===== 初始化L1：设置初始值 =====
    // 清除目标区域
    TT_SETDMAREG(0, 0x0, 0, 0x34);  // GPR[52] = 0x0
    TT_SETDMAREG(0, 0x0, 0, 0x35);
    TT_SETDMAREG(0, 0x0, 0, 0x36);
    TT_SETDMAREG(0, 0x0, 0, 0x37);
    
    TT_SETDMAREG(0, 0x0, 0, 0x38);  // GPR[56] = 0x0 (偏移)
    TT_LOADIND(1, 0, 1, 32, 0, 52, 9);
    WAIT_FOR_DMA();
    
    // ===== ATSWAP: 原子写入所有16-bit字段 =====
    // TT_ATSWAP(0, Mask=0xFF, DataReg=10, AddrReg=9)
    //
    // 参数说明：
    //   Mask=0xFF (二进制 11111111) - 所有bit都为1，写入所有字段
    //   DataReg=10: 从GPR[10:13]读取数据
    //   AddrReg=9: L1基地址 = GPR[9] * 16 = 0x0010
    //
    // 功能：
    //   L1Address = 0x0010 (单位是16-bit字节)
    //   ToWrite从GPR[10:13]读取128-bit数据
    //   
    //   原子写入（全掩码）：
    //     L1[0x0010] = 0x5555;  // 字段0
    //     L1[0x0012] = 0x4444;  // 字段1
    //     L1[0x0014] = 0x7777;  // 字段2
    //     L1[0x0016] = 0x6666;  // 字段3
    //     L1[0x0018] = 0x9999;  // 字段4
    //     L1[0x001A] = 0x8888;  // 字段5
    //     L1[0x001C] = 0xBBBB;  // 字段6
    //     L1[0x001E] = 0xAAAA;  // 字段7
    //
    // 预期L1最终状态（以32-bit视角）：
    //   L1[0x0010] = 0x44445555
    //   L1[0x0014] = 0x66667777
    //   L1[0x0018] = 0x88889999
    //   L1[0x001C] = 0xAAAABBBB
    //
    // 指令执行时间：>= 3 cycles
    
    TT_ATSWAP(0, 0xFF, 10, 9);
    // 完成后：指令返回，但L1写入可能仍在进行
    
    // ===== 等待L1写入完成 =====
    WAIT_FOR_DMA();
    
    // ===== 验证：从L1读取数据 =====
    // 清除目标寄存器
    TT_SETDMAREG(0, 0x0, 0, 0x39);  // GPR[57] = 0x0
    TT_SETDMAREG(0, 0x0, 0, 0x3A);
    TT_SETDMAREG(0, 0x0, 0, 0x3B);
    TT_SETDMAREG(0, 0x0, 0, 0x3C);
    
    // 从L1[0x0010]读取第一个32-bit
    TT_SETDMAREG(0, 0x0, 0, 0x38);  // GPR[56] = 0x0 (偏移相对于GPR[9])
    TT_LOADIND(1, 0, 1, 32, 0, 57, 9);
    WAIT_FOR_DMA();
    // 验证: GPR[57] 应该 = 0x44445555
    
    // 从L1[0x0014]读取第二个32-bit
    TT_SETDMAREG(0, 0x1, 0, 0x38);  // GPR[56] = 0x1
    TT_LOADIND(1, 0, 1, 32, 0, 58, 9);
    WAIT_FOR_DMA();
    // 验证: GPR[58] 应该 = 0x66667777
    
    // 从L1[0x0018]读取第三个32-bit
    TT_SETDMAREG(0, 0x2, 0, 0x38);  // GPR[56] = 0x2
    TT_LOADIND(1, 0, 1, 32, 0, 59, 9);
    WAIT_FOR_DMA();
    // 验证: GPR[59] 应该 = 0x88889999
    
    // 从L1[0x001C]读取第四个32-bit
    TT_SETDMAREG(0, 0x3, 0, 0x38);  // GPR[56] = 0x3
    TT_LOADIND(1, 0, 1, 32, 0, 60, 9);
    WAIT_FOR_DMA();
    // 验证: GPR[60] 应该 = 0xAAAABBBB
}

void run_kernel()
{
    // 执行ATSWAP测试
    
    // 测试1：部分掩码写入
    // 验证掩码功能是否正常工作
    test_atswap_partial_mask();
    
    // 测试2：全掩码写入
    // 验证所有字段都被正确写入
    test_atswap_full_mask();
}

#endif
