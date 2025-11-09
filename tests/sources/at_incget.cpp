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

// 更长的等待，确保异步操作完成
#define WAIT_FOR_ATINCGET() \
    TT_DMANOP(); \
    TT_DMANOP(); \
    TT_DMANOP(); \
    TT_DMANOP(); \
    TT_DMANOP(); \
    TT_DMANOP(); \
    TT_DMANOP(); \
    TT_DMANOP(); \
    TT_DMANOP(); \
    TT_DMANOP(); \
    TT_DMANOP(); \
    TT_DMANOP(); \
    TT_DMANOP(); \
    TT_DMANOP(); \
    TT_DMANOP(); \
    TT_DMANOP(); \
    TT_DMANOP(); \
    TT_DMANOP(); \
    TT_DMANOP(); \
    TT_DMANOP(); \
    TT_DMANOP(); \
    TT_DMANOP(); \
    TT_DMANOP(); \
    TT_DMANOP();

// =======================================================================
// 测试1：ATINCGET 异步原子递增（8-bit整数）
//
// 场景：
// - L1中存储一个8-bit计数器，初始值=0x05
// - GPR中设置递增值=0x03
// - 执行ATINCGET，返回原始值到GPR
// - 验证L1中的值已被递增到0x08
// =======================================================================
void test_atincget_8bit_increment()
{
    // ===== 初始化阶段 =====
    
    // 初始化L1中的计数器（8-bit）
    // L1地址 = GPR[8] * 16 + Ofs * 4 = 0x0000
    // 初始值 = 0x00000005（低8-bit = 0x05）
    
    // 设置基地址寄存器
    // GPR[8] = 0x0，基地址 = 0x0 * 16 = 0x0000
    TT_SETDMAREG(0, 0x0, 0, 0x8);
    // 当前: GPR[8] = 0x0
    
    // 初始化L1计数器为0x05
    TT_SETDMAREG(0, 0x0005, 0, 0x0);
    TT_SETDMAREG(0, 0x0000, 0, 0x1);
    // 当前: GPR[0] = 0x00000005
    
    // 设置偏移量为0（Ofs=0）
    TT_SETDMAREG(0, 0x0, 0, 0x10);
    // 当前: GPR[16] = 0x0
    
    // 通过STOREIND写入L1[0x0000] = 0x00000005
    TT_STOREIND(1, 0, 1, 32, 0, 0, 8);
    // 完成后: L1[0x0000] = 0x00000005
    
    WAIT_FOR_DMA();
    // 此时: L1[0x0000] = 0x00000005（低8-bit = 0x05）
    
    // ===== 准备ATINCGET操作 =====
    
    // 设置递增值到GPR[32]
    // 递增量 = 0x03
    TT_SETDMAREG(0, 0x0003, 0, 0x40);
    TT_SETDMAREG(0, 0x0000, 0, 0x41);
    // 当前: GPR[32] = 0x00000003（递增值）
    
    // ===== ATINCGET: 原子递增8-bit整数 =====
    // TT_ATINCGET(0, IntWidth=7, Ofs=0, InOutReg=32, AddrReg=8)
    //
    // 参数说明：
    //   IntWidth=7: 整数宽度 = 8位（实际宽度 = IntWidth + 1）
    //              IntMask = (2u << 7) - 1 = 0x100 - 1 = 0xFF
    //   Ofs=0: L1地址偏移 = 0 * 4 = 0
    //   InOutReg=32: 使用GPR[32]作为输入（递增值）和输出（原始值）
    //   AddrReg=8: 基地址 = GPR[8] * 16 = 0x0000
    //
    // L1地址计算：
    //   L1Address = GPR[8] * 16 + 0 * 4 = 0x0000 + 0x0 = 0x0000
    //
    // 指令执行流程：
    //   1. 捕获GPR[32] = 0x00000003（递增值）
    //   2. 发送原子请求到L1（后续异步执行）
    //   3. 立即返回（线程继续，约3 cycles占用Scalar Unit）
    //
    //   后续在L1（异步）：
    //     原子块：
    //       OriginalValue = L1[0x0000] = 0x00000005
    //       Incremented = 0x00000005 + 0x00000003 = 0x00000008
    //       IntMask = 0xFF
    //       L1[0x0000] = (0x08 & 0xFF) | (0x05 & ~0xFF) = 0x00000008
    //       返回 OriginalValue = 0x00000005
    //
    //   后续更晚（写回GPR）：
    //       GPR[32] = 0x00000005（原始值）
    //
    // 预期行为：
    //   - 指令立即返回，线程继续
    //   - L1[0x0000]最终 = 0x08
    //   - GPR[32]最终 = 0x05（原始值）
    
    TT_ATINCGET(0, 7, 0, 32, 8);
    // 完成后：指令返回，但异步操作仍在进行
    
    // ===== 等待异步操作完成 =====
    // 需要等待足够长的时间让：
    //   1. L1原子操作完成
    //   2. GPR写回完成
    WAIT_FOR_ATINCGET();
    
    // ===== 验证1：检查GPR[32]是否包含原始值 =====
    // 此时GPR[32]应该 = 0x00000005（L1[0x0000]的原始值）
    
    // 清除验证寄存器
    TT_SETDMAREG(0, 0x0, 0, 0x42);  // GPR[33] = 0x0
    TT_SETDMAREG(0, 0x0, 0, 0x43);
    // 当前: GPR[33] = 0x0
    
    // 从GPR[32]复制值到GPR[33]（用于观测）
    // 注：这里我们直接通过后续LOADIND来验证L1值
    
    // ===== 验证2：从L1读取递增后的值 =====
    // 清除目标寄存器
    TT_SETDMAREG(0, 0x0, 0, 0x44);  // GPR[34] = 0x0
    TT_SETDMAREG(0, 0x0, 0, 0x45);
    // 当前: GPR[34] = 0x0
    
    // 设置偏移为0
    TT_SETDMAREG(0, 0x0, 0, 0x10);
    // 当前: GPR[16] = 0x0
    
    // 从L1[0x0000]读取递增后的值
    TT_LOADIND(1, 0, 1, 32, 0, 34, 8);
    // 完成后: GPR[34] = L1[0x0000]
    
    WAIT_FOR_DMA();
    // 验证: GPR[34] 应该 = 0x00000008（递增后的值）
    //      GPR[32] 应该 = 0x00000005（原始值）
}

// =======================================================================
// 测试2：ATINCGET 带位掩码的16-bit递增
//
// 场景：
// - L1中存储一个16-bit计数器，初始值=0xAABB（低16-bit = 0xBB, 高16-bit = 0xAA）
// - GPR中设置递增值=0x0010
// - 执行ATINCGET（IntWidth=15，掩码=0xFFFF）
// - 验证低16-bit被递增，高16-bit保持不变
// =======================================================================
void test_atincget_16bit_masked_increment()
{
    // ===== 初始化阶段 =====
    
    // 设置基地址寄存器
    // GPR[9] = 0x1，基地址 = 0x1 * 16 = 0x0010
    TT_SETDMAREG(0, 0x1, 0, 0x9);
    // 当前: GPR[9] = 0x1
    
    // 初始化L1计数器
    // L1地址 = GPR[9] * 16 + Ofs * 4 = 0x0010 + 0x0 = 0x0010
    // 初始值 = 0xAAAABBBB
    //   低16-bit (masked) = 0xBBBB
    //   高16-bit (unmasked) = 0xAAAA
    
    TT_SETDMAREG(0, 0xBBBB, 0, 0x2);
    TT_SETDMAREG(0, 0xAAAA, 0, 0x3);
    // 当前: GPR[1] = 0xAAAABBBB
    
    // 设置偏移量为0
    TT_SETDMAREG(0, 0x0, 0, 0x11);
    // 当前: GPR[17] = 0x0
    
    // 通过STOREIND写入L1[0x0010] = 0xAAAABBBB
    TT_STOREIND(1, 0, 1, 32, 0, 1, 9);
    // 完成后: L1[0x0010] = 0xAAAABBBB
    
    WAIT_FOR_DMA();
    // 此时: L1[0x0010] = 0xAAAABBBB
    
    // ===== 准备ATINCGET操作 =====
    
    // 设置递增值到GPR[48]
    // 递增量 = 0x0010
    TT_SETDMAREG(0, 0x0010, 0, 0x60);
    TT_SETDMAREG(0, 0x0000, 0, 0x61);
    // 当前: GPR[48] = 0x00000010
    
    // ===== ATINCGET: 原子递增16-bit整数（带掩码） =====
    // TT_ATINCGET(0, IntWidth=15, Ofs=0, InOutReg=48, AddrReg=9)
    //
    // 参数说明：
    //   IntWidth=15: 整数宽度 = 16位（实际宽度 = IntWidth + 1 = 16）
    //               IntMask = (2u << 15) - 1 = 0x10000 - 1 = 0xFFFF
    //   Ofs=0: L1地址偏移 = 0 * 4 = 0
    //   InOutReg=48: 使用GPR[48]作为输入（递增值）和输出（原始值）
    //   AddrReg=9: 基地址 = GPR[9] * 16 = 0x0010
    //
    // L1地址计算：
    //   L1Address = GPR[9] * 16 + 0 * 4 = 0x0010 + 0x0 = 0x0010
    //
    // 指令执行流程：
    //   1. 捕获GPR[48] = 0x00000010（递增值）
    //   2. 发送原子请求到L1
    //   3. 立即返回
    //
    //   后续在L1（异步）：
    //     原子块：
    //       OriginalValue = L1[0x0010] = 0xAAAABBBB
    //       Incremented = 0xAAAABBBB + 0x00000010 = 0xAAAABBCB
    //       IntMask = 0xFFFF（只保留低16-bit）
    //       L1[0x0010] = (0xBBCB & 0xFFFF) | (0xAAAABBBB & ~0xFFFF)
    //                  = 0x0000BBCB | 0xAAAA0000
    //                  = 0xAAAABBCB
    //       返回 OriginalValue = 0xAAAABBBB
    //
    //   后续更晚（写回GPR）：
    //       GPR[48] = 0xAAAABBBB（原始值）
    //
    // 预期行为：
    //   - 指令立即返回
    //   - L1[0x0010]最终 = 0xAAAABBCB（低16-bit被递增）
    //   - GPR[48]最终 = 0xAAAABBBB（原始值）
    
    TT_ATINCGET(0, 15, 0, 48, 9);
    // 完成后：指令返回，但异步操作仍在进行
    
    // ===== 等待异步操作完成 =====
    WAIT_FOR_ATINCGET();
    
    // ===== 验证：从L1读取递增后的值 =====
    // 清除目标寄存器
    TT_SETDMAREG(0, 0x0, 0, 0x62);  // GPR[49] = 0x0
    TT_SETDMAREG(0, 0x0, 0, 0x63);
    // 当前: GPR[49] = 0x0
    
    // 从L1[0x0010]读取递增后的值
    TT_SETDMAREG(0, 0x0, 0, 0x11);
    TT_LOADIND(1, 0, 1, 32, 0, 49, 9);
    // 完成后: GPR[49] = L1[0x0010]
    
    WAIT_FOR_DMA();
    // 验证: GPR[49] 应该 = 0xAAAABBCB（低16-bit递增0x10）
    //      GPR[48] 应该 = 0xAAAABBBB（原始值）
}

// =======================================================================
// 测试3：ATINCGET 多次连续递增
//
// 场景：
// - L1中存储一个32-bit计数器，初始值=0x00000000
// - 执行多次ATINCGET，每次递增0x01
// - 验证计数器逐次递增
// =======================================================================
void test_atincget_multiple_increments()
{
    // ===== 初始化阶段 =====
    
    // 设置基地址寄存器
    // GPR[10] = 0x2，基地址 = 0x2 * 16 = 0x0020
    TT_SETDMAREG(0, 0x2, 0, 0xA);
    // 当前: GPR[10] = 0x2
    
    // 初始化L1计数器为0x00000000
    TT_SETDMAREG(0, 0x0000, 0, 0x4);
    TT_SETDMAREG(0, 0x0000, 0, 0x5);
    // 当前: GPR[2] = 0x00000000
    
    // 设置偏移量为0
    TT_SETDMAREG(0, 0x0, 0, 0x12);
    // 当前: GPR[18] = 0x0
    
    // 通过STOREIND写入L1[0x0020] = 0x00000000
    TT_STOREIND(1, 0, 1, 32, 0, 2, 10);
    // 完成后: L1[0x0020] = 0x00000000
    
    WAIT_FOR_DMA();
    
    // ===== 第1次递增：+1 =====
    
    // 设置递增值到GPR[56]
    TT_SETDMAREG(0, 0x0001, 0, 0x70);
    TT_SETDMAREG(0, 0x0000, 0, 0x71);
    // 当前: GPR[56] = 0x00000001
    
    // 执行第一次ATINCGET
    // TT_ATINCGET(0, IntWidth=31, Ofs=0, InOutReg=56, AddrReg=10)
    // 预期：返回0x00000000，L1更新为0x00000001
    TT_ATINCGET(0, 31, 0, 56, 10);
    
    WAIT_FOR_ATINCGET();
    // 此时：GPR[56] = 0x00000000（第一次的原始值）
    //      L1[0x0020] = 0x00000001
    
    // ===== 第2次递增：+2 =====
    
    // 设置新的递增值到GPR[57]
    TT_SETDMAREG(0, 0x0002, 0, 0x72);
    TT_SETDMAREG(0, 0x0000, 0, 0x73);
    // 当前: GPR[57] = 0x00000002
    
    // 执行第二次ATINCGET
    // TT_ATINCGET(0, IntWidth=31, Ofs=0, InOutReg=57, AddrReg=10)
    // 预期：返回0x00000001，L1更新为0x00000003
    TT_ATINCGET(0, 31, 0, 57, 10);
    
    WAIT_FOR_ATINCGET();
    // 此时：GPR[57] = 0x00000001（第二次的原始值）
    //      L1[0x0020] = 0x00000003
    
    // ===== 第3次递增：+3 =====
    
    // 设置新的递增值到GPR[58]
    TT_SETDMAREG(0, 0x0003, 0, 0x74);
    TT_SETDMAREG(0, 0x0000, 0, 0x75);
    // 当前: GPR[58] = 0x00000003
    
    // 执行第三次ATINCGET
    // TT_ATINCGET(0, IntWidth=31, Ofs=0, InOutReg=58, AddrReg=10)
    // 预期：返回0x00000003，L1更新为0x00000006
    TT_ATINCGET(0, 31, 0, 58, 10);
    
    WAIT_FOR_ATINCGET();
    // 此时：GPR[58] = 0x00000003（第三次的原始值）
    //      L1[0x0020] = 0x00000006
    
    // ===== 验证最终结果 =====
    // 清除目标寄存器
    TT_SETDMAREG(0, 0x0, 0, 0x76);  // GPR[59] = 0x0
    TT_SETDMAREG(0, 0x0, 0, 0x77);
    // 当前: GPR[59] = 0x0
    
    // 从L1[0x0020]读取最终的计数器值
    TT_SETDMAREG(0, 0x0, 0, 0x12);
    TT_LOADIND(1, 0, 1, 32, 0, 59, 10);
    // 完成后: GPR[59] = L1[0x0020]
    
    WAIT_FOR_DMA();
    // 验证: GPR[59] 应该 = 0x00000006（1+2+3）
    //      GPR[56] 应该 = 0x00000000（第一次的原始值）
    //      GPR[57] 应该 = 0x00000001（第二次的原始值）
    //      GPR[58] 应该 = 0x00000003（第三次的原始值）
}

void run_kernel()
{
    // 执行ATINCGET测试
    
    // 测试1：8-bit递增
    // 验证基本的原子递增和原始值返回
    test_atincget_8bit_increment();
    
    // 测试2：16-bit递增（带掩码）
    // 验证位宽掩码功能
    test_atincget_16bit_masked_increment();
    
    // 测试3：多次连续递增
    // 验证递增的正确累积
    test_atincget_multiple_increments();
}

#endif
