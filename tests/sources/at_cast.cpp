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
// 测试1：ATCAS 一次成功退出
// CmpVal匹配L1中的值，直接成功，线程继续执行下一条指令
// =======================================================================
void test_atcas_exit()
{
    // ===== 初始化阶段 =====
    
    // 初始化数据到GPR，用于STOREIND写入L1
    // 我们要写入的初始值 = 0x5 (4-bit值)
    TT_SETDMAREG(0, 0x0005, 0, 0x0);  // GPR[0] = 0x00000005
    // 当前: GPR[0] = 0x00000005 (低4位 = 0x5)
    
    // 设置基地址寄存器
    // GPR[8] = 0x0，基地址 = 0x0 * 16 = 0x0000
    TT_SETDMAREG(0, 0x0, 0, 0x8);
    // 当前: GPR[8] = 0x0
    
    // 设置偏移量 (用于STOREIND)
    // GPR[16] 低16位 = 0x0000
    TT_SETDMAREG(0, 0x0, 0, 0x10);
    // 当前: GPR[16] = 0x0000, 偏移量 = 0x0000
    
    // ===== STOREIND: 将初始值写入L1 =====
    // 使用Size=1 (32-bit)写入
    // 写 GPR[0]=0x00000005 到 L1[0*16 + 0*4] = L1[0x0000]
    TT_STOREIND(1, 0, 1, 32, 0, 0, 8);
    // 完成后: L1[0x0000] = 0x00000005, GPR[16] = 0x0000
    
    WAIT_FOR_DMA();
    // 此时: L1[0x0000] = 0x00000005
    
    // ===== ATCAS: 原子比较和设置（成功退出） =====
    // TT_ATCAS(0, SetVal=0x7, CmpVal=0x5, Ofs=0, 0, AddrReg=8)
    // 
    // 功能流程:
    //   L1Address = GPR[8] * 16 + 0 * 4 = 0x0000
    //   
    //   retry_loop:
    //     if (L1[0x0000] != 0x5) {
    //       goto retry_loop;  // 条件不满足，继续retry
    //     }
    //     atomic {
    //       L1[0x0000] = 0x7;  // 条件满足，原子性写入
    //     }
    //     return;  // 成功退出
    //
    // 当前L1[0x0000] = 0x5，条件满足
    // 预期: 立即设置为0x7，线程继续执行
    TT_ATCAS(0, 0x7, 0x5, 0, 0, 8);
    // 
    // 指令特性:
    // - 这是一个**同步指令**
    // - 不需要TT_DMANOP()等待
    // - 线程会**阻塞在这里**直到条件满足
    // - 一旦条件满足，立即执行，设置L1值，然后继续执行下一条指令
    // 
    // 执行时间: >= 15 cycles (但线程会阻塞等待，无法观测)
    // 完成后: L1[0x0000] = 0x00000007
    
    // 线程到达这里，说明ATCAS已成功完成
    
    // ===== 验证: 从L1读取结果 =====
    // 清除目标寄存器
    TT_SETDMAREG(0, 0x0, 0, 0x20);  // GPR[32] = 0x0
    // 当前: GPR[32] = 0x0
    
    // 从 L1[0x0000] 读取32-bit数据到 GPR[32]
    TT_STOREIND(1, 0, 1, 32, 0, 32, 8);
    // 完成后: GPR[32] = 0x00000007
    
    WAIT_FOR_DMA();
    // 验证: GPR[32] 应该 = 0x00000007 (确认ATCAS成功改写了L1)
}

// =======================================================================
// 测试2：ATCAS 无限阻塞
// CmpVal 不匹配 L1 中的值，线程永远阻塞，无法退出
// =======================================================================
void test_atcas_infinite_block()
{
    // ===== 初始化阶段 =====
    
    // 初始化数据到GPR，用于STOREIND写入L1
    // 我们要写入的初始值 = 0x3 (4-bit值)
    TT_SETDMAREG(0, 0x0003, 0, 0x1);  // GPR[1] = 0x00000003
    // 当前: GPR[1] = 0x00000003 (低4位 = 0x3)
    
    // 设置基地址寄存器 (使用不同的地址避免与Test1冲突)
    // GPR[9] = 0x1，基地址 = 0x1 * 16 = 0x0010
    TT_SETDMAREG(0, 0x1, 0, 0x9);
    // 当前: GPR[9] = 0x1
    
    // 设置偏移量 (用于STOREIND)
    // GPR[17] 低16位 = 0x0000
    TT_SETDMAREG(0, 0x0, 0, 0x11);
    // 当前: GPR[17] = 0x0000, 偏移量 = 0x0000
    
    // ===== STOREIND: 将初始值写入L1 =====
    // 使用Size=1 (32-bit)写入
    // 写 GPR[1]=0x00000003 到 L1[1*16 + 0*4] = L1[0x0010]
    TT_STOREIND(1, 0, 1, 32, 0, 1, 9);
    // 完成后: L1[0x0010] = 0x00000003, GPR[17] = 0x0000
    
    WAIT_FOR_DMA();
    // 此时: L1[0x0010] = 0x00000003
    
    // ===== ATCAS: 原子比较和设置（无限阻塞） =====
    // TT_ATCAS(0, SetVal=0xA, CmpVal=0x5, Ofs=0, 0, AddrReg=9)
    // 
    // 功能流程:
    //   L1Address = GPR[9] * 16 + 0 * 4 = 0x0010
    //   
    //   retry_loop:
    //     if (L1[0x0010] != 0x5) {
    //       goto retry_loop;  // 条件不满足，继续retry
    //     }
    //     atomic {
    //       L1[0x0010] = 0xA;  // 这行永远不会执行
    //     }
    //     return;  // 这行永远无法到达
    //
    // 当前L1[0x0010] = 0x3 ≠ 0x5，条件**不满足**
    // 预期: 线程**永远阻塞**在这里，不断retry，无法继续执行
    // 
    // Retry机制:
    // - ATCAS指令会反复发送请求到ThCon L1访问端口
    // - 每次retry间隔 >= 15 cycles
    // - 由于L1[0x0010]始终 = 0x3，条件永不满足
    // - 线程**永远卡在这里**，无法继续执行下一条指令
    TT_ATCAS(0, 0xA, 0x5, 0, 0, 9);
    // 
    // 指令特性:
    // - 这是一个**同步指令**，线程阻塞等待
    // - 不需要TT_DMANOP()，因为线程已经卡在这里
    // - Scalar Unit 会持续发送原子比较请求
    // - gem5 log中会看到大量重复的L1访问尝试
    
    // **以下代码永远无法执行** ↓
    // 线程已永久阻塞在ATCAS指令处
    
    // 这行代码无法到达
    TT_SETDMAREG(0, 0xDEAD, 0, 0x20);
}

void run_kernel()
{
    // 执行ATCAS测试
    
    // 测试1：成功退出
    // 这个函数会正常执行完毕，线程继续
    test_atcas_exit();
    
    // 测试2：无限阻塞
    // 这个函数会在ATCAS指令处永远阻塞
    // 线程无法继续执行后续代码
    test_atcas_infinite_block();
    
    // 这行代码如果能执行，说明Test2中的ATCAS某种程度上完成了
    // 但实际上无法到达这里
}

#endif
