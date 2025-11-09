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
// 测试1：ATINCGETPTR 一次成功退出（push操作）
// 
// 场景：FIFO未满，执行push操作
// - 初始化FIFO控制字：Rd=0x0, Wr=0x2（FIFO有2个元素）
// - 容量为8（IntWidth对应），所以FIFO未满
// - 执行push操作：增加写指针，返回原始写指针值
// =======================================================================
void test_atincgetptr_push_success()
{
    // ===== 初始化阶段 =====
    
    // 初始化FIFO控制字到GPR
    // 结构: Rd在低32位，Wr在次32位
    // 我们要设置：Rd=0x00000000, Wr=0x00000002
    
    // GPR[0] = Rd = 0x00000000
    TT_SETDMAREG(0, 0x0000, 0, 0x0);
    TT_SETDMAREG(0, 0x0000, 0, 0x1);
    // 当前: GPR[0] = 0x00000000
    
    // GPR[1] = Wr = 0x00000002
    TT_SETDMAREG(0, 0x0002, 0, 0x2);
    TT_SETDMAREG(0, 0x0000, 0, 0x3);
    // 当前: GPR[1] = 0x00000002
    
    // 设置基地址寄存器
    // GPR[8] = 0x0，基地址 = 0x0 * 16 = 0x0000
    TT_SETDMAREG(0, 0x0, 0, 0x8);
    // 当前: GPR[8] = 0x0
    
    // 设置偏移量 (用于STOREIND)
    // GPR[16] 低16位 = 0x0000
    TT_SETDMAREG(0, 0x0, 0, 0x10);
    // 当前: GPR[16] = 0x0000, 偏移量 = 0x0000
    
    // ===== STOREIND: 将FIFO控制字写入L1 =====
    // 第一个32-bit字段存放Rd=0x0
    // 地址：L1[0x0 * 16 + 0 * 4] = L1[0x0000]
    TT_STOREIND(1, 0, 1, 32, 0, 0, 8);
    // 完成后: L1[0x0000] = 0x00000000 (Rd), GPR[16] = 0x0000
    
    WAIT_FOR_DMA();
    // 此时: L1[0x0000] = Rd = 0x00000000
    
    // 第二个32-bit字段存放Wr=0x2
    // 设置新的偏移量以存储Wr
    // 对于STOREIND的下一个字，我们需要改变偏移
    // 实际上，让我们改用不同的基地址或直接计算
    // 为简化，我们把Wr存放在L1[0x0004]（下一个32-bit位置）
    
    // 更新GPR[16]为新的偏移
    TT_SETDMAREG(0, 0x0001, 0, 0x10);
    // 当前: GPR[16] = 0x0001, 这使得L1地址 = 0 * 16 + 1 * 4 = 0x0004
    
    // 存储Wr到L1[0x0004]
    TT_STOREIND(1, 0, 1, 32, 0, 2, 8);
    // 完成后: L1[0x0004] = 0x00000002 (Wr), GPR[16] = 0x0001
    
    WAIT_FOR_DMA();
    // 此时: L1[0x0000] = Rd = 0x0
    //       L1[0x0004] = Wr = 0x2
    //       FIFO大小 = Wr - Rd = 2
    //       假设容量为8（2^3），FIFO未满
    
    // ===== ATINCGETPTR: 原子操作读取和增量 =====
    // TT_ATINCGETPTR(0, 
    //     NoIncr=0,        // 执行增量操作（push）
    //     IncrLog2=1,      // 增量 = 2^1 = 2
    //     IntWidth=3,      // 计数器宽度为3+1=4位（容量为2^4=16？）
    //     Ofs=1,           // 访问Ofs=1，即写指针（Wr）
    //     ResultReg=32,    // 结果保存到GPR[32]
    //     AddrReg=8)       // 基地址寄存器=GPR[8]=0x0
    //
    // 功能：
    //   L1Address = GPR[8] * 16 + 0 = 0x0000
    //   FIFOSize = L1[0x0004] - L1[0x0000] = 2 - 0 = 2
    //   FIFOCapacity = 2^(IntWidth) = 2^3 = 8（因为IntWidth=3，代表N+1位中的N=3，容量=2^N）
    //   FIFOEmpty = (FIFOSize == 0) = false
    //   
    //   由于Ofs=1（写指针），检查FIFO是否未满：
    //     FIFOSize < FIFOCapacity? 2 < 8? YES ✓
    //   
    //   条件满足，执行原子操作：
    //     OriginalValue = L1[0x0004] (Wr) = 0x00000002
    //     IncrementBy = 2^1 = 2
    //     Incremented = 0x00000002 + 2 = 0x00000004
    //     L1[0x0004] = Incremented = 0x00000004
    //   
    //   返回原始值到GPR[32]：
    //     GPR[32] = 0x00000002
    
    TT_ATINCGETPTR(0, 0, 1, 3, 0, 32, 8);
    // 
    // 指令特性：
    // - 同步指令，线程阻塞等待 >= 15 cycles
    // - 条件满足（FIFO未满），立即执行，然后继续
    // - 完成后：L1[0x0004] = 0x4, GPR[32] = 0x2
    
    // 线程到达这里，说明ATINCGETPTR已成功完成
    
    // ===== 验证：读取修改后的FIFO控制字 =====
    // 清除目标寄存器
    TT_SETDMAREG(0, 0x0, 0, 0x21);  // GPR[33] = 0x0
    TT_SETDMAREG(0, 0x0, 0, 0x22);  // GPR[34] = 0x0
    // 当前: GPR[33] = 0x0, GPR[34] = 0x0
    
    // 读取修改后的Wr值
    // 设置偏移回到Wr位置
    TT_SETDMAREG(0, 0x0001, 0, 0x10);
    // 当前: GPR[16] = 0x0001
    
    TT_LOADIND(1, 0, 1, 32, 0, 33, 8);
    // 完成后: GPR[33] = 0x00000004 (新的Wr值)
    
    WAIT_FOR_DMA();
    // 验证：GPR[33] 应该 = 0x00000004（Wr已增加2）
    //      GPR[32] 应该 = 0x00000002（原始Wr值）
}

// =======================================================================
// 测试2：ATINCGETPTR 无限阻塞（pop操作失败）
//
// 场景：FIFO为空，执行pop操作
// - 初始化FIFO控制字：Rd=0x5, Wr=0x5（FIFO为空）
// - 执行pop操作：需要FIFO非空，但实际为空
// - 线程永远阻塞，不断retry
// =======================================================================
void test_atincgetptr_pop_infinite_block()
{
    // ===== 初始化阶段 =====
    
    // 初始化FIFO控制字到GPR
    // 我们要设置：Rd=0x00000005, Wr=0x00000005（FIFO为空）
    
    // GPR[3] = Rd = 0x00000005
    TT_SETDMAREG(0, 0x0005, 0, 0x6);
    TT_SETDMAREG(0, 0x0000, 0, 0x7);
    // 当前: GPR[3] = 0x00000005
    
    // GPR[4] = Wr = 0x00000005
    TT_SETDMAREG(0, 0x0005, 0, 0x8);
    TT_SETDMAREG(0, 0x0000, 0, 0x9);
    // 当前: GPR[4] = 0x00000005
    
    // 设置基地址寄存器
    // GPR[9] = 0x1，基地址 = 0x1 * 16 = 0x0010
    TT_SETDMAREG(0, 0x1, 0, 0x9);
    // 当前: GPR[9] = 0x1
    
    // 设置偏移量 (用于STOREIND)
    // GPR[17] 低16位 = 0x0000
    TT_SETDMAREG(0, 0x0, 0, 0x11);
    // 当前: GPR[17] = 0x0000, 偏移量 = 0x0000
    
    // ===== STOREIND: 将FIFO控制字写入L1 =====
    // 第一个32-bit字段存放Rd=0x5
    // 地址：L1[0x1 * 16 + 0 * 4] = L1[0x0010]
    TT_STOREIND(1, 0, 1, 32, 0, 3, 9);
    // 完成后: L1[0x0010] = 0x00000005 (Rd), GPR[17] = 0x0000
    
    WAIT_FOR_DMA();
    // 此时: L1[0x0010] = Rd = 0x00000005
    
    // 第二个32-bit字段存放Wr=0x5
    // 更新GPR[17]为新的偏移
    TT_SETDMAREG(0, 0x0001, 0, 0x11);
    // 当前: GPR[17] = 0x0001, 这使得L1地址 = 1 * 16 + 1 * 4 = 0x0014
    
    // 存储Wr到L1[0x0014]
    TT_STOREIND(1, 0, 1, 32, 0, 4, 9);
    // 完成后: L1[0x0014] = 0x00000005 (Wr), GPR[17] = 0x0001
    
    WAIT_FOR_DMA();
    // 此时: L1[0x0010] = Rd = 0x5
    //       L1[0x0014] = Wr = 0x5
    //       FIFO大小 = Wr - Rd = 0（FIFO为空）
    
    // ===== ATINCGETPTR: 原子操作读取和增量（pop失败，无限阻塞） =====
    // TT_ATINCGETPTR(0, 
    //     NoIncr=0,        // 执行增量操作（pop）
    //     IncrLog2=2,      // 增量 = 2^2 = 4
    //     IntWidth=3,      // 计数器宽度=4位（容量2^3=8）
    //     Ofs=0,           // 访问Ofs=0，即读指针（Rd）
    //     ResultReg=40,    // 结果保存到GPR[40]
    //     AddrReg=9)       // 基地址寄存器=GPR[9]=0x1
    //
    // 功能：
    //   L1Address = GPR[9] * 16 + 0 = 0x0010
    //   FIFOSize = L1[0x0014] - L1[0x0010] = 5 - 5 = 0
    //   FIFOEmpty = (FIFOSize == 0) = true ✓
    //   
    //   由于Ofs=0（读指针），检查FIFO是否非空：
    //     FIFOEmpty? true → 条件**不满足** ✗
    //   
    //   条件失败，线程**阻塞并retry**：
    //     retry_loop:
    //       重新读取L1[0x0010]和L1[0x0014]
    //       检查是否 (Wr - Rd) != 0
    //       实际值仍为 5 - 5 = 0，条件仍不满足
    //       继续retry...（每次retry >= 15 cycles）
    //   
    //   由于L1值永不改变，条件永不满足
    //   线程**永远阻塞**在这里
    
    TT_ATINCGETPTR(0, 0, 2, 3, 0, 40, 9);
    // 
    // 指令特性：
    // - 同步指令，线程阻塞等待
    // - 条件不满足（FIFO为空），不断retry
    // - 线程永久卡在这里，无法执行下一条指令
    
    // **以下代码永远无法执行** ↓
    // 线程已永久阻塞在ATINCGETPTR指令处
    
    // 这行代码无法到达
    TT_SETDMAREG(0, 0xDEAD, 0, 0x28);
}

void run_kernel()
{
    // 执行ATINCGETPTR测试
    
    // 测试1：push成功退出
    // 这个函数会正常执行完毕，线程继续
    test_atincgetptr_push_success();
    
    // 测试2：pop无限阻塞
    // 这个函数会在ATINCGETPTR指令处永远阻塞
    // 线程无法继续执行后续代码
    test_atincgetptr_pop_infinite_block();
    
    // 这行代码如果能执行，说明Test2中的ATINCGETPTR某种程度上完成了
    // 但实际上无法到达这里
}

#endif
