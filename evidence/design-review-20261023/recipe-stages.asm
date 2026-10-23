
/usr/lib/habanalabs/libSynapse.so:     file format elf64-x86-64


Disassembly of section .text:

000000000171f100 <RecipeLauncher::getMappedMemoryInfo()@@Base>:
 171f100:	f3 0f 1e fa          	endbr64
 171f104:	55                   	push   %rbp
 171f105:	48 89 e5             	mov    %rsp,%rbp
 171f108:	41 57                	push   %r15
 171f10a:	41 56                	push   %r14
 171f10c:	4c 8d 77 30          	lea    0x30(%rdi),%r14
 171f110:	41 55                	push   %r13
 171f112:	41 54                	push   %r12
 171f114:	53                   	push   %rbx
 171f115:	48 89 fb             	mov    %rdi,%rbx
 171f118:	48 83 ec 38          	sub    $0x38,%rsp
 171f11c:	4c 8b 25 dd c1 44 02 	mov    0x244c1dd(%rip),%r12        # 3b6b300 <g_globalStat@@Base-0x25c560>
 171f123:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
 171f12a:	00 00 
 171f12c:	48 89 45 c8          	mov    %rax,-0x38(%rbp)
 171f130:	31 c0                	xor    %eax,%eax
 171f132:	48 8b 43 28          	mov    0x28(%rbx),%rax
 171f136:	48 8b 93 38 01 00 00 	mov    0x138(%rbx),%rdx
 171f13d:	4d 89 f0             	mov    %r14,%r8
 171f140:	48 8b 8b 40 01 00 00 	mov    0x140(%rbx),%rcx
 171f147:	48 8d b0 90 02 00 00 	lea    0x290(%rax),%rsi
 171f14e:	48 8b 83 00 01 00 00 	mov    0x100(%rbx),%rax
 171f155:	48 8b 78 20          	mov    0x20(%rax),%rdi
 171f159:	e8 52 60 fa fe       	call   6c51b0 <MappedMemMgr::getAddrForId(RecipeStaticInfoScal const&, EntryIds, MemorySectionsScal&)@plt>
 171f15e:	8b 43 68             	mov    0x68(%rbx),%eax
 171f161:	83 f8 02             	cmp    $0x2,%eax
 171f164:	74 5a                	je     171f1c0 <RecipeLauncher::getMappedMemoryInfo()@@Base+0xc0>
 171f166:	8b 93 98 00 00 00    	mov    0x98(%rbx),%edx
 171f16c:	83 fa 02             	cmp    $0x2,%edx
 171f16f:	74 4f                	je     171f1c0 <RecipeLauncher::getMappedMemoryInfo()@@Base+0xc0>
 171f171:	83 f8 01             	cmp    $0x1,%eax
 171f174:	0f 94 c0             	sete   %al
 171f177:	0f b6 c0             	movzbl %al,%eax
 171f17a:	48 89 c1             	mov    %rax,%rcx
 171f17d:	48 83 c9 02          	or     $0x2,%rcx
 171f181:	83 fa 01             	cmp    $0x1,%edx
 171f184:	48 0f 44 c1          	cmove  %rcx,%rax
 171f188:	45 31 ed             	xor    %r13d,%r13d
 171f18b:	41 80 7c 24 2c 00    	cmpb   $0x0,0x2c(%r12)
 171f191:	0f 85 41 01 00 00    	jne    171f2d8 <RecipeLauncher::getMappedMemoryInfo()@@Base+0x1d8>
 171f197:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
 171f19b:	64 48 2b 04 25 28 00 	sub    %fs:0x28,%rax
 171f1a2:	00 00 
 171f1a4:	0f 85 0d 02 00 00    	jne    171f3b7 <RecipeLauncher::getMappedMemoryInfo()@@Base+0x2b7>
 171f1aa:	48 83 c4 38          	add    $0x38,%rsp
 171f1ae:	44 89 e8             	mov    %r13d,%eax
 171f1b1:	5b                   	pop    %rbx
 171f1b2:	41 5c                	pop    %r12
 171f1b4:	41 5d                	pop    %r13
 171f1b6:	41 5e                	pop    %r14
 171f1b8:	41 5f                	pop    %r15
 171f1ba:	5d                   	pop    %rbp
 171f1bb:	c3                   	ret
 171f1bc:	0f 1f 40 00          	nopl   0x0(%rax)
 171f1c0:	45 31 ff             	xor    %r15d,%r15d
 171f1c3:	41 80 7c 24 2c 00    	cmpb   $0x0,0x2c(%r12)
 171f1c9:	0f 85 79 01 00 00    	jne    171f348 <RecipeLauncher::getMappedMemoryInfo()@@Base+0x248>
 171f1cf:	48 8b 7b 20          	mov    0x20(%rbx),%rdi
 171f1d3:	48 8b 07             	mov    (%rdi),%rax
 171f1d6:	ff 10                	call   *(%rax)
 171f1d8:	41 80 7c 24 2c 00    	cmpb   $0x0,0x2c(%r12)
 171f1de:	41 89 c5             	mov    %eax,%r13d
 171f1e1:	0f 85 21 01 00 00    	jne    171f308 <RecipeLauncher::getMappedMemoryInfo()@@Base+0x208>
 171f1e7:	85 c0                	test   %eax,%eax
 171f1e9:	0f 84 43 ff ff ff    	je     171f132 <RecipeLauncher::getMappedMemoryInfo()@@Base+0x32>
 171f1ef:	48 8b 1d 6a 5d 44 02 	mov    0x2445d6a(%rip),%rbx        # 3b64f60 <hl_logger::v12_inline_fmt_compile::ModuleLoggerData<synapse::LogManager::LogType>::levels@@Base-0x7540>
 171f1f6:	80 7b 18 04          	cmpb   $0x4,0x18(%rbx)
 171f1fa:	0f 86 58 01 00 00    	jbe    171f358 <RecipeLauncher::getMappedMemoryInfo()@@Base+0x258>
 171f200:	80 7b 19 04          	cmpb   $0x4,0x19(%rbx)
 171f204:	4c 8d 65 b0          	lea    -0x50(%rbp),%r12
 171f208:	77 8d                	ja     171f197 <RecipeLauncher::getMappedMemoryInfo()@@Base+0x97>
 171f20a:	48 8b 05 77 c5 44 02 	mov    0x244c577(%rip),%rax        # 3b6b788 <hl_logger::v12_inline_fmt_compile::moduleLoggerData<synapse::LogManager::LogType>@@Base-0x25ca58>
 171f211:	be 01 00 00 00       	mov    $0x1,%esi
 171f216:	48 8b 80 d0 0b 00 00 	mov    0xbd0(%rax),%rax
 171f21d:	f0 48 0f c1 70 10    	lock xadd %rsi,0x10(%rax)
 171f223:	0f b6 48 08          	movzbl 0x8(%rax),%ecx
 171f227:	49 89 f6             	mov    %rsi,%r14
 171f22a:	4c 89 e7             	mov    %r12,%rdi
 171f22d:	49 d3 ee             	shr    %cl,%r14
 171f230:	48 8b 08             	mov    (%rax),%rcx
 171f233:	48 8d 51 ff          	lea    -0x1(%rcx),%rdx
 171f237:	48 21 f2             	and    %rsi,%rdx
 171f23a:	48 8d 1c d5 00 00 00 	lea    0x0(,%rdx,8),%rbx
 171f241:	00 
 171f242:	48 29 d3             	sub    %rdx,%rbx
 171f245:	48 c1 e3 04          	shl    $0x4,%rbx
 171f249:	48 03 58 18          	add    0x18(%rax),%rbx
 171f24d:	48 89 de             	mov    %rbx,%rsi
 171f250:	e8 0b 55 f9 fe       	call   6b4760 <containers::hllog_1_0::ConcurrentRecentElementsQueue<hl_logger::v12_inline_fmt_compile::LazyLogItem<64u> >::acquireItem(containers::hllog_1_0::ConcurrentRecentElementsQueue<hl_logger::v12_inline_fmt_compile::LazyLogItem<64u> >::InternalItem const&)@plt>
 171f255:	44 89 f0             	mov    %r14d,%eax
 171f258:	2a 43 01             	sub    0x1(%rbx),%al
 171f25b:	3c 03                	cmp    $0x3,%al
 171f25d:	77 5c                	ja     171f2bb <RecipeLauncher::getMappedMemoryInfo()@@Base+0x1bb>
 171f25f:	80 7b 68 00          	cmpb   $0x0,0x68(%rbx)
 171f263:	4c 8d 63 08          	lea    0x8(%rbx),%r12
 171f267:	0f 85 39 01 00 00    	jne    171f3a6 <RecipeLauncher::getMappedMemoryInfo()@@Base+0x2a6>
 171f26d:	c6 43 08 04          	movb   $0x4,0x8(%rbx)
 171f271:	e8 1a a6 f8 fe       	call   6a9890 <hl_logger::internal::v1_0::getThreadId()@plt>
 171f276:	48 89 43 10          	mov    %rax,0x10(%rbx)
 171f27a:	e8 11 80 f9 fe       	call   6b7290 <std::chrono::_V2::system_clock::now()@plt>
 171f27f:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
 171f283:	49 89 44 24 10       	mov    %rax,0x10(%r12)
 171f288:	48 8d 43 20          	lea    0x20(%rbx),%rax
 171f28c:	48 89 43 60          	mov    %rax,0x60(%rbx)
 171f290:	48 8d 05 19 de 36 02 	lea    0x236de19(%rip),%rax        # 3a8d0b0 <MappedMemMgr::enumNamePoints@@Base+0x5f0>
 171f297:	c5 fa 7f 43 20       	vmovdqu %xmm0,0x20(%rbx)
 171f29c:	c6 43 68 01          	movb   $0x1,0x68(%rbx)
 171f2a0:	48 89 43 20          	mov    %rax,0x20(%rbx)
 171f2a4:	44 89 6b 28          	mov    %r13d,0x28(%rbx)
 171f2a8:	44 88 73 01          	mov    %r14b,0x1(%rbx)
 171f2ac:	c5 fa 7f 43 30       	vmovdqu %xmm0,0x30(%rbx)
 171f2b1:	c5 fa 7f 43 40       	vmovdqu %xmm0,0x40(%rbx)
 171f2b6:	c5 fa 7f 43 50       	vmovdqu %xmm0,0x50(%rbx)
 171f2bb:	80 7d b8 00          	cmpb   $0x0,-0x48(%rbp)
 171f2bf:	0f 84 d2 fe ff ff    	je     171f197 <RecipeLauncher::getMappedMemoryInfo()@@Base+0x97>
 171f2c5:	48 8b 45 b0          	mov    -0x50(%rbp),%rax
 171f2c9:	c6 00 00             	movb   $0x0,(%rax)
 171f2cc:	e9 c6 fe ff ff       	jmp    171f197 <RecipeLauncher::getMappedMemoryInfo()@@Base+0x97>
 171f2d1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
 171f2d8:	49 8b 54 24 40       	mov    0x40(%r12),%rdx
 171f2dd:	f0 48 01 82 b0 08 00 	lock add %rax,0x8b0(%rdx)
 171f2e4:	00 
 171f2e5:	49 8b 44 24 40       	mov    0x40(%r12),%rax
 171f2ea:	f0 48 83 80 b8 08 00 	lock addq $0x1,0x8b8(%rax)
 171f2f1:	00 01 
 171f2f3:	49 8b 44 24 40       	mov    0x40(%r12),%rax
 171f2f8:	48 8b 80 b8 08 00 00 	mov    0x8b8(%rax),%rax
 171f2ff:	e9 93 fe ff ff       	jmp    171f197 <RecipeLauncher::getMappedMemoryInfo()@@Base+0x97>
 171f304:	0f 1f 40 00          	nopl   0x0(%rax)
 171f308:	e8 d3 1b fa fe       	call   6c0ee0 <std::chrono::_V2::steady_clock::now()@plt>
 171f30d:	49 8b 54 24 40       	mov    0x40(%r12),%rdx
 171f312:	4c 29 f8             	sub    %r15,%rax
 171f315:	f0 48 01 82 50 08 00 	lock add %rax,0x850(%rdx)
 171f31c:	00 
 171f31d:	49 8b 44 24 40       	mov    0x40(%r12),%rax
 171f322:	f0 48 83 80 58 08 00 	lock addq $0x1,0x858(%rax)
 171f329:	00 01 
 171f32b:	49 8b 44 24 40       	mov    0x40(%r12),%rax
 171f330:	48 8b 80 58 08 00 00 	mov    0x858(%rax),%rax
 171f337:	45 85 ed             	test   %r13d,%r13d
 171f33a:	0f 84 f2 fd ff ff    	je     171f132 <RecipeLauncher::getMappedMemoryInfo()@@Base+0x32>
 171f340:	e9 aa fe ff ff       	jmp    171f1ef <RecipeLauncher::getMappedMemoryInfo()@@Base+0xef>
 171f345:	0f 1f 00             	nopl   (%rax)
 171f348:	e8 93 1b fa fe       	call   6c0ee0 <std::chrono::_V2::steady_clock::now()@plt>
 171f34d:	49 89 c7             	mov    %rax,%r15
 171f350:	e9 7a fe ff ff       	jmp    171f1cf <RecipeLauncher::getMappedMemoryInfo()@@Base+0xcf>
 171f355:	0f 1f 00             	nopl   (%rax)
 171f358:	4c 8d 65 b0          	lea    -0x50(%rbp),%r12
 171f35c:	be 0c 00 00 00       	mov    $0xc,%esi
 171f361:	44 89 6d ac          	mov    %r13d,-0x54(%rbp)
 171f365:	4c 89 e7             	mov    %r12,%rdi
 171f368:	e8 13 ff f9 fe       	call   6bf280 <std::shared_ptr<hl_logger::Logger> hl_logger::v12_inline_fmt_compile::getLogger<synapse::LogManager::LogType>(synapse::LogManager::LogType)@plt>
 171f36d:	4c 8d 4d ac          	lea    -0x54(%rbp),%r9
 171f371:	b9 13 00 00 00       	mov    $0x13,%ecx
 171f376:	31 d2                	xor    %edx,%edx
 171f378:	4c 8d 05 11 38 a5 01 	lea    0x1a53811(%rip),%r8        # 3172b90 <void std::deque<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > >, std::allocator<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > > >::_M_push_back_aux<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&>(std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&)@@Base+0x4f3a0>
 171f37f:	31 f6                	xor    %esi,%esi
 171f381:	4c 89 e7             	mov    %r12,%rdi
 171f384:	e8 77 fb ff ff       	call   171ef00 <RecipeLauncher::checkCompletionCompute(unsigned long)@@Base+0xd10>
 171f389:	48 8b 7d b8          	mov    -0x48(%rbp),%rdi
 171f38d:	48 85 ff             	test   %rdi,%rdi
 171f390:	74 05                	je     171f397 <RecipeLauncher::getMappedMemoryInfo()@@Base+0x297>
 171f392:	e8 59 b8 f8 fe       	call   6aabf0 <std::_Sp_counted_base<(__gnu_cxx::_Lock_policy)2>::_M_release()@plt>
 171f397:	80 7b 19 04          	cmpb   $0x4,0x19(%rbx)
 171f39b:	0f 87 f6 fd ff ff    	ja     171f197 <RecipeLauncher::getMappedMemoryInfo()@@Base+0x97>
 171f3a1:	e9 64 fe ff ff       	jmp    171f20a <RecipeLauncher::getMappedMemoryInfo()@@Base+0x10a>
 171f3a6:	c6 43 68 00          	movb   $0x0,0x68(%rbx)
 171f3aa:	4c 89 e7             	mov    %r12,%rdi
 171f3ad:	e8 de 2c f9 fe       	call   6b2090 <hl_logger::v12_inline_fmt_compile::LazyLogItem<64u>::~LazyLogItem()@plt>
 171f3b2:	e9 b6 fe ff ff       	jmp    171f26d <RecipeLauncher::getMappedMemoryInfo()@@Base+0x16d>
 171f3b7:	e8 84 9b f8 fe       	call   6a8f40 <__stack_chk_fail@plt>
 171f3bc:	f3 0f 1e fa          	endbr64
 171f3c0:	48 89 c7             	mov    %rax,%rdi
 171f3c3:	e9 62 88 15 ff       	jmp    877c2a <gaudi::QmansDefinition::getArbitrationMaster(internalStreamType) const@@Base+0x38b62>
 171f3c8:	f3 0f 1e fa          	endbr64
 171f3cc:	49 89 c4             	mov    %rax,%r12
 171f3cf:	e9 6b 88 15 ff       	jmp    877c3f <gaudi::QmansDefinition::getArbitrationMaster(internalStreamType) const@@Base+0x38b77>

/usr/lib/habanalabs/libSynapse.so:     file format elf64-x86-64


Disassembly of section .text:

00000000017229c0 <RecipeLauncher::downloadToDev(LaunchInfo const&)@@Base>:
 17229c0:	f3 0f 1e fa          	endbr64
 17229c4:	55                   	push   %rbp
 17229c5:	48 89 e5             	mov    %rsp,%rbp
 17229c8:	41 57                	push   %r15
 17229ca:	45 31 ff             	xor    %r15d,%r15d
 17229cd:	41 56                	push   %r14
 17229cf:	41 55                	push   %r13
 17229d1:	49 89 f5             	mov    %rsi,%r13
 17229d4:	41 54                	push   %r12
 17229d6:	49 89 fc             	mov    %rdi,%r12
 17229d9:	53                   	push   %rbx
 17229da:	48 83 ec 08          	sub    $0x8,%rsp
 17229de:	48 8b 1d 1b 89 44 02 	mov    0x244891b(%rip),%rbx        # 3b6b300 <g_globalStat@@Base-0x25c560>
 17229e5:	80 7b 2c 00          	cmpb   $0x0,0x2c(%rbx)
 17229e9:	0f 85 f1 00 00 00    	jne    1722ae0 <RecipeLauncher::downloadToDev(LaunchInfo const&)@@Base+0x120>
 17229ef:	49 8b 44 24 28       	mov    0x28(%r12),%rax
 17229f4:	4d 8d 74 24 30       	lea    0x30(%r12),%r14
 17229f9:	41 0f b6 8c 24 49 01 	movzbl 0x149(%r12),%ecx
 1722a00:	00 00 
 1722a02:	41 0f b6 94 24 48 01 	movzbl 0x148(%r12),%edx
 1722a09:	00 00 
 1722a0b:	4c 89 f7             	mov    %r14,%rdi
 1722a0e:	45 0f b6 84 24 58 01 	movzbl 0x158(%r12),%r8d
 1722a15:	00 00 
 1722a17:	48 8d b0 90 02 00 00 	lea    0x290(%rax),%rsi
 1722a1e:	e8 5d bb f9 fe       	call   6be580 <MappedMemorySectionsUtils::memcpyToMapped(MemorySectionsScal const&, llvm_vecsmall::SmallVector<RecipeSingleSection, 14u> const&, bool, bool, bool)@plt>
 1722a23:	80 7b 2c 00          	cmpb   $0x0,0x2c(%rbx)
 1722a27:	75 7f                	jne    1722aa8 <RecipeLauncher::downloadToDev(LaunchInfo const&)@@Base+0xe8>
 1722a29:	4c 89 ee             	mov    %r13,%rsi
 1722a2c:	4c 89 e7             	mov    %r12,%rdi
 1722a2f:	e8 dc 52 f8 fe       	call   6a7d10 <RecipeLauncher::patch(LaunchInfo const&)@plt>
 1722a34:	41 89 c5             	mov    %eax,%r13d
 1722a37:	85 c0                	test   %eax,%eax
 1722a39:	0f 85 c1 00 00 00    	jne    1722b00 <RecipeLauncher::downloadToDev(LaunchInfo const&)@@Base+0x140>
 1722a3f:	45 31 ff             	xor    %r15d,%r15d
 1722a42:	80 7b 2c 00          	cmpb   $0x0,0x2c(%rbx)
 1722a46:	0f 85 a4 00 00 00    	jne    1722af0 <RecipeLauncher::downloadToDev(LaunchInfo const&)@@Base+0x130>
 1722a4c:	4c 89 f6             	mov    %r14,%rsi
 1722a4f:	4c 89 e7             	mov    %r12,%rdi
 1722a52:	e8 99 82 fa fe       	call   6cacf0 <RecipeLauncher::pdmaDownload(MemorySectionsScal const&)@plt>
 1722a57:	41 89 c5             	mov    %eax,%r13d
 1722a5a:	85 c0                	test   %eax,%eax
 1722a5c:	74 12                	je     1722a70 <RecipeLauncher::downloadToDev(LaunchInfo const&)@@Base+0xb0>
 1722a5e:	48 83 c4 08          	add    $0x8,%rsp
 1722a62:	44 89 e8             	mov    %r13d,%eax
 1722a65:	5b                   	pop    %rbx
 1722a66:	41 5c                	pop    %r12
 1722a68:	41 5d                	pop    %r13
 1722a6a:	41 5e                	pop    %r14
 1722a6c:	41 5f                	pop    %r15
 1722a6e:	5d                   	pop    %rbp
 1722a6f:	c3                   	ret
 1722a70:	80 7b 2c 00          	cmpb   $0x0,0x2c(%rbx)
 1722a74:	74 e8                	je     1722a5e <RecipeLauncher::downloadToDev(LaunchInfo const&)@@Base+0x9e>
 1722a76:	e8 65 e4 f9 fe       	call   6c0ee0 <std::chrono::_V2::steady_clock::now()@plt>
 1722a7b:	48 8b 53 40          	mov    0x40(%rbx),%rdx
 1722a7f:	4c 29 f8             	sub    %r15,%rax
 1722a82:	f0 48 01 82 30 08 00 	lock add %rax,0x830(%rdx)
 1722a89:	00 
 1722a8a:	48 8b 43 40          	mov    0x40(%rbx),%rax
 1722a8e:	f0 48 83 80 38 08 00 	lock addq $0x1,0x838(%rax)
 1722a95:	00 01 
 1722a97:	48 8b 43 40          	mov    0x40(%rbx),%rax
 1722a9b:	48 8b 80 38 08 00 00 	mov    0x838(%rax),%rax
 1722aa2:	eb ba                	jmp    1722a5e <RecipeLauncher::downloadToDev(LaunchInfo const&)@@Base+0x9e>
 1722aa4:	0f 1f 40 00          	nopl   0x0(%rax)
 1722aa8:	e8 33 e4 f9 fe       	call   6c0ee0 <std::chrono::_V2::steady_clock::now()@plt>
 1722aad:	48 8b 53 40          	mov    0x40(%rbx),%rdx
 1722ab1:	4c 29 f8             	sub    %r15,%rax
 1722ab4:	f0 48 01 82 10 08 00 	lock add %rax,0x810(%rdx)
 1722abb:	00 
 1722abc:	48 8b 43 40          	mov    0x40(%rbx),%rax
 1722ac0:	f0 48 83 80 18 08 00 	lock addq $0x1,0x818(%rax)
 1722ac7:	00 01 
 1722ac9:	48 8b 43 40          	mov    0x40(%rbx),%rax
 1722acd:	48 8b 80 18 08 00 00 	mov    0x818(%rax),%rax
 1722ad4:	e9 50 ff ff ff       	jmp    1722a29 <RecipeLauncher::downloadToDev(LaunchInfo const&)@@Base+0x69>
 1722ad9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
 1722ae0:	e8 fb e3 f9 fe       	call   6c0ee0 <std::chrono::_V2::steady_clock::now()@plt>
 1722ae5:	49 89 c7             	mov    %rax,%r15
 1722ae8:	e9 02 ff ff ff       	jmp    17229ef <RecipeLauncher::downloadToDev(LaunchInfo const&)@@Base+0x2f>
 1722aed:	0f 1f 00             	nopl   (%rax)
 1722af0:	e8 eb e3 f9 fe       	call   6c0ee0 <std::chrono::_V2::steady_clock::now()@plt>
 1722af5:	49 89 c7             	mov    %rax,%r15
 1722af8:	e9 4f ff ff ff       	jmp    1722a4c <RecipeLauncher::downloadToDev(LaunchInfo const&)@@Base+0x8c>
 1722afd:	0f 1f 00             	nopl   (%rax)
 1722b00:	4c 89 e7             	mov    %r12,%rdi
 1722b03:	e8 e8 43 f9 fe       	call   6b6ef0 <RecipeLauncher::clearMemMgrs()@plt>
 1722b08:	e9 51 ff ff ff       	jmp    1722a5e <RecipeLauncher::downloadToDev(LaunchInfo const&)@@Base+0x9e>

/usr/lib/habanalabs/libSynapse.so:     file format elf64-x86-64


Disassembly of section .text:

00000000017246e0 <RecipeLauncher::scalEnqueue(unsigned long)@@Base>:
 17246e0:	f3 0f 1e fa          	endbr64
 17246e4:	55                   	push   %rbp
 17246e5:	31 d2                	xor    %edx,%edx
 17246e7:	48 89 e5             	mov    %rsp,%rbp
 17246ea:	41 56                	push   %r14
 17246ec:	49 89 f6             	mov    %rsi,%r14
 17246ef:	be 01 00 00 00       	mov    $0x1,%esi
 17246f4:	41 55                	push   %r13
 17246f6:	41 54                	push   %r12
 17246f8:	53                   	push   %rbx
 17246f9:	48 89 fb             	mov    %rdi,%rbx
 17246fc:	48 83 ec 20          	sub    $0x20,%rsp
 1724700:	4c 8b 6f 28          	mov    0x28(%rdi),%r13
 1724704:	48 8b bf 08 01 00 00 	mov    0x108(%rdi),%rdi
 172470b:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
 1724712:	00 00 
 1724714:	48 89 45 d8          	mov    %rax,-0x28(%rbp)
 1724718:	31 c0                	xor    %eax,%eax
 172471a:	48 8b 07             	mov    (%rdi),%rax
 172471d:	49 81 c5 98 05 00 00 	add    $0x598,%r13
 1724724:	ff 90 c8 00 00 00    	call   *0xc8(%rax)
 172472a:	4c 89 ee             	mov    %r13,%rsi
 172472d:	48 89 df             	mov    %rbx,%rdi
 1724730:	e8 cb 66 fa fe       	call   6cae00 <RecipeLauncher::addBaseAddresses(EngineGrpArr const&)@plt>
 1724735:	41 89 c4             	mov    %eax,%r12d
 1724738:	85 c0                	test   %eax,%eax
 172473a:	0f 85 ab 00 00 00    	jne    17247eb <RecipeLauncher::scalEnqueue(unsigned long)@@Base+0x10b>
 1724740:	48 8b 43 28          	mov    0x28(%rbx),%rax
 1724744:	48 8b 00             	mov    (%rax),%rax
 1724747:	80 78 48 00          	cmpb   $0x0,0x48(%rax)
 172474b:	74 16                	je     1724763 <RecipeLauncher::scalEnqueue(unsigned long)@@Base+0x83>
 172474d:	4c 89 ee             	mov    %r13,%rsi
 1724750:	48 89 df             	mov    %rbx,%rdi
 1724753:	e8 08 93 f8 fe       	call   6ada60 <RecipeLauncher::addPrgDataBaseAddresses(EngineGrpArr const&)@plt>
 1724758:	41 89 c4             	mov    %eax,%r12d
 172475b:	85 c0                	test   %eax,%eax
 172475d:	0f 85 88 00 00 00    	jne    17247eb <RecipeLauncher::scalEnqueue(unsigned long)@@Base+0x10b>
 1724763:	4c 89 f6             	mov    %r14,%rsi
 1724766:	48 89 df             	mov    %rbx,%rdi
 1724769:	e8 d2 7f f8 fe       	call   6ac740 <RecipeLauncher::addEcbListWithBarrierAndSend(unsigned long)@plt>
 172476e:	41 89 c4             	mov    %eax,%r12d
 1724771:	85 c0                	test   %eax,%eax
 1724773:	75 76                	jne    17247eb <RecipeLauncher::scalEnqueue(unsigned long)@@Base+0x10b>
 1724775:	48 8b bb 08 01 00 00 	mov    0x108(%rbx),%rdi
 172477c:	be 01 00 00 00       	mov    $0x1,%esi
 1724781:	48 8b 07             	mov    (%rdi),%rax
 1724784:	ff 90 d0 00 00 00    	call   *0xd0(%rax)
 172478a:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
 172478e:	48 8d 55 c0          	lea    -0x40(%rbp),%rdx
 1724792:	48 8b bb 08 01 00 00 	mov    0x108(%rbx),%rdi
 1724799:	c5 f9 7f 45 c0       	vmovdqa %xmm0,-0x40(%rbp)
 172479e:	be 01 00 00 00       	mov    $0x1,%esi
 17247a3:	48 8b 07             	mov    (%rdi),%rax
 17247a6:	48 03 b8 28 ff ff ff 	add    -0xd8(%rax),%rdi
 17247ad:	48 8b 07             	mov    (%rdi),%rax
 17247b0:	ff 50 28             	call   *0x28(%rax)
 17247b3:	48 8b 83 00 01 00 00 	mov    0x100(%rbx),%rax
 17247ba:	48 8b 75 c8          	mov    -0x38(%rbp),%rsi
 17247be:	48 8b 00             	mov    (%rax),%rax
 17247c1:	48 8b 78 08          	mov    0x8(%rax),%rdi
 17247c5:	48 8b 07             	mov    (%rdi),%rax
 17247c8:	ff 50 10             	call   *0x10(%rax)
 17247cb:	48 8b 83 00 01 00 00 	mov    0x100(%rbx),%rax
 17247d2:	48 8b 75 c8          	mov    -0x38(%rbp),%rsi
 17247d6:	48 8b 40 10          	mov    0x10(%rax),%rax
 17247da:	48 8b 78 08          	mov    0x8(%rax),%rdi
 17247de:	48 8b 07             	mov    (%rdi),%rax
 17247e1:	ff 50 10             	call   *0x10(%rax)
 17247e4:	c6 83 5a 01 00 00 00 	movb   $0x0,0x15a(%rbx)
 17247eb:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
 17247ef:	64 48 2b 04 25 28 00 	sub    %fs:0x28,%rax
 17247f6:	00 00 
 17247f8:	75 10                	jne    172480a <RecipeLauncher::scalEnqueue(unsigned long)@@Base+0x12a>
 17247fa:	48 83 c4 20          	add    $0x20,%rsp
 17247fe:	44 89 e0             	mov    %r12d,%eax
 1724801:	5b                   	pop    %rbx
 1724802:	41 5c                	pop    %r12
 1724804:	41 5d                	pop    %r13
 1724806:	41 5e                	pop    %r14
 1724808:	5d                   	pop    %rbp
 1724809:	c3                   	ret
 172480a:	e8 31 47 f8 fe       	call   6a8f40 <__stack_chk_fail@plt>
