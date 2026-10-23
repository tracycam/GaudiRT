
/usr/lib/habanalabs/libSynapse.so:     file format elf64-x86-64


Disassembly of section .text:

0000000001770190 <QueueComputeScal::launch(LaunchInfo const&)@@Base>:
 1770190:	f3 0f 1e fa          	endbr64
 1770194:	55                   	push   %rbp
 1770195:	48 89 e5             	mov    %rsp,%rbp
 1770198:	41 57                	push   %r15
 177019a:	4c 8d 7f 38          	lea    0x38(%rdi),%r15
 177019e:	41 56                	push   %r14
 17701a0:	49 89 f6             	mov    %rsi,%r14
 17701a3:	41 55                	push   %r13
 17701a5:	41 54                	push   %r12
 17701a7:	49 89 fc             	mov    %rdi,%r12
 17701aa:	4c 89 ff             	mov    %r15,%rdi
 17701ad:	53                   	push   %rbx
 17701ae:	48 81 ec b8 02 00 00 	sub    $0x2b8,%rsp
 17701b5:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
 17701bc:	00 00 
 17701be:	48 89 45 c8          	mov    %rax,-0x38(%rbp)
 17701c2:	31 c0                	xor    %eax,%eax
 17701c4:	e8 e7 84 f3 fe       	call   6a86b0 <pthread_mutex_lock@plt>
 17701c9:	85 c0                	test   %eax,%eax
 17701cb:	0f 85 45 07 00 00    	jne    1770916 <QueueComputeScal::launch(LaunchInfo const&)@@Base+0x786>
 17701d1:	48 8b 1d 28 b1 3f 02 	mov    0x23fb128(%rip),%rbx        # 3b6b300 <g_globalStat@@Base-0x25c560>
 17701d8:	45 31 ed             	xor    %r13d,%r13d
 17701db:	80 7b 2c 00          	cmpb   $0x0,0x2c(%rbx)
 17701df:	0f 85 2b 04 00 00    	jne    1770610 <QueueComputeScal::launch(LaunchInfo const&)@@Base+0x480>
 17701e5:	49 8d 44 24 78       	lea    0x78(%r12),%rax
 17701ea:	48 8d 95 68 fd ff ff 	lea    -0x298(%rbp),%rdx
 17701f1:	48 8d b5 60 fd ff ff 	lea    -0x2a0(%rbp),%rsi
 17701f8:	48 89 c7             	mov    %rax,%rdi
 17701fb:	48 89 85 58 fd ff ff 	mov    %rax,-0x2a8(%rbp)
 1770202:	e8 29 76 f3 fe       	call   6a7830 <LaunchTracker::checkForCompletion(unsigned long&, unsigned long&)@plt>
 1770207:	80 7b 2c 00          	cmpb   $0x0,0x2c(%rbx)
 177020b:	0f 85 0f 04 00 00    	jne    1770620 <QueueComputeScal::launch(LaunchInfo const&)@@Base+0x490>
 1770211:	49 8b 46 18          	mov    0x18(%r14),%rax
 1770215:	48 8b 58 08          	mov    0x8(%rax),%rbx
 1770219:	48 85 db             	test   %rbx,%rbx
 177021c:	74 0e                	je     177022c <QueueComputeScal::launch(LaunchInfo const&)@@Base+0x9c>
 177021e:	4c 89 f6             	mov    %r14,%rsi
 1770221:	4c 89 e7             	mov    %r12,%rdi
 1770224:	e8 d7 04 f4 fe       	call   6b0700 <QueueComputeScal::getDynamicShapeProcessor(LaunchInfo const&)@plt>
 1770229:	48 89 c3             	mov    %rax,%rbx
 177022c:	49 8b 7c 24 30       	mov    0x30(%r12),%rdi
 1770231:	48 85 ff             	test   %rdi,%rdi
 1770234:	0f 84 7e 02 00 00    	je     17704b8 <QueueComputeScal::launch(LaunchInfo const&)@@Base+0x328>
 177023a:	48 8b 15 7f 6b 3f 02 	mov    0x23f6b7f(%rip),%rdx        # 3b66dc0 <typeinfo for ScalStreamComputeInterface@@Base+0x8e488>
 1770241:	48 8b 35 b0 5f 3f 02 	mov    0x23f5fb0(%rip),%rsi        # 3b661f8 <typeinfo for ScalStreamCopyInterface@@Base+0x8d8d8>
 1770248:	48 c7 c1 ff ff ff ff 	mov    $0xffffffffffffffff,%rcx
 177024f:	e8 dc 10 f5 fe       	call   6c1330 <__dynamic_cast@plt>
 1770254:	49 89 c5             	mov    %rax,%r13
 1770257:	48 85 c0             	test   %rax,%rax
 177025a:	0f 84 58 02 00 00    	je     17704b8 <QueueComputeScal::launch(LaunchInfo const&)@@Base+0x328>
 1770260:	49 8b 84 24 e8 00 00 	mov    0xe8(%r12),%rax
 1770267:	00 
 1770268:	49 8b 7e 18          	mov    0x18(%r14),%rdi
 177026c:	49 8d b4 24 28 01 00 	lea    0x128(%r12),%rsi
 1770273:	00 
 1770274:	49 8b 4c 24 70       	mov    0x70(%r12),%rcx
 1770279:	49 8b 94 24 e0 00 00 	mov    0xe0(%r12),%rdx
 1770280:	00 
 1770281:	48 89 b5 50 fd ff ff 	mov    %rsi,-0x2b0(%rbp)
 1770288:	48 83 c0 01          	add    $0x1,%rax
 177028c:	48 89 bd 28 fd ff ff 	mov    %rdi,-0x2d8(%rbp)
 1770293:	bf 70 01 00 00       	mov    $0x170,%edi
 1770298:	49 89 84 24 e8 00 00 	mov    %rax,0xe8(%r12)
 177029f:	00 
 17702a0:	45 0f b6 56 40       	movzbl 0x40(%r14),%r10d
 17702a5:	48 89 85 48 fd ff ff 	mov    %rax,-0x2b8(%rbp)
 17702ac:	48 89 8d 30 fd ff ff 	mov    %rcx,-0x2d0(%rbp)
 17702b3:	48 89 95 40 fd ff ff 	mov    %rdx,-0x2c0(%rbp)
 17702ba:	44 89 95 24 fd ff ff 	mov    %r10d,-0x2dc(%rbp)
 17702c1:	e8 6a 3f f3 fe       	call   6a4230 <operator new(unsigned long)@plt>
 17702c6:	44 8b 95 24 fd ff ff 	mov    -0x2dc(%rbp),%r10d
 17702cd:	4c 8b 8d 58 fd ff ff 	mov    -0x2a8(%rbp),%r9
 17702d4:	49 8d 94 24 28 01 00 	lea    0x128(%r12),%rdx
 17702db:	00 
 17702dc:	4c 89 ee             	mov    %r13,%rsi
 17702df:	4c 8b 85 30 fd ff ff 	mov    -0x2d0(%rbp),%r8
 17702e6:	48 8b 8d 40 fd ff ff 	mov    -0x2c0(%rbp),%rcx
 17702ed:	48 89 c7             	mov    %rax,%rdi
 17702f0:	48 89 85 50 fd ff ff 	mov    %rax,-0x2b0(%rbp)
 17702f7:	41 52                	push   %r10
 17702f9:	ff b5 48 fd ff ff    	push   -0x2b8(%rbp)
 17702ff:	53                   	push   %rbx
 1770300:	48 89 c3             	mov    %rax,%rbx
 1770303:	ff b5 28 fd ff ff    	push   -0x2d8(%rbp)
 1770309:	e8 32 e4 f4 fe       	call   6be740 <RecipeLauncher::RecipeLauncher(ScalStreamComputeInterface*, ComputeCompoundResources const*, DevMemoryAllocInterface&, MemMgrs&, LaunchTrackerInterface&, InternalRecipeHandle const*, DynamicRecipe*, unsigned long, unsigned char)@plt>
 177030e:	48 83 c4 20          	add    $0x20,%rsp
 1770312:	4c 89 f6             	mov    %r14,%rsi
 1770315:	48 89 df             	mov    %rbx,%rdi
 1770318:	e8 73 40 f4 fe       	call   6b4390 <RecipeLauncher::launch(LaunchInfo const&)@plt>
 177031d:	41 89 c5             	mov    %eax,%r13d
 1770320:	85 c0                	test   %eax,%eax
 1770322:	0f 84 30 05 00 00    	je     1770858 <QueueComputeScal::launch(LaunchInfo const&)@@Base+0x6c8>
 1770328:	48 8b 1d 31 4c 3f 02 	mov    0x23f4c31(%rip),%rbx        # 3b64f60 <hl_logger::v12_inline_fmt_compile::ModuleLoggerData<synapse::LogManager::LogType>::levels@@Base-0x7540>
 177032f:	80 7b 18 04          	cmpb   $0x4,0x18(%rbx)
 1770333:	0f 87 17 04 00 00    	ja     1770750 <QueueComputeScal::launch(LaunchInfo const&)@@Base+0x5c0>
 1770339:	49 8b 46 18          	mov    0x18(%r14),%rax
 177033d:	4c 8d a5 70 fd ff ff 	lea    -0x290(%rbp),%r12
 1770344:	be 0c 00 00 00       	mov    $0xc,%esi
 1770349:	4c 89 e7             	mov    %r12,%rdi
 177034c:	48 8b 80 b8 06 00 00 	mov    0x6b8(%rax),%rax
 1770353:	48 89 85 28 fd ff ff 	mov    %rax,-0x2d8(%rbp)
 177035a:	e8 21 ef f4 fe       	call   6bf280 <std::shared_ptr<hl_logger::Logger> hl_logger::v12_inline_fmt_compile::getLogger<synapse::LogManager::LogType>(synapse::LogManager::LogType)@plt>
 177035f:	48 8d 0d b2 38 a0 01 	lea    0x1a038b2(%rip),%rcx        # 3173c18 <void std::deque<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > >, std::allocator<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > > >::_M_push_back_aux<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&>(std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&)@@Base+0x50428>
 1770366:	48 8d 85 d0 fd ff ff 	lea    -0x230(%rbp),%rax
 177036d:	be 06 00 00 00       	mov    $0x6,%esi
 1770372:	48 c7 85 30 fd ff ff 	movq   $0x17,-0x2d0(%rbp)
 1770379:	17 00 00 00 
 177037d:	48 89 8d 38 fd ff ff 	mov    %rcx,-0x2c8(%rbp)
 1770384:	48 8d 0d fd be 34 02 	lea    0x234befd(%rip),%rcx        # 3abc288 <typeinfo for hl_gcfg::inline_ver_1_11::GcfgItem@@Base+0x18>
 177038b:	48 8d bd b0 fd ff ff 	lea    -0x250(%rbp),%rdi
 1770392:	48 89 85 40 fd ff ff 	mov    %rax,-0x2c0(%rbp)
 1770399:	48 8d 51 10          	lea    0x10(%rcx),%rdx
 177039d:	48 89 8d 48 fd ff ff 	mov    %rcx,-0x2b8(%rbp)
 17703a4:	48 c7 85 a0 fd ff ff 	movq   $0x2b,-0x260(%rbp)
 17703ab:	2b 00 00 00 
 17703af:	c4 e1 f9 6e c2       	vmovq  %rdx,%xmm0
 17703b4:	48 8d 15 1f 39 a0 01 	lea    0x1a0391f(%rip),%rdx        # 3173cda <void std::deque<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > >, std::allocator<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > > >::_M_push_back_aux<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&>(std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&)@@Base+0x504ea>
 17703bb:	c4 e3 f9 22 c0 01    	vpinsrq $0x1,%rax,%xmm0,%xmm0
 17703c1:	48 89 95 98 fd ff ff 	mov    %rdx,-0x268(%rbp)
 17703c8:	48 8d 15 95 ac 9b 01 	lea    0x19bac95(%rip),%rdx        # 312b064 <void std::deque<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > >, std::allocator<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > > >::_M_push_back_aux<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&>(std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&)@@Base+0x7874>
 17703cf:	c5 f9 7f 85 b0 fd ff 	vmovdqa %xmm0,-0x250(%rbp)
 17703d6:	ff 
 17703d7:	c5 f9 6f 05 31 73 a8 	vmovdqa 0x1a87331(%rip),%xmm0        # 31f7710 <std::__detail::__to_chars_10_impl<unsigned long>(char*, unsigned int, unsigned long)::__digits@@Base+0xd0>
 17703de:	01 
 17703df:	c5 f9 7f 85 c0 fd ff 	vmovdqa %xmm0,-0x240(%rbp)
 17703e6:	ff 
 17703e7:	e8 34 f2 ff ff       	call   176f620 <QueueComputeScal::getDynamicShapeProcessor(LaunchInfo const&)@@Base+0x1640>
 17703ec:	48 8b b5 98 fd ff ff 	mov    -0x268(%rbp),%rsi
 17703f3:	48 8b 95 a0 fd ff ff 	mov    -0x260(%rbp),%rdx
 17703fa:	48 89 c7             	mov    %rax,%rdi
 17703fd:	e8 8e 6e 44 ff       	call   bb7290 <std::_Rb_tree<std::pair<unsigned int, std::shared_ptr<BasicFieldInfo> >, std::pair<unsigned int, std::shared_ptr<BasicFieldInfo> >, std::_Identity<std::pair<unsigned int, std::shared_ptr<BasicFieldInfo> > >, PairComparator<unsigned int, std::shared_ptr<BasicFieldInfo> >, std::allocator<std::pair<unsigned int, std::shared_ptr<BasicFieldInfo> > > >::_M_get_insert_hint_unique_pos(std::_Rb_tree_const_iterator<std::pair<unsigned int, std::shared_ptr<BasicFieldInfo> > >, std::pair<unsigned int, std::shared_ptr<BasicFieldInfo> > const&)@@Base+0xc770>
 1770402:	48 8b b5 28 fd ff ff 	mov    -0x2d8(%rbp),%rsi
 1770409:	48 89 c7             	mov    %rax,%rdi
 177040c:	e8 0f 5e 4c ff       	call   c36220 <std::_Sp_counted_ptr<DeviceHeapAllocatorManager*, (__gnu_cxx::_Lock_policy)2>::_M_dispose()@@Base+0xb0>
 1770411:	48 83 ec 08          	sub    $0x8,%rsp
 1770415:	48 8b 8d c0 fd ff ff 	mov    -0x240(%rbp),%rcx
 177041c:	31 d2                	xor    %edx,%edx
 177041e:	4c 89 e7             	mov    %r12,%rdi
 1770421:	6a 00                	push   $0x0
 1770423:	41 b9 29 01 00 00    	mov    $0x129,%r9d
 1770429:	be 04 00 00 00       	mov    $0x4,%esi
 177042e:	4c 8b 85 b8 fd ff ff 	mov    -0x248(%rbp),%r8
 1770435:	ff b5 38 fd ff ff    	push   -0x2c8(%rbp)
 177043b:	ff b5 30 fd ff ff    	push   -0x2d0(%rbp)
 1770441:	e8 6a 5c f3 fe       	call   6a60b0 <hl_logger::v1_0::log(std::shared_ptr<hl_logger::Logger> const&, int, bool, std::basic_string_view<char, std::char_traits<char> >, std::basic_string_view<char, std::char_traits<char> >, int, bool)@plt>
 1770446:	48 8b 8d 48 fd ff ff 	mov    -0x2b8(%rbp),%rcx
 177044d:	48 8b bd b8 fd ff ff 	mov    -0x248(%rbp),%rdi
 1770454:	48 83 c4 20          	add    $0x20,%rsp
 1770458:	48 8d 41 10          	lea    0x10(%rcx),%rax
 177045c:	48 89 85 b0 fd ff ff 	mov    %rax,-0x250(%rbp)
 1770463:	48 3b bd 40 fd ff ff 	cmp    -0x2c0(%rbp),%rdi
 177046a:	74 0c                	je     1770478 <QueueComputeScal::launch(LaunchInfo const&)@@Base+0x2e8>
 177046c:	48 8b b5 c8 fd ff ff 	mov    -0x238(%rbp),%rsi
 1770473:	e8 08 28 f3 fe       	call   6a2c80 <operator delete(void*, unsigned long)@plt>
 1770478:	48 8b bd 78 fd ff ff 	mov    -0x288(%rbp),%rdi
 177047f:	48 85 ff             	test   %rdi,%rdi
 1770482:	74 05                	je     1770489 <QueueComputeScal::launch(LaunchInfo const&)@@Base+0x2f9>
 1770484:	e8 67 a7 f3 fe       	call   6aabf0 <std::_Sp_counted_base<(__gnu_cxx::_Lock_policy)2>::_M_release()@plt>
 1770489:	80 7b 19 04          	cmpb   $0x4,0x19(%rbx)
 177048d:	0f 86 ce 02 00 00    	jbe    1770761 <QueueComputeScal::launch(LaunchInfo const&)@@Base+0x5d1>
 1770493:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
 1770498:	41 83 fd 1e          	cmp    $0x1e,%r13d
 177049c:	0f 84 1e 04 00 00    	je     17708c0 <QueueComputeScal::launch(LaunchInfo const&)@@Base+0x730>
 17704a2:	48 8b bd 50 fd ff ff 	mov    -0x2b0(%rbp),%rdi
 17704a9:	48 8b 07             	mov    (%rdi),%rax
 17704ac:	ff 50 08             	call   *0x8(%rax)
 17704af:	e9 e3 03 00 00       	jmp    1770897 <QueueComputeScal::launch(LaunchInfo const&)@@Base+0x707>
 17704b4:	0f 1f 40 00          	nopl   0x0(%rax)
 17704b8:	48 8b 1d a1 4a 3f 02 	mov    0x23f4aa1(%rip),%rbx        # 3b64f60 <hl_logger::v12_inline_fmt_compile::ModuleLoggerData<synapse::LogManager::LogType>::levels@@Base-0x7540>
 17704bf:	80 7b 18 04          	cmpb   $0x4,0x18(%rbx)
 17704c3:	0f 87 8f 01 00 00    	ja     1770658 <QueueComputeScal::launch(LaunchInfo const&)@@Base+0x4c8>
 17704c9:	4c 8d b5 90 fd ff ff 	lea    -0x270(%rbp),%r14
 17704d0:	be 0c 00 00 00       	mov    $0xc,%esi
 17704d5:	4c 89 f7             	mov    %r14,%rdi
 17704d8:	e8 a3 ed f4 fe       	call   6bf280 <std::shared_ptr<hl_logger::Logger> hl_logger::v12_inline_fmt_compile::getLogger<synapse::LogManager::LogType>(synapse::LogManager::LogType)@plt>
 17704dd:	48 8d 0d a4 bd 34 02 	lea    0x234bda4(%rip),%rcx        # 3abc288 <typeinfo for hl_gcfg::inline_ver_1_11::GcfgItem@@Base+0x18>
 17704e4:	be 06 00 00 00       	mov    $0x6,%esi
 17704e9:	48 8d 85 d0 fd ff ff 	lea    -0x230(%rbp),%rax
 17704f0:	48 8d bd b0 fd ff ff 	lea    -0x250(%rbp),%rdi
 17704f7:	48 89 85 40 fd ff ff 	mov    %rax,-0x2c0(%rbp)
 17704fe:	41 bc 17 00 00 00    	mov    $0x17,%r12d
 1770504:	4c 8d 2d 0d 37 a0 01 	lea    0x1a0370d(%rip),%r13        # 3173c18 <void std::deque<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > >, std::allocator<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > > >::_M_push_back_aux<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&>(std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&)@@Base+0x50428>
 177050b:	48 8d 51 10          	lea    0x10(%rcx),%rdx
 177050f:	48 89 8d 48 fd ff ff 	mov    %rcx,-0x2b8(%rbp)
 1770516:	48 c7 85 80 fd ff ff 	movq   $0x17,-0x280(%rbp)
 177051d:	17 00 00 00 
 1770521:	c4 e1 f9 6e c2       	vmovq  %rdx,%xmm0
 1770526:	48 8d 15 68 40 9b 01 	lea    0x19b4068(%rip),%rdx        # 3124595 <void std::deque<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > >, std::allocator<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > > >::_M_push_back_aux<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&>(std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&)@@Base+0xda5>
 177052d:	c4 e3 f9 22 c0 01    	vpinsrq $0x1,%rax,%xmm0,%xmm0
 1770533:	48 89 95 78 fd ff ff 	mov    %rdx,-0x288(%rbp)
 177053a:	48 8d 15 23 ab 9b 01 	lea    0x19bab23(%rip),%rdx        # 312b064 <void std::deque<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > >, std::allocator<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > > >::_M_push_back_aux<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&>(std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&)@@Base+0x7874>
 1770541:	c5 f9 7f 85 b0 fd ff 	vmovdqa %xmm0,-0x250(%rbp)
 1770548:	ff 
 1770549:	c5 f9 6f 05 bf 71 a8 	vmovdqa 0x1a871bf(%rip),%xmm0        # 31f7710 <std::__detail::__to_chars_10_impl<unsigned long>(char*, unsigned int, unsigned long)::__digits@@Base+0xd0>
 1770550:	01 
 1770551:	c5 f9 7f 85 c0 fd ff 	vmovdqa %xmm0,-0x240(%rbp)
 1770558:	ff 
 1770559:	e8 c2 f0 ff ff       	call   176f620 <QueueComputeScal::getDynamicShapeProcessor(LaunchInfo const&)@@Base+0x1640>
 177055e:	48 8b b5 78 fd ff ff 	mov    -0x288(%rbp),%rsi
 1770565:	48 8b 95 80 fd ff ff 	mov    -0x280(%rbp),%rdx
 177056c:	48 89 c7             	mov    %rax,%rdi
 177056f:	e8 1c 6d 44 ff       	call   bb7290 <std::_Rb_tree<std::pair<unsigned int, std::shared_ptr<BasicFieldInfo> >, std::pair<unsigned int, std::shared_ptr<BasicFieldInfo> >, std::_Identity<std::pair<unsigned int, std::shared_ptr<BasicFieldInfo> > >, PairComparator<unsigned int, std::shared_ptr<BasicFieldInfo> >, std::allocator<std::pair<unsigned int, std::shared_ptr<BasicFieldInfo> > > >::_M_get_insert_hint_unique_pos(std::_Rb_tree_const_iterator<std::pair<unsigned int, std::shared_ptr<BasicFieldInfo> > >, std::pair<unsigned int, std::shared_ptr<BasicFieldInfo> > const&)@@Base+0xc770>
 1770574:	48 89 c7             	mov    %rax,%rdi
 1770577:	48 8d 35 dc cf 9b 01 	lea    0x19bcfdc(%rip),%rsi        # 312d55a <void std::deque<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > >, std::allocator<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > > >::_M_push_back_aux<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&>(std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&)@@Base+0x9d6a>
 177057e:	e8 dd a8 4a ff       	call   c1ae60 <void std::vector<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::allocator<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > > >::_M_realloc_insert<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >(__gnu_cxx::__normal_iterator<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >*, std::vector<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::allocator<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > > > >, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >&&)@@Base+0x270>
 1770583:	48 83 ec 08          	sub    $0x8,%rsp
 1770587:	48 8b 8d c0 fd ff ff 	mov    -0x240(%rbp),%rcx
 177058e:	31 d2                	xor    %edx,%edx
 1770590:	4c 89 f7             	mov    %r14,%rdi
 1770593:	6a 00                	push   $0x0
 1770595:	41 b9 1a 01 00 00    	mov    $0x11a,%r9d
 177059b:	be 04 00 00 00       	mov    $0x4,%esi
 17705a0:	4c 8b 85 b8 fd ff ff 	mov    -0x248(%rbp),%r8
 17705a7:	41 55                	push   %r13
 17705a9:	41 54                	push   %r12
 17705ab:	e8 00 5b f3 fe       	call   6a60b0 <hl_logger::v1_0::log(std::shared_ptr<hl_logger::Logger> const&, int, bool, std::basic_string_view<char, std::char_traits<char> >, std::basic_string_view<char, std::char_traits<char> >, int, bool)@plt>
 17705b0:	48 8b 8d 48 fd ff ff 	mov    -0x2b8(%rbp),%rcx
 17705b7:	48 8b bd b8 fd ff ff 	mov    -0x248(%rbp),%rdi
 17705be:	48 83 c4 20          	add    $0x20,%rsp
 17705c2:	48 8d 41 10          	lea    0x10(%rcx),%rax
 17705c6:	48 89 85 b0 fd ff ff 	mov    %rax,-0x250(%rbp)
 17705cd:	48 3b bd 40 fd ff ff 	cmp    -0x2c0(%rbp),%rdi
 17705d4:	74 0c                	je     17705e2 <QueueComputeScal::launch(LaunchInfo const&)@@Base+0x452>
 17705d6:	48 8b b5 c8 fd ff ff 	mov    -0x238(%rbp),%rsi
 17705dd:	e8 9e 26 f3 fe       	call   6a2c80 <operator delete(void*, unsigned long)@plt>
 17705e2:	48 8b bd 98 fd ff ff 	mov    -0x268(%rbp),%rdi
 17705e9:	48 85 ff             	test   %rdi,%rdi
 17705ec:	74 05                	je     17705f3 <QueueComputeScal::launch(LaunchInfo const&)@@Base+0x463>
 17705ee:	e8 fd a5 f3 fe       	call   6aabf0 <std::_Sp_counted_base<(__gnu_cxx::_Lock_policy)2>::_M_release()@plt>
 17705f3:	80 7b 19 04          	cmpb   $0x4,0x19(%rbx)
 17705f7:	76 6c                	jbe    1770665 <QueueComputeScal::launch(LaunchInfo const&)@@Base+0x4d5>
 17705f9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
 1770600:	41 bd 01 00 00 00    	mov    $0x1,%r13d
 1770606:	e9 8c 02 00 00       	jmp    1770897 <QueueComputeScal::launch(LaunchInfo const&)@@Base+0x707>
 177060b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
 1770610:	e8 cb 08 f5 fe       	call   6c0ee0 <std::chrono::_V2::steady_clock::now()@plt>
 1770615:	49 89 c5             	mov    %rax,%r13
 1770618:	e9 c8 fb ff ff       	jmp    17701e5 <QueueComputeScal::launch(LaunchInfo const&)@@Base+0x55>
 177061d:	0f 1f 00             	nopl   (%rax)
 1770620:	e8 bb 08 f5 fe       	call   6c0ee0 <std::chrono::_V2::steady_clock::now()@plt>
 1770625:	48 8b 53 40          	mov    0x40(%rbx),%rdx
 1770629:	4c 29 e8             	sub    %r13,%rax
 177062c:	f0 48 01 82 a0 08 00 	lock add %rax,0x8a0(%rdx)
 1770633:	00 
 1770634:	48 8b 43 40          	mov    0x40(%rbx),%rax
 1770638:	f0 48 83 80 a8 08 00 	lock addq $0x1,0x8a8(%rax)
 177063f:	00 01 
 1770641:	48 8b 43 40          	mov    0x40(%rbx),%rax
 1770645:	48 8b 80 a8 08 00 00 	mov    0x8a8(%rax),%rax
 177064c:	e9 c0 fb ff ff       	jmp    1770211 <QueueComputeScal::launch(LaunchInfo const&)@@Base+0x81>
 1770651:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
 1770658:	80 7b 19 04          	cmpb   $0x4,0x19(%rbx)
 177065c:	4c 8d b5 90 fd ff ff 	lea    -0x270(%rbp),%r14
 1770663:	77 9b                	ja     1770600 <QueueComputeScal::launch(LaunchInfo const&)@@Base+0x470>
 1770665:	48 8b 05 1c b1 3f 02 	mov    0x23fb11c(%rip),%rax        # 3b6b788 <hl_logger::v12_inline_fmt_compile::moduleLoggerData<synapse::LogManager::LogType>@@Base-0x25ca58>
 177066c:	be 01 00 00 00       	mov    $0x1,%esi
 1770671:	48 8b 80 d0 0b 00 00 	mov    0xbd0(%rax),%rax
 1770678:	f0 48 0f c1 70 10    	lock xadd %rsi,0x10(%rax)
 177067e:	0f b6 48 08          	movzbl 0x8(%rax),%ecx
 1770682:	49 89 f4             	mov    %rsi,%r12
 1770685:	4c 89 f7             	mov    %r14,%rdi
 1770688:	49 d3 ec             	shr    %cl,%r12
 177068b:	48 8b 08             	mov    (%rax),%rcx
 177068e:	48 8d 51 ff          	lea    -0x1(%rcx),%rdx
 1770692:	48 21 f2             	and    %rsi,%rdx
 1770695:	48 8d 1c d5 00 00 00 	lea    0x0(,%rdx,8),%rbx
 177069c:	00 
 177069d:	48 29 d3             	sub    %rdx,%rbx
 17706a0:	48 c1 e3 04          	shl    $0x4,%rbx
 17706a4:	48 03 58 18          	add    0x18(%rax),%rbx
 17706a8:	48 89 de             	mov    %rbx,%rsi
 17706ab:	e8 b0 40 f4 fe       	call   6b4760 <containers::hllog_1_0::ConcurrentRecentElementsQueue<hl_logger::v12_inline_fmt_compile::LazyLogItem<64u> >::acquireItem(containers::hllog_1_0::ConcurrentRecentElementsQueue<hl_logger::v12_inline_fmt_compile::LazyLogItem<64u> >::InternalItem const&)@plt>
 17706b0:	44 89 e0             	mov    %r12d,%eax
 17706b3:	2a 43 01             	sub    0x1(%rbx),%al
 17706b6:	3c 03                	cmp    $0x3,%al
 17706b8:	77 7a                	ja     1770734 <QueueComputeScal::launch(LaunchInfo const&)@@Base+0x5a4>
 17706ba:	80 7b 68 00          	cmpb   $0x0,0x68(%rbx)
 17706be:	4c 8d 6b 08          	lea    0x8(%rbx),%r13
 17706c2:	0f 85 27 02 00 00    	jne    17708ef <QueueComputeScal::launch(LaunchInfo const&)@@Base+0x75f>
 17706c8:	c6 43 08 04          	movb   $0x4,0x8(%rbx)
 17706cc:	e8 bf 91 f3 fe       	call   6a9890 <hl_logger::internal::v1_0::getThreadId()@plt>
 17706d1:	48 89 43 10          	mov    %rax,0x10(%rbx)
 17706d5:	e8 b6 6b f4 fe       	call   6b7290 <std::chrono::_V2::system_clock::now()@plt>
 17706da:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
 17706de:	48 8d 7b 28          	lea    0x28(%rbx),%rdi
 17706e2:	48 8d 35 71 ce 9b 01 	lea    0x19bce71(%rip),%rsi        # 312d55a <void std::deque<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > >, std::allocator<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > > >::_M_push_back_aux<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&>(std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&)@@Base+0x9d6a>
 17706e9:	49 89 45 10          	mov    %rax,0x10(%r13)
 17706ed:	48 8d 43 20          	lea    0x20(%rbx),%rax
 17706f1:	48 89 43 60          	mov    %rax,0x60(%rbx)
 17706f5:	48 8d 05 64 d3 31 02 	lea    0x231d364(%rip),%rax        # 3a8da60 <MappedMemMgr::enumNamePoints@@Base+0xfa0>
 17706fc:	c5 fa 7f 43 20       	vmovdqu %xmm0,0x20(%rbx)
 1770701:	48 89 43 20          	mov    %rax,0x20(%rbx)
 1770705:	c5 fa 7f 43 30       	vmovdqu %xmm0,0x30(%rbx)
 177070a:	c5 fa 7f 43 40       	vmovdqu %xmm0,0x40(%rbx)
 177070f:	c5 fa 7f 43 50       	vmovdqu %xmm0,0x50(%rbx)
 1770714:	e8 47 76 ff ff       	call   1767d60 <QueueComputeScal::getMappedMemorySize(unsigned long&) const@@Base+0x5a0>
 1770719:	48 8d 05 44 a9 9b 01 	lea    0x19ba944(%rip),%rax        # 312b064 <void std::deque<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > >, std::allocator<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > > >::_M_push_back_aux<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&>(std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&)@@Base+0x7874>
 1770720:	48 c7 43 48 06 00 00 	movq   $0x6,0x48(%rbx)
 1770727:	00 
 1770728:	48 89 43 50          	mov    %rax,0x50(%rbx)
 177072c:	c6 43 68 01          	movb   $0x1,0x68(%rbx)
 1770730:	44 88 63 01          	mov    %r12b,0x1(%rbx)
 1770734:	80 bd 98 fd ff ff 00 	cmpb   $0x0,-0x268(%rbp)
 177073b:	0f 84 bf fe ff ff    	je     1770600 <QueueComputeScal::launch(LaunchInfo const&)@@Base+0x470>
 1770741:	48 8b 85 90 fd ff ff 	mov    -0x270(%rbp),%rax
 1770748:	c6 00 00             	movb   $0x0,(%rax)
 177074b:	e9 b0 fe ff ff       	jmp    1770600 <QueueComputeScal::launch(LaunchInfo const&)@@Base+0x470>
 1770750:	80 7b 19 04          	cmpb   $0x4,0x19(%rbx)
 1770754:	4c 8d a5 70 fd ff ff 	lea    -0x290(%rbp),%r12
 177075b:	0f 87 37 fd ff ff    	ja     1770498 <QueueComputeScal::launch(LaunchInfo const&)@@Base+0x308>
 1770761:	49 8b 46 18          	mov    0x18(%r14),%rax
 1770765:	be 01 00 00 00       	mov    $0x1,%esi
 177076a:	4c 8b b0 b8 06 00 00 	mov    0x6b8(%rax),%r14
 1770771:	48 8b 05 10 b0 3f 02 	mov    0x23fb010(%rip),%rax        # 3b6b788 <hl_logger::v12_inline_fmt_compile::moduleLoggerData<synapse::LogManager::LogType>@@Base-0x25ca58>
 1770778:	48 8b 80 d0 0b 00 00 	mov    0xbd0(%rax),%rax
 177077f:	f0 48 0f c1 70 10    	lock xadd %rsi,0x10(%rax)
 1770785:	0f b6 48 08          	movzbl 0x8(%rax),%ecx
 1770789:	48 89 f2             	mov    %rsi,%rdx
 177078c:	4c 89 e7             	mov    %r12,%rdi
 177078f:	48 d3 ea             	shr    %cl,%rdx
 1770792:	48 8b 08             	mov    (%rax),%rcx
 1770795:	48 89 95 48 fd ff ff 	mov    %rdx,-0x2b8(%rbp)
 177079c:	48 8d 51 ff          	lea    -0x1(%rcx),%rdx
 17707a0:	48 21 f2             	and    %rsi,%rdx
 17707a3:	48 8d 1c d5 00 00 00 	lea    0x0(,%rdx,8),%rbx
 17707aa:	00 
 17707ab:	48 29 d3             	sub    %rdx,%rbx
 17707ae:	48 c1 e3 04          	shl    $0x4,%rbx
 17707b2:	48 03 58 18          	add    0x18(%rax),%rbx
 17707b6:	48 89 de             	mov    %rbx,%rsi
 17707b9:	e8 a2 3f f4 fe       	call   6b4760 <containers::hllog_1_0::ConcurrentRecentElementsQueue<hl_logger::v12_inline_fmt_compile::LazyLogItem<64u> >::acquireItem(containers::hllog_1_0::ConcurrentRecentElementsQueue<hl_logger::v12_inline_fmt_compile::LazyLogItem<64u> >::InternalItem const&)@plt>
 17707be:	0f b6 85 48 fd ff ff 	movzbl -0x2b8(%rbp),%eax
 17707c5:	2a 43 01             	sub    0x1(%rbx),%al
 17707c8:	3c 03                	cmp    $0x3,%al
 17707ca:	77 6b                	ja     1770837 <QueueComputeScal::launch(LaunchInfo const&)@@Base+0x6a7>
 17707cc:	80 7b 68 00          	cmpb   $0x0,0x68(%rbx)
 17707d0:	4c 8d 63 08          	lea    0x8(%rbx),%r12
 17707d4:	0f 85 26 01 00 00    	jne    1770900 <QueueComputeScal::launch(LaunchInfo const&)@@Base+0x770>
 17707da:	c6 43 08 04          	movb   $0x4,0x8(%rbx)
 17707de:	e8 ad 90 f3 fe       	call   6a9890 <hl_logger::internal::v1_0::getThreadId()@plt>
 17707e3:	48 89 43 10          	mov    %rax,0x10(%rbx)
 17707e7:	e8 a4 6a f4 fe       	call   6b7290 <std::chrono::_V2::system_clock::now()@plt>
 17707ec:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
 17707f0:	49 89 44 24 10       	mov    %rax,0x10(%r12)
 17707f5:	48 8d 43 20          	lea    0x20(%rbx),%rax
 17707f9:	48 89 43 60          	mov    %rax,0x60(%rbx)
 17707fd:	48 8d 05 84 d2 31 02 	lea    0x231d284(%rip),%rax        # 3a8da88 <MappedMemMgr::enumNamePoints@@Base+0xfc8>
 1770804:	48 89 43 20          	mov    %rax,0x20(%rbx)
 1770808:	48 8d 05 55 a8 9b 01 	lea    0x19ba855(%rip),%rax        # 312b064 <void std::deque<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > >, std::allocator<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > > >::_M_push_back_aux<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&>(std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&)@@Base+0x7874>
 177080f:	48 89 43 38          	mov    %rax,0x38(%rbx)
 1770813:	0f b6 85 48 fd ff ff 	movzbl -0x2b8(%rbp),%eax
 177081a:	4c 89 73 28          	mov    %r14,0x28(%rbx)
 177081e:	48 c7 43 30 06 00 00 	movq   $0x6,0x30(%rbx)
 1770825:	00 
 1770826:	c6 43 68 01          	movb   $0x1,0x68(%rbx)
 177082a:	88 43 01             	mov    %al,0x1(%rbx)
 177082d:	c5 fa 7f 43 40       	vmovdqu %xmm0,0x40(%rbx)
 1770832:	c5 fa 7f 43 50       	vmovdqu %xmm0,0x50(%rbx)
 1770837:	80 bd 78 fd ff ff 00 	cmpb   $0x0,-0x288(%rbp)
 177083e:	0f 84 54 fc ff ff    	je     1770498 <QueueComputeScal::launch(LaunchInfo const&)@@Base+0x308>
 1770844:	48 8b 85 70 fd ff ff 	mov    -0x290(%rbp),%rax
 177084b:	c6 00 00             	movb   $0x0,(%rax)
 177084e:	e9 45 fc ff ff       	jmp    1770498 <QueueComputeScal::launch(LaunchInfo const&)@@Base+0x308>
 1770853:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
 1770858:	49 8b 76 30          	mov    0x30(%r14),%rsi
 177085c:	4c 89 e7             	mov    %r12,%rdi
 177085f:	e8 2c da f3 fe       	call   6ae290 <QueueComputeScal::updateSfgEventsScalStream(std::vector<SlotMapItemSptr<EventWithMappedTensor>, std::allocator<SlotMapItemSptr<EventWithMappedTensor> > >&)@plt>
 1770864:	48 8b 85 50 fd ff ff 	mov    -0x2b0(%rbp),%rax
 177086b:	48 8b bd 58 fd ff ff 	mov    -0x2a8(%rbp),%rdi
 1770872:	48 8d b5 90 fd ff ff 	lea    -0x270(%rbp),%rsi
 1770879:	48 89 85 90 fd ff ff 	mov    %rax,-0x270(%rbp)
 1770880:	e8 db d2 f4 fe       	call   6bdb60 <LaunchTracker::add(std::unique_ptr<RecipeLauncherInterface, std::default_delete<RecipeLauncherInterface> >)@plt>
 1770885:	48 8b bd 90 fd ff ff 	mov    -0x270(%rbp),%rdi
 177088c:	48 85 ff             	test   %rdi,%rdi
 177088f:	74 06                	je     1770897 <QueueComputeScal::launch(LaunchInfo const&)@@Base+0x707>
 1770891:	48 8b 07             	mov    (%rdi),%rax
 1770894:	ff 50 08             	call   *0x8(%rax)
 1770897:	4c 89 ff             	mov    %r15,%rdi
 177089a:	e8 91 6d f3 fe       	call   6a7630 <pthread_mutex_unlock@plt>
 177089f:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
 17708a3:	64 48 2b 04 25 28 00 	sub    %fs:0x28,%rax
 17708aa:	00 00 
 17708ac:	75 63                	jne    1770911 <QueueComputeScal::launch(LaunchInfo const&)@@Base+0x781>
 17708ae:	48 8d 65 d8          	lea    -0x28(%rbp),%rsp
 17708b2:	44 89 e8             	mov    %r13d,%eax
 17708b5:	5b                   	pop    %rbx
 17708b6:	41 5c                	pop    %r12
 17708b8:	41 5d                	pop    %r13
 17708ba:	41 5e                	pop    %r14
 17708bc:	41 5f                	pop    %r15
 17708be:	5d                   	pop    %rbp
 17708bf:	c3                   	ret
 17708c0:	48 8b 85 50 fd ff ff 	mov    -0x2b0(%rbp),%rax
 17708c7:	48 8b bd 58 fd ff ff 	mov    -0x2a8(%rbp),%rdi
 17708ce:	48 8d b5 70 fd ff ff 	lea    -0x290(%rbp),%rsi
 17708d5:	48 89 85 70 fd ff ff 	mov    %rax,-0x290(%rbp)
 17708dc:	e8 7f d2 f4 fe       	call   6bdb60 <LaunchTracker::add(std::unique_ptr<RecipeLauncherInterface, std::default_delete<RecipeLauncherInterface> >)@plt>
 17708e1:	48 8b bd 70 fd ff ff 	mov    -0x290(%rbp),%rdi
 17708e8:	48 85 ff             	test   %rdi,%rdi
 17708eb:	75 a4                	jne    1770891 <QueueComputeScal::launch(LaunchInfo const&)@@Base+0x701>
 17708ed:	eb a8                	jmp    1770897 <QueueComputeScal::launch(LaunchInfo const&)@@Base+0x707>
 17708ef:	c6 43 68 00          	movb   $0x0,0x68(%rbx)
 17708f3:	4c 89 ef             	mov    %r13,%rdi
 17708f6:	e8 95 17 f4 fe       	call   6b2090 <hl_logger::v12_inline_fmt_compile::LazyLogItem<64u>::~LazyLogItem()@plt>
 17708fb:	e9 c8 fd ff ff       	jmp    17706c8 <QueueComputeScal::launch(LaunchInfo const&)@@Base+0x538>
 1770900:	c6 43 68 00          	movb   $0x0,0x68(%rbx)
 1770904:	4c 89 e7             	mov    %r12,%rdi
 1770907:	e8 84 17 f4 fe       	call   6b2090 <hl_logger::v12_inline_fmt_compile::LazyLogItem<64u>::~LazyLogItem()@plt>
 177090c:	e9 c9 fe ff ff       	jmp    17707da <QueueComputeScal::launch(LaunchInfo const&)@@Base+0x64a>
 1770911:	e8 2a 86 f3 fe       	call   6a8f40 <__stack_chk_fail@plt>
 1770916:	89 c7                	mov    %eax,%edi
 1770918:	e8 43 bf f3 fe       	call   6ac860 <std::__throw_system_error(int)@plt>
 177091d:	f3 0f 1e fa          	endbr64
 1770921:	49 89 c5             	mov    %rax,%r13
 1770924:	49 89 d4             	mov    %rdx,%r12
 1770927:	e9 c6 ed 10 ff       	jmp    87f6f2 <gaudi::QmansDefinition::getArbitrationMaster(internalStreamType) const@@Base+0x4062a>
 177092c:	f3 0f 1e fa          	endbr64
 1770930:	49 89 c4             	mov    %rax,%r12
 1770933:	e9 77 ee 10 ff       	jmp    87f7af <gaudi::QmansDefinition::getArbitrationMaster(internalStreamType) const@@Base+0x406e7>
 1770938:	f3 0f 1e fa          	endbr64
 177093c:	49 89 c4             	mov    %rax,%r12
 177093f:	e9 a3 ee 10 ff       	jmp    87f7e7 <gaudi::QmansDefinition::getArbitrationMaster(internalStreamType) const@@Base+0x4071f>
 1770944:	f3 0f 1e fa          	endbr64
 1770948:	49 89 c4             	mov    %rax,%r12
 177094b:	e9 cf ee 10 ff       	jmp    87f81f <gaudi::QmansDefinition::getArbitrationMaster(internalStreamType) const@@Base+0x40757>
 1770950:	f3 0f 1e fa          	endbr64
 1770954:	49 89 c4             	mov    %rax,%r12
 1770957:	e9 d9 ee 10 ff       	jmp    87f835 <gaudi::QmansDefinition::getArbitrationMaster(internalStreamType) const@@Base+0x4076d>
 177095c:	f3 0f 1e fa          	endbr64
 1770960:	49 89 c4             	mov    %rax,%r12
 1770963:	c5 f8 77             	vzeroupper
 1770966:	e9 5a ee 10 ff       	jmp    87f7c5 <gaudi::QmansDefinition::getArbitrationMaster(internalStreamType) const@@Base+0x406fd>
 177096b:	f3 0f 1e fa          	endbr64
 177096f:	49 89 c4             	mov    %rax,%r12
 1770972:	e9 87 ee 10 ff       	jmp    87f7fe <gaudi::QmansDefinition::getArbitrationMaster(internalStreamType) const@@Base+0x40736>
 1770977:	f3 0f 1e fa          	endbr64
 177097b:	49 89 c4             	mov    %rax,%r12
 177097e:	c5 f8 77             	vzeroupper
 1770981:	e9 4c ee 10 ff       	jmp    87f7d2 <gaudi::QmansDefinition::getArbitrationMaster(internalStreamType) const@@Base+0x4070a>
 1770986:	f3 0f 1e fa          	endbr64
 177098a:	48 89 95 30 fd ff ff 	mov    %rdx,-0x2d0(%rbp)
 1770991:	49 89 c0             	mov    %rax,%r8
 1770994:	e9 b8 ee 10 ff       	jmp    87f851 <gaudi::QmansDefinition::getArbitrationMaster(internalStreamType) const@@Base+0x40789>
