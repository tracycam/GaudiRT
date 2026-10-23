
/usr/lib/habanalabs/libSynapse.so:     file format elf64-x86-64


Disassembly of section .text:

000000000171e1c0 <RecipeLauncher::checkCompletionCopy(unsigned long)@@Base+0xce0>:
 171e1c0:	e9 12 93 15 ff       	jmp    8774d7 <gaudi::QmansDefinition::getArbitrationMaster(internalStreamType) const@@Base+0x3840f>
 171e1c5:	f3 0f 1e fa          	endbr64
 171e1c9:	48 89 c3             	mov    %rax,%rbx
 171e1cc:	c5 f8 77             	vzeroupper
 171e1cf:	e9 b3 92 15 ff       	jmp    877487 <gaudi::QmansDefinition::getArbitrationMaster(internalStreamType) const@@Base+0x383bf>
 171e1d4:	f3 0f 1e fa          	endbr64
 171e1d8:	49 89 c7             	mov    %rax,%r15
 171e1db:	49 89 d6             	mov    %rdx,%r14
 171e1de:	e9 b6 93 15 ff       	jmp    877599 <gaudi::QmansDefinition::getArbitrationMaster(internalStreamType) const@@Base+0x384d1>
 171e1e3:	f3 0f 1e fa          	endbr64
 171e1e7:	49 89 c4             	mov    %rax,%r12
 171e1ea:	e9 72 94 15 ff       	jmp    877661 <gaudi::QmansDefinition::getArbitrationMaster(internalStreamType) const@@Base+0x38599>
 171e1ef:	90                   	nop

000000000171e1f0 <RecipeLauncher::checkCompletionCompute(unsigned long)@@Base>:
 171e1f0:	f3 0f 1e fa          	endbr64
 171e1f4:	55                   	push   %rbp
 171e1f5:	48 89 e5             	mov    %rsp,%rbp
 171e1f8:	41 57                	push   %r15
 171e1fa:	49 89 f7             	mov    %rsi,%r15
 171e1fd:	41 56                	push   %r14
 171e1ff:	41 55                	push   %r13
 171e201:	49 89 fd             	mov    %rdi,%r13
 171e204:	41 54                	push   %r12
 171e206:	53                   	push   %rbx
 171e207:	48 81 ec 38 03 00 00 	sub    $0x338,%rsp
 171e20e:	48 8b 0d 4b 6d 44 02 	mov    0x2446d4b(%rip),%rcx        # 3b64f60 <hl_logger::v12_inline_fmt_compile::ModuleLoggerData<synapse::LogManager::LogType>::levels@@Base-0x7540>
 171e215:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
 171e21c:	00 00 
 171e21e:	48 89 45 c8          	mov    %rax,-0x38(%rbp)
 171e222:	48 8b 07             	mov    (%rdi),%rax
 171e225:	80 79 18 00          	cmpb   $0x0,0x18(%rcx)
 171e229:	0f 85 71 03 00 00    	jne    171e5a0 <RecipeLauncher::checkCompletionCompute(unsigned long)@@Base+0x3b0>
 171e22f:	48 8d bd 70 fd ff ff 	lea    -0x290(%rbp),%rdi
 171e236:	4c 89 ee             	mov    %r13,%rsi
 171e239:	48 8d 9d 80 fd ff ff 	lea    -0x280(%rbp),%rbx
 171e240:	ff 50 28             	call   *0x28(%rax)
 171e243:	48 8d 85 a0 fd ff ff 	lea    -0x260(%rbp),%rax
 171e24a:	48 89 9d f0 fc ff ff 	mov    %rbx,-0x310(%rbp)
 171e251:	48 89 85 f8 fc ff ff 	mov    %rax,-0x308(%rbp)
 171e258:	48 89 85 90 fd ff ff 	mov    %rax,-0x270(%rbp)
 171e25f:	48 8b 85 70 fd ff ff 	mov    -0x290(%rbp),%rax
 171e266:	48 39 d8             	cmp    %rbx,%rax
 171e269:	0f 84 f9 0a 00 00    	je     171ed68 <RecipeLauncher::checkCompletionCompute(unsigned long)@@Base+0xb78>
 171e26f:	48 89 85 90 fd ff ff 	mov    %rax,-0x270(%rbp)
 171e276:	48 8b 85 80 fd ff ff 	mov    -0x280(%rbp),%rax
 171e27d:	48 89 85 a0 fd ff ff 	mov    %rax,-0x260(%rbp)
 171e284:	48 8b 85 78 fd ff ff 	mov    -0x288(%rbp),%rax
 171e28b:	4c 8d b5 10 fd ff ff 	lea    -0x2f0(%rbp),%r14
 171e292:	be 0c 00 00 00       	mov    $0xc,%esi
 171e297:	48 c7 85 78 fd ff ff 	movq   $0x0,-0x288(%rbp)
 171e29e:	00 00 00 00 
 171e2a2:	4c 89 f7             	mov    %r14,%rdi
 171e2a5:	c6 85 80 fd ff ff 00 	movb   $0x0,-0x280(%rbp)
 171e2ac:	48 89 85 98 fd ff ff 	mov    %rax,-0x268(%rbp)
 171e2b3:	48 8b 85 f0 fc ff ff 	mov    -0x310(%rbp),%rax
 171e2ba:	48 89 85 70 fd ff ff 	mov    %rax,-0x290(%rbp)
 171e2c1:	e8 ba 0f fa fe       	call   6bf280 <std::shared_ptr<hl_logger::Logger> hl_logger::v12_inline_fmt_compile::getLogger<synapse::LogManager::LogType>(synapse::LogManager::LogType)@plt>
 171e2c6:	48 8d 0d bb df 39 02 	lea    0x239dfbb(%rip),%rcx        # 3abc288 <typeinfo for hl_gcfg::inline_ver_1_11::GcfgItem@@Base+0x18>
 171e2cd:	48 8d 05 bc 48 a5 01 	lea    0x1a548bc(%rip),%rax        # 3172b90 <void std::deque<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > >, std::allocator<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > > >::_M_push_back_aux<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&>(std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&)@@Base+0x4f3a0>
 171e2d4:	48 8d 35 a5 47 a5 01 	lea    0x1a547a5(%rip),%rsi        # 3172a80 <void std::deque<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > >, std::allocator<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > > >::_M_push_back_aux<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&>(std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&)@@Base+0x4f290>
 171e2db:	ba 0f 00 00 00       	mov    $0xf,%edx
 171e2e0:	48 89 85 d8 fc ff ff 	mov    %rax,-0x328(%rbp)
 171e2e7:	48 8d 85 d0 fd ff ff 	lea    -0x230(%rbp),%rax
 171e2ee:	48 8d bd b0 fd ff ff 	lea    -0x250(%rbp),%rdi
 171e2f5:	48 8d 59 10          	lea    0x10(%rcx),%rbx
 171e2f9:	48 89 85 b0 fc ff ff 	mov    %rax,-0x350(%rbp)
 171e300:	c4 e1 f9 6e c3       	vmovq  %rbx,%xmm0
 171e305:	48 89 8d b8 fc ff ff 	mov    %rcx,-0x348(%rbp)
 171e30c:	c4 e3 f9 22 c0 01    	vpinsrq $0x1,%rax,%xmm0,%xmm0
 171e312:	48 8d 46 11          	lea    0x11(%rsi),%rax
 171e316:	48 c7 85 d0 fc ff ff 	movq   $0x13,-0x330(%rbp)
 171e31d:	13 00 00 00 
 171e321:	c5 f9 7f 85 b0 fd ff 	vmovdqa %xmm0,-0x250(%rbp)
 171e328:	ff 
 171e329:	c5 f9 6f 05 df 93 ad 	vmovdqa 0x1ad93df(%rip),%xmm0        # 31f7710 <std::__detail::__to_chars_10_impl<unsigned long>(char*, unsigned int, unsigned long)::__digits@@Base+0xd0>
 171e330:	01 
 171e331:	48 89 b5 40 fd ff ff 	mov    %rsi,-0x2c0(%rbp)
 171e338:	48 c7 85 48 fd ff ff 	movq   $0xf,-0x2b8(%rbp)
 171e33f:	0f 00 00 00 
 171e343:	48 89 85 58 fd ff ff 	mov    %rax,-0x2a8(%rbp)
 171e34a:	48 c7 85 60 fd ff ff 	movq   $0x1e,-0x2a0(%rbp)
 171e351:	1e 00 00 00 
 171e355:	c5 f9 7f 85 c0 fd ff 	vmovdqa %xmm0,-0x240(%rbp)
 171e35c:	ff 
 171e35d:	e8 2e 8f 49 ff       	call   bb7290 <std::_Rb_tree<std::pair<unsigned int, std::shared_ptr<BasicFieldInfo> >, std::pair<unsigned int, std::shared_ptr<BasicFieldInfo> >, std::_Identity<std::pair<unsigned int, std::shared_ptr<BasicFieldInfo> > >, PairComparator<unsigned int, std::shared_ptr<BasicFieldInfo> >, std::allocator<std::pair<unsigned int, std::shared_ptr<BasicFieldInfo> > > >::_M_get_insert_hint_unique_pos(std::_Rb_tree_const_iterator<std::pair<unsigned int, std::shared_ptr<BasicFieldInfo> > >, std::pair<unsigned int, std::shared_ptr<BasicFieldInfo> > const&)@@Base+0xc770>
 171e362:	48 8b b5 90 fd ff ff 	mov    -0x270(%rbp),%rsi
 171e369:	48 8b 95 98 fd ff ff 	mov    -0x268(%rbp),%rdx
 171e370:	48 89 c7             	mov    %rax,%rdi
 171e373:	e8 18 8f 49 ff       	call   bb7290 <std::_Rb_tree<std::pair<unsigned int, std::shared_ptr<BasicFieldInfo> >, std::pair<unsigned int, std::shared_ptr<BasicFieldInfo> >, std::_Identity<std::pair<unsigned int, std::shared_ptr<BasicFieldInfo> > >, PairComparator<unsigned int, std::shared_ptr<BasicFieldInfo> >, std::allocator<std::pair<unsigned int, std::shared_ptr<BasicFieldInfo> > > >::_M_get_insert_hint_unique_pos(std::_Rb_tree_const_iterator<std::pair<unsigned int, std::shared_ptr<BasicFieldInfo> > >, std::pair<unsigned int, std::shared_ptr<BasicFieldInfo> > const&)@@Base+0xc770>
 171e378:	48 8b b5 58 fd ff ff 	mov    -0x2a8(%rbp),%rsi
 171e37f:	48                   	rex.W

/usr/lib/habanalabs/libSynapse.so:     file format elf64-x86-64


Disassembly of section .text:

0000000001720a00 <RecipeLauncher::dfaLogDescription(bool, unsigned long, bool, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, bool) const@@Base+0x1620>:
 1720a00:	e8 8b 68 f9 fe       	call   6b7290 <std::chrono::_V2::system_clock::now()@plt>
 1720a05:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
 1720a09:	49 89 45 10          	mov    %rax,0x10(%r13)
 1720a0d:	48 8d 43 20          	lea    0x20(%rbx),%rax
 1720a11:	48 89 43 60          	mov    %rax,0x60(%rbx)
 1720a15:	48 8d 05 74 c3 36 02 	lea    0x236c374(%rip),%rax        # 3a8cd90 <MappedMemMgr::enumNamePoints@@Base+0x2d0>
 1720a1c:	48 89 43 20          	mov    %rax,0x20(%rbx)
 1720a20:	48 8d 43 38          	lea    0x38(%rbx),%rax
 1720a24:	48 89 43 28          	mov    %rax,0x28(%rbx)
 1720a28:	c5 fa 7f 43 30       	vmovdqu %xmm0,0x30(%rbx)
 1720a2d:	c5 fa 7f 43 40       	vmovdqu %xmm0,0x40(%rbx)
 1720a32:	c5 fa 7f 43 50       	vmovdqu %xmm0,0x50(%rbx)
 1720a37:	48 8b 85 b0 fd ff ff 	mov    -0x250(%rbp),%rax
 1720a3e:	48 3b 85 e0 f3 ff ff 	cmp    -0xc20(%rbp),%rax
 1720a45:	0f 84 56 0e 00 00    	je     17218a1 <RecipeLauncher::dfaLogDescription(bool, unsigned long, bool, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, bool) const@@Base+0x24c1>
 1720a4b:	48 89 43 28          	mov    %rax,0x28(%rbx)
 1720a4f:	48 8b 85 c0 fd ff ff 	mov    -0x240(%rbp),%rax
 1720a56:	48 89 43 38          	mov    %rax,0x38(%rbx)
 1720a5a:	48 8b 85 b8 fd ff ff 	mov    -0x248(%rbp),%rax
 1720a61:	48 89 43 30          	mov    %rax,0x30(%rbx)
 1720a65:	48 8b 85 e0 f3 ff ff 	mov    -0xc20(%rbp),%rax
 1720a6c:	48 c7 85 b8 fd ff ff 	movq   $0x0,-0x248(%rbp)
 1720a73:	00 00 00 00 
 1720a77:	48 89 85 b0 fd ff ff 	mov    %rax,-0x250(%rbp)
 1720a7e:	c6                   	.byte 0xc6
 1720a7f:	85                   	.byte 0x85
