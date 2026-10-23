
/usr/lib/habanalabs/libSynapse.so:     file format elf64-x86-64


Disassembly of section .text:

0000000001724810 <RecipeLauncher::launch(LaunchInfo const&)@@Base>:
 1724810:	f3 0f 1e fa          	endbr64
 1724814:	55                   	push   %rbp
 1724815:	48 89 e5             	mov    %rsp,%rbp
 1724818:	41 57                	push   %r15
 172481a:	49 89 ff             	mov    %rdi,%r15
 172481d:	41 56                	push   %r14
 172481f:	49 89 f6             	mov    %rsi,%r14
 1724822:	41 55                	push   %r13
 1724824:	41 54                	push   %r12
 1724826:	53                   	push   %rbx
 1724827:	48 8d 9d b0 fb ff ff 	lea    -0x450(%rbp),%rbx
 172482e:	48 81 ec c8 06 00 00 	sub    $0x6c8,%rsp
 1724835:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
 172483c:	00 00 
 172483e:	48 89 45 c8          	mov    %rax,-0x38(%rbp)
 1724842:	48 8d 05 3f 7a 39 02 	lea    0x2397a3f(%rip),%rax        # 3abc288 <typeinfo for hl_gcfg::inline_ver_1_11::GcfgItem@@Base+0x18>
 1724849:	4c 8d 60 10          	lea    0x10(%rax),%r12
 172484d:	48 8b 05 0c 07 44 02 	mov    0x244070c(%rip),%rax        # 3b64f60 <hl_logger::v12_inline_fmt_compile::ModuleLoggerData<synapse::LogManager::LogType>::levels@@Base-0x7540>
 1724854:	c4 c1 f9 6e d4       	vmovq  %r12,%xmm2
 1724859:	80 78 18 00          	cmpb   $0x0,0x18(%rax)
 172485d:	c4 e3 e9 22 c3 01    	vpinsrq $0x1,%rbx,%xmm2,%xmm0
 1724863:	0f 85 17 03 00 00    	jne    1724b80 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x370>
 1724869:	48 8b b7 38 01 00 00 	mov    0x138(%rdi),%rsi
 1724870:	48 8b 87 40 01 00 00 	mov    0x140(%rdi),%rax
 1724877:	48 8d 8d 00 fa ff ff 	lea    -0x600(%rbp),%rcx
 172487e:	c5 f9 7f 85 40 f9 ff 	vmovdqa %xmm0,-0x6c0(%rbp)
 1724885:	ff 
 1724886:	48 89 cf             	mov    %rcx,%rdi
 1724889:	48 89 8d 80 f9 ff ff 	mov    %rcx,-0x680(%rbp)
 1724890:	4c 8d ad 90 fb ff ff 	lea    -0x470(%rbp),%r13
 1724897:	48 89 b5 68 f9 ff ff 	mov    %rsi,-0x698(%rbp)
 172489e:	be 0c 00 00 00       	mov    $0xc,%esi
 17248a3:	48 89 85 70 f9 ff ff 	mov    %rax,-0x690(%rbp)
 17248aa:	e8 d1 a9 f9 fe       	call   6bf280 <std::shared_ptr<hl_logger::Logger> hl_logger::v12_inline_fmt_compile::getLogger<synapse::LogManager::LogType>(synapse::LogManager::LogType)@plt>
 17248af:	48 8d 0d da e2 a4 01 	lea    0x1a4e2da(%rip),%rcx        # 3172b90 <void std::deque<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > >, std::allocator<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > > >::_M_push_back_aux<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&>(std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&)@@Base+0x4f3a0>
 17248b6:	31 c0                	xor    %eax,%eax
 17248b8:	48 8d bd 00 fb ff ff 	lea    -0x500(%rbp),%rdi
 17248bf:	48 89 8d 58 f9 ff ff 	mov    %rcx,-0x6a8(%rbp)
 17248c6:	b9 12 00 00 00       	mov    $0x12,%ecx
 17248cb:	c5 f9 6f 85 40 f9 ff 	vmovdqa -0x6c0(%rbp),%xmm0
 17248d2:	ff 
 17248d3:	48 89 da             	mov    %rbx,%rdx
 17248d6:	f3 48 ab             	rep stos %rax,(%rdi)
 17248d9:	48 8d 3d 98 e8 a4 01 	lea    0x1a4e898(%rip),%rdi        # 3173178 <void std::deque<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > >, std::allocator<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > > >::_M_push_back_aux<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&>(std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&)@@Base+0x4f988>
 17248e0:	48 c7 85 50 f9 ff ff 	movq   $0x13,-0x6b0(%rbp)
 17248e7:	13 00 00 00 
 17248eb:	c5 f9 7f 85 90 fb ff 	vmovdqa %xmm0,-0x470(%rbp)
 17248f2:	ff 
 17248f3:	48 8d 4f 14          	lea    0x14(%rdi),%rcx
 17248f7:	48 8d 77 10          	lea    0x10(%rdi),%rsi
 17248fb:	c5 f9 6f 05 8d 2e ad 	vmovdqa 0x1ad2e8d(%rip),%xmm0        # 31f7790 <std::__detail::__to_chars_10_impl<unsigned long>(char*, unsigned int, unsigned long)::__digits@@Base+0x150>
 1724902:	01 
 1724903:	48 89 bd f0 fa ff ff 	mov    %rdi,-0x510(%rbp)
 172490a:	48 c7 85 f8 fa ff ff 	movq   $0x10,-0x508(%rbp)
 1724911:	10 00 00 00 
 1724915:	c7 85 04 fb ff ff ff 	movl   $0xffffffff,-0x4fc(%rbp)
 172491c:	ff ff ff 
 172491f:	c6 85 08 fb ff ff 03 	movb   $0x3,-0x4f8(%rbp)
 1724926:	c6 85 0b fb ff ff 20 	movb   $0x20,-0x4f5(%rbp)
 172492d:	c6 85 0f fb ff ff 01 	movb   $0x1,-0x4f1(%rbp)
 1724934:	48 89 8d 40 fb ff ff 	mov    %rcx,-0x4c0(%rbp)
 172493b:	48 c7 85 48 fb ff ff 	movq   $0xb,-0x4b8(%rbp)
 1724942:	0b 00 00 00 
 1724946:	c7 85 54 fb ff ff ff 	movl   $0xffffffff,-0x4ac(%rbp)
 172494d:	ff ff ff 
 1724950:	c6 85 58 fb ff ff 03 	movb   $0x3,-0x4a8(%rbp)
 1724957:	c6 85 5b fb ff ff 20 	movb   $0x20,-0x4a5(%rbp)
 172495e:	c6 85 5f fb ff ff 01 	movb   $0x1,-0x4a1(%rbp)
 1724965:	c5 f9 7f 85 a0 fb ff 	vmovdqa %xmm0,-0x460(%rbp)
 172496c:	ff 
 172496d:	e8 8e 58 fe ff       	call   170a200 <NBuffQueueAllocator::getNumBuffers()@@Base+0xed0>
 1724972:	48 8b 85 70 f9 ff ff 	mov    -0x690(%rbp),%rax
 1724979:	48 8b b5 68 f9 ff ff 	mov    -0x698(%rbp),%rsi
 1724980:	48 8d 95 c0 f9 ff ff 	lea    -0x640(%rbp),%rdx
 1724987:	48 8d bd 00 fb ff ff 	lea    -0x500(%rbp),%rdi
 172498e:	4c 89 ad c0 f9 ff ff 	mov    %r13,-0x640(%rbp)
 1724995:	48 89 85 30 fa ff ff 	mov    %rax,-0x5d0(%rbp)
 172499c:	48 8d 85 20 fa ff ff 	lea    -0x5e0(%rbp),%rax
 17249a3:	48 89 b5 20 fa ff ff 	mov    %rsi,-0x5e0(%rbp)
 17249aa:	48 c7 85 c8 f9 ff ff 	movq   $0x44,-0x638(%rbp)
 17249b1:	44 00 00 00 
 17249b5:	48 89 85 d0 f9 ff ff 	mov    %rax,-0x630(%rbp)
 17249bc:	48 c7 85 d8 f9 ff ff 	movq   $0x0,-0x628(%rbp)
 17249c3:	00 00 00 00 
 17249c7:	e8 74 d5 fe ff       	call   1711f40 <validateEvents(std::vector<SlotMapItemSptr<EventWithMappedTensor>, std::allocator<SlotMapItemSptr<EventWithMappedTensor> > >&)@@Base+0x560>
 17249cc:	49 89 c5             	mov    %rax,%r13
 17249cf:	48 8b 85 40 fb ff ff 	mov    -0x4c0(%rbp),%rax
 17249d6:	49 8b 4d 10          	mov    0x10(%r13),%rcx
 17249da:	49 8b 55 18          	mov    0x18(%r13),%rdx
 17249de:	48 89 85 20 f9 ff ff 	mov    %rax,-0x6e0(%rbp)
 17249e5:	48 8b 85 48 fb ff ff 	mov    -0x4b8(%rbp),%rax
 17249ec:	48 89 8d 28 f9 ff ff 	mov    %rcx,-0x6d8(%rbp)
 17249f3:	48 89 85 18 f9 ff ff 	mov    %rax,-0x6e8(%rbp)
 17249fa:	48 01 c1             	add    %rax,%rcx
 17249fd:	49 8b 45 08          	mov    0x8(%r13),%rax
 1724a01:	48 89 8d 30 f9 ff ff 	mov    %rcx,-0x6d0(%rbp)
 1724a08:	48 89 95 60 f9 ff ff 	mov    %rdx,-0x6a0(%rbp)
 1724a0f:	48 89 85 40 f9 ff ff 	mov    %rax,-0x6c0(%rbp)
 1724a16:	48 39 d1             	cmp    %rdx,%rcx
 1724a19:	0f 87 d1 07 00 00    	ja     17251f0 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x9e0>
 1724a1f:	48 8b 8d 60 f9 ff ff 	mov    -0x6a0(%rbp),%rcx
 1724a26:	48 8b 85 30 f9 ff ff 	mov    -0x6d0(%rbp),%rax
 1724a2d:	48 8b bd 20 f9 ff ff 	mov    -0x6e0(%rbp),%rdi
 1724a34:	48 8b b5 18 f9 ff ff 	mov    -0x6e8(%rbp),%rsi
 1724a3b:	48 39 c1             	cmp    %rax,%rcx
 1724a3e:	48 8b 95 40 f9 ff ff 	mov    -0x6c0(%rbp),%rdx
 1724a45:	48 0f 46 c1          	cmovbe %rcx,%rax
 1724a49:	48 01 fe             	add    %rdi,%rsi
 1724a4c:	48 03 95 28 f9 ff ff 	add    -0x6d8(%rbp),%rdx
 1724a53:	49 89 45 10          	mov    %rax,0x10(%r13)
 1724a57:	e8 a4 57 fe ff       	call   170a200 <NBuffQueueAllocator::getNumBuffers()@@Base+0xed0>
 1724a5c:	48 8b 85 68 f9 ff ff 	mov    -0x698(%rbp),%rax
 1724a63:	48 8b b5 70 f9 ff ff 	mov    -0x690(%rbp),%rsi
 1724a6a:	48 8d 95 e0 f9 ff ff 	lea    -0x620(%rbp),%rdx
 1724a71:	48 8d bd 50 fb ff ff 	lea    -0x4b0(%rbp),%rdi
 1724a78:	4c 89 ad e0 f9 ff ff 	mov    %r13,-0x620(%rbp)
 1724a7f:	48 89 85 40 fa ff ff 	mov    %rax,-0x5c0(%rbp)
 1724a86:	48 8d 85 40 fa ff ff 	lea    -0x5c0(%rbp),%rax
 1724a8d:	48 89 b5 50 fa ff ff 	mov    %rsi,-0x5b0(%rbp)
 1724a94:	48 c7 85 e8 f9 ff ff 	movq   $0x44,-0x618(%rbp)
 1724a9b:	44 00 00 00 
 1724a9f:	48 89 85 f0 f9 ff ff 	mov    %rax,-0x610(%rbp)
 1724aa6:	48 c7 85 f8 f9 ff ff 	movq   $0x0,-0x608(%rbp)
 1724aad:	00 00 00 00 
 1724ab1:	e8 8a d4 fe ff       	call   1711f40 <validateEvents(std::vector<SlotMapItemSptr<EventWithMappedTensor>, std::allocator<SlotMapItemSptr<EventWithMappedTensor> > >&)@@Base+0x560>
 1724ab6:	48 83 ec 08          	sub    $0x8,%rsp
 1724aba:	48 8b 8d a0 fb ff ff 	mov    -0x460(%rbp),%rcx
 1724ac1:	31 d2                	xor    %edx,%edx
 1724ac3:	31 f6                	xor    %esi,%esi
 1724ac5:	6a 00                	push   $0x0
 1724ac7:	4c 8b 85 98 fb ff ff 	mov    -0x468(%rbp),%r8
 1724ace:	41 b9 77 00 00 00    	mov    $0x77,%r9d
 1724ad4:	ff b5 58 f9 ff ff    	push   -0x6a8(%rbp)
 1724ada:	48 8b bd 80 f9 ff ff 	mov    -0x680(%rbp),%rdi
 1724ae1:	ff b5 50 f9 ff ff    	push   -0x6b0(%rbp)
 1724ae7:	e8 c4 15 f8 fe       	call   6a60b0 <hl_logger::v1_0::log(std::shared_ptr<hl_logger::Logger> const&, int, bool, std::basic_string_view<char, std::char_traits<char> >, std::basic_string_view<char, std::char_traits<char> >, int, bool)@plt>
 1724aec:	48 8d 05 95 77 39 02 	lea    0x2397795(%rip),%rax        # 3abc288 <typeinfo for hl_gcfg::inline_ver_1_11::GcfgItem@@Base+0x18>
 1724af3:	48 8b bd 98 fb ff ff 	mov    -0x468(%rbp),%rdi
 1724afa:	48 83 c4 20          	add    $0x20,%rsp
 1724afe:	48 83 c0 10          	add    $0x10,%rax
 1724b02:	48 89 85 90 fb ff ff 	mov    %rax,-0x470(%rbp)
 1724b09:	48 39 df             	cmp    %rbx,%rdi
 1724b0c:	74 0c                	je     1724b1a <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x30a>
 1724b0e:	48 8b b5 a8 fb ff ff 	mov    -0x458(%rbp),%rsi
 1724b15:	e8 66 e1 f7 fe       	call   6a2c80 <operator delete(void*, unsigned long)@plt>
 1724b1a:	48 8b bd 08 fa ff ff 	mov    -0x5f8(%rbp),%rdi
 1724b21:	48 85 ff             	test   %rdi,%rdi
 1724b24:	74 05                	je     1724b2b <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x31b>
 1724b26:	e8 c5 60 f8 fe       	call   6aabf0 <std::_Sp_counted_base<(__gnu_cxx::_Lock_policy)2>::_M_release()@plt>
 1724b2b:	48 8b 05 2e 04 44 02 	mov    0x244042e(%rip),%rax        # 3b64f60 <hl_logger::v12_inline_fmt_compile::ModuleLoggerData<synapse::LogManager::LogType>::levels@@Base-0x7540>
 1724b32:	80 78 19 00          	cmpb   $0x0,0x19(%rax)
 1724b36:	74 5c                	je     1724b94 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x384>
 1724b38:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
 1724b3f:	00 
 1724b40:	49 8b 7e 30          	mov    0x30(%r14),%rdi
 1724b44:	e8 b7 fa f9 fe       	call   6c4600 <validateEvents(std::vector<SlotMapItemSptr<EventWithMappedTensor>, std::allocator<SlotMapItemSptr<EventWithMappedTensor> > >&)@plt>
 1724b49:	41 89 c5             	mov    %eax,%r13d
 1724b4c:	85 c0                	test   %eax,%eax
 1724b4e:	0f 84 5c 01 00 00    	je     1724cb0 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x4a0>
 1724b54:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
 1724b58:	64 48 2b 04 25 28 00 	sub    %fs:0x28,%rax
 1724b5f:	00 00 
 1724b61:	0f 85 fe 07 00 00    	jne    1725365 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0xb55>
 1724b67:	48 8d 65 d8          	lea    -0x28(%rbp),%rsp
 1724b6b:	44 89 e8             	mov    %r13d,%eax
 1724b6e:	5b                   	pop    %rbx
 1724b6f:	41 5c                	pop    %r12
 1724b71:	41 5d                	pop    %r13
 1724b73:	41 5e                	pop    %r14
 1724b75:	41 5f                	pop    %r15
 1724b77:	5d                   	pop    %rbp
 1724b78:	c3                   	ret
 1724b79:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
 1724b80:	80 78 19 00          	cmpb   $0x0,0x19(%rax)
 1724b84:	75 ba                	jne    1724b40 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x330>
 1724b86:	48 8d 85 00 fa ff ff 	lea    -0x600(%rbp),%rax
 1724b8d:	48 89 85 80 f9 ff ff 	mov    %rax,-0x680(%rbp)
 1724b94:	49 8b 87 40 01 00 00 	mov    0x140(%r15),%rax
 1724b9b:	be 01 00 00 00       	mov    $0x1,%esi
 1724ba0:	48 89 85 70 f9 ff ff 	mov    %rax,-0x690(%rbp)
 1724ba7:	49 8b 87 38 01 00 00 	mov    0x138(%r15),%rax
 1724bae:	48 89 85 68 f9 ff ff 	mov    %rax,-0x698(%rbp)
 1724bb5:	48 8b 05 cc 6b 44 02 	mov    0x2446bcc(%rip),%rax        # 3b6b788 <hl_logger::v12_inline_fmt_compile::moduleLoggerData<synapse::LogManager::LogType>@@Base-0x25ca58>
 1724bbc:	48 8b 80 d0 0b 00 00 	mov    0xbd0(%rax),%rax
 1724bc3:	f0 48 0f c1 70 10    	lock xadd %rsi,0x10(%rax)
 1724bc9:	0f b6 48 08          	movzbl 0x8(%rax),%ecx
 1724bcd:	48 8b bd 80 f9 ff ff 	mov    -0x680(%rbp),%rdi
 1724bd4:	49 89 f5             	mov    %rsi,%r13
 1724bd7:	49 d3 ed             	shr    %cl,%r13
 1724bda:	48 8b 08             	mov    (%rax),%rcx
 1724bdd:	48 8d 51 ff          	lea    -0x1(%rcx),%rdx
 1724be1:	48 21 f2             	and    %rsi,%rdx
 1724be4:	48 8d 1c d5 00 00 00 	lea    0x0(,%rdx,8),%rbx
 1724beb:	00 
 1724bec:	48 29 d3             	sub    %rdx,%rbx
 1724bef:	48 c1 e3 04          	shl    $0x4,%rbx
 1724bf3:	48 03 58 18          	add    0x18(%rax),%rbx
 1724bf7:	48 89 de             	mov    %rbx,%rsi
 1724bfa:	e8 61 fb f8 fe       	call   6b4760 <containers::hllog_1_0::ConcurrentRecentElementsQueue<hl_logger::v12_inline_fmt_compile::LazyLogItem<64u> >::acquireItem(containers::hllog_1_0::ConcurrentRecentElementsQueue<hl_logger::v12_inline_fmt_compile::LazyLogItem<64u> >::InternalItem const&)@plt>
 1724bff:	44 89 e8             	mov    %r13d,%eax
 1724c02:	2a 43 01             	sub    0x1(%rbx),%al
 1724c05:	3c 03                	cmp    $0x3,%al
 1724c07:	0f 87 87 00 00 00    	ja     1724c94 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x484>
 1724c0d:	48 8d 43 08          	lea    0x8(%rbx),%rax
 1724c11:	80 7b 68 00          	cmpb   $0x0,0x68(%rbx)
 1724c15:	48 89 85 80 f9 ff ff 	mov    %rax,-0x680(%rbp)
 1724c1c:	0f 85 52 06 00 00    	jne    1725274 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0xa64>
 1724c22:	c6 43 08 00          	movb   $0x0,0x8(%rbx)
 1724c26:	e8 65 4c f8 fe       	call   6a9890 <hl_logger::internal::v1_0::getThreadId()@plt>
 1724c2b:	48 89 43 10          	mov    %rax,0x10(%rbx)
 1724c2f:	c5 fa 7e a5 70 f9 ff 	vmovq  -0x690(%rbp),%xmm4
 1724c36:	ff 
 1724c37:	c4 e3 d9 22 85 68 f9 	vpinsrq $0x1,-0x698(%rbp),%xmm4,%xmm0
 1724c3e:	ff ff 01 
 1724c41:	c5 f9 7f 85 70 f9 ff 	vmovdqa %xmm0,-0x690(%rbp)
 1724c48:	ff 
 1724c49:	e8 42 26 f9 fe       	call   6b7290 <std::chrono::_V2::system_clock::now()@plt>
 1724c4e:	48 8b 8d 80 f9 ff ff 	mov    -0x680(%rbp),%rcx
 1724c55:	c5 f1 ef c9          	vpxor  %xmm1,%xmm1,%xmm1
 1724c59:	c5 f9 6f 85 70 f9 ff 	vmovdqa -0x690(%rbp),%xmm0
 1724c60:	ff 
 1724c61:	48 89 41 10          	mov    %rax,0x10(%rcx)
 1724c65:	48 8d 43 20          	lea    0x20(%rbx),%rax
 1724c69:	48 89 43 60          	mov    %rax,0x60(%rbx)
 1724c6d:	48 8d 05 dc 7f 36 02 	lea    0x2367fdc(%rip),%rax        # 3a8cc50 <MappedMemMgr::enumNamePoints@@Base+0x190>
 1724c74:	48 89 43 20          	mov    %rax,0x20(%rbx)
 1724c78:	c6 43 68 01          	movb   $0x1,0x68(%rbx)
 1724c7c:	44 88 6b 01          	mov    %r13b,0x1(%rbx)
 1724c80:	c5 fa 7f 4b 30       	vmovdqu %xmm1,0x30(%rbx)
 1724c85:	c5 fa 7f 4b 40       	vmovdqu %xmm1,0x40(%rbx)
 1724c8a:	c5 fa 7f 4b 50       	vmovdqu %xmm1,0x50(%rbx)
 1724c8f:	c5 fa 7f 43 28       	vmovdqu %xmm0,0x28(%rbx)
 1724c94:	80 bd 08 fa ff ff 00 	cmpb   $0x0,-0x5f8(%rbp)
 1724c9b:	0f 84 9f fe ff ff    	je     1724b40 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x330>
 1724ca1:	48 8b 85 00 fa ff ff 	mov    -0x600(%rbp),%rax
 1724ca8:	c6 00 00             	movb   $0x0,(%rax)
 1724cab:	e9 90 fe ff ff       	jmp    1724b40 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x330>
 1724cb0:	4c 89 f6             	mov    %r14,%rsi
 1724cb3:	4c 89 ff             	mov    %r15,%rdi
 1724cb6:	e8 e5 c0 f7 fe       	call   6a0da0 <RecipeLauncher::prepareForDownload(LaunchInfo const&)@plt>
 1724cbb:	41 89 c5             	mov    %eax,%r13d
 1724cbe:	85 c0                	test   %eax,%eax
 1724cc0:	0f 85 8a 04 00 00    	jne    1725150 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x940>
 1724cc6:	4c 89 f6             	mov    %r14,%rsi
 1724cc9:	4c 89 ff             	mov    %r15,%rdi
 1724ccc:	e8 2f 61 f9 fe       	call   6bae00 <RecipeLauncher::downloadToDev(LaunchInfo const&)@plt>
 1724cd1:	41 89 c5             	mov    %eax,%r13d
 1724cd4:	85 c0                	test   %eax,%eax
 1724cd6:	0f 85 a4 04 00 00    	jne    1725180 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x970>
 1724cdc:	49 8b bf 08 01 00 00 	mov    0x108(%r15),%rdi
 1724ce3:	b9 01 00 00 00       	mov    $0x1,%ecx
 1724ce8:	31 d2                	xor    %edx,%edx
 1724cea:	be 01 00 00 00       	mov    $0x1,%esi
 1724cef:	48 8b 07             	mov    (%rdi),%rax
 1724cf2:	48 03 b8 28 ff ff ff 	add    -0xd8(%rax),%rdi
 1724cf9:	48 8b 07             	mov    (%rdi),%rax
 1724cfc:	ff 50 68             	call   *0x68(%rax)
 1724cff:	41 89 c5             	mov    %eax,%r13d
 1724d02:	85 c0                	test   %eax,%eax
 1724d04:	0f 84 b6 02 00 00    	je     1724fc0 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x7b0>
 1724d0a:	48 8b 05 4f 02 44 02 	mov    0x244024f(%rip),%rax        # 3b64f60 <hl_logger::v12_inline_fmt_compile::ModuleLoggerData<synapse::LogManager::LogType>::levels@@Base-0x7540>
 1724d11:	48 8d 9d d0 fd ff ff 	lea    -0x230(%rbp),%rbx
 1724d18:	c4 c1 f9 6e dc       	vmovq  %r12,%xmm3
 1724d1d:	4c 8d 85 b0 fd ff ff 	lea    -0x250(%rbp),%r8
 1724d24:	c4 e3 e1 22 c3 01    	vpinsrq $0x1,%rbx,%xmm3,%xmm0
 1724d2a:	80 78 18 04          	cmpb   $0x4,0x18(%rax)
 1724d2e:	0f 87 2c 04 00 00    	ja     1725160 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x950>
 1724d34:	4c 8d a5 90 f9 ff ff 	lea    -0x670(%rbp),%r12
 1724d3b:	be 0c 00 00 00       	mov    $0xc,%esi
 1724d40:	c5 f9 7f 85 80 f9 ff 	vmovdqa %xmm0,-0x680(%rbp)
 1724d47:	ff 
 1724d48:	41 be 13 00 00 00    	mov    $0x13,%r14d
 1724d4e:	4c 89 e7             	mov    %r12,%rdi
 1724d51:	4c 89 85 70 f9 ff ff 	mov    %r8,-0x690(%rbp)
 1724d58:	4c 8d 3d 31 de a4 01 	lea    0x1a4de31(%rip),%r15        # 3172b90 <void std::deque<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > >, std::allocator<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > > >::_M_push_back_aux<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&>(std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&)@@Base+0x4f3a0>
 1724d5f:	e8 1c a5 f9 fe       	call   6bf280 <std::shared_ptr<hl_logger::Logger> hl_logger::v12_inline_fmt_compile::getLogger<synapse::LogManager::LogType>(synapse::LogManager::LogType)@plt>
 1724d64:	c5 f9 6f 85 80 f9 ff 	vmovdqa -0x680(%rbp),%xmm0
 1724d6b:	ff 
 1724d6c:	ba 2f 00 00 00       	mov    $0x2f,%edx
 1724d71:	48 8b bd 70 f9 ff ff 	mov    -0x690(%rbp),%rdi
 1724d78:	48 8d 35 21 e4 a4 01 	lea    0x1a4e421(%rip),%rsi        # 31731a0 <void std::deque<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > >, std::allocator<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > > >::_M_push_back_aux<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&>(std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&)@@Base+0x4f9b0>
 1724d7f:	48 c7 85 a8 f9 ff ff 	movq   $0x2f,-0x658(%rbp)
 1724d86:	2f 00 00 00 
 1724d8a:	c5 f9 7f 85 b0 fd ff 	vmovdqa %xmm0,-0x250(%rbp)
 1724d91:	ff 
 1724d92:	c5 f9 6f 05 76 29 ad 	vmovdqa 0x1ad2976(%rip),%xmm0        # 31f7710 <std::__detail::__to_chars_10_impl<unsigned long>(char*, unsigned int, unsigned long)::__digits@@Base+0xd0>
 1724d99:	01 
 1724d9a:	48 89 b5 a0 f9 ff ff 	mov    %rsi,-0x660(%rbp)
 1724da1:	c5 f9 7f 85 c0 fd ff 	vmovdqa %xmm0,-0x240(%rbp)
 1724da8:	ff 
 1724da9:	e8 e2 24 49 ff       	call   bb7290 <std::_Rb_tree<std::pair<unsigned int, std::shared_ptr<BasicFieldInfo> >, std::pair<unsigned int, std::shared_ptr<BasicFieldInfo> >, std::_Identity<std::pair<unsigned int, std::shared_ptr<BasicFieldInfo> > >, PairComparator<unsigned int, std::shared_ptr<BasicFieldInfo> >, std::allocator<std::pair<unsigned int, std::shared_ptr<BasicFieldInfo> > > >::_M_get_insert_hint_unique_pos(std::_Rb_tree_const_iterator<std::pair<unsigned int, std::shared_ptr<BasicFieldInfo> > >, std::pair<unsigned int, std::shared_ptr<BasicFieldInfo> > const&)@@Base+0xc770>
 1724dae:	48 8b 15 e3 76 ad 01 	mov    0x1ad76e3(%rip),%rdx        # 31fc498 <std::__detail::__to_chars_10_impl<unsigned int>(char*, unsigned int, unsigned int)::__digits@@Base+0x11b8>
 1724db5:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
 1724db9:	44 89 ee             	mov    %r13d,%esi
 1724dbc:	48 89 85 00 fa ff ff 	mov    %rax,-0x600(%rbp)
 1724dc3:	48 8d bd 60 fa ff ff 	lea    -0x5a0(%rbp),%rdi
 1724dca:	48 b8 00 00 00 20 00 	movabs $0x100000020000000,%rax
 1724dd1:	00 00 01 
 1724dd4:	48 c7 85 08 fa ff ff 	movq   $0x0,-0x5f8(%rbp)
 1724ddb:	00 00 00 00 
 1724ddf:	48 89 95 60 fa ff ff 	mov    %rdx,-0x5a0(%rbp)
 1724de6:	48 89 95 a8 fa ff ff 	mov    %rdx,-0x558(%rbp)
 1724ded:	48 8d 95 00 fa ff ff 	lea    -0x600(%rbp),%rdx
 1724df4:	48 89 85 68 fa ff ff 	mov    %rax,-0x598(%rbp)
 1724dfb:	48 c7 85 70 fa ff ff 	movq   $0x0,-0x590(%rbp)
 1724e02:	00 00 00 00 
 1724e06:	48 c7 85 78 fa ff ff 	movq   $0x0,-0x588(%rbp)
 1724e0d:	00 00 00 00 
 1724e11:	48 c7 85 88 fa ff ff 	movq   $0x0,-0x578(%rbp)
 1724e18:	00 00 00 00 
 1724e1c:	48 c7 85 90 fa ff ff 	movq   $0x0,-0x570(%rbp)
 1724e23:	00 00 00 00 
 1724e27:	48 c7 85 a0 fa ff ff 	movq   $0x0,-0x560(%rbp)
 1724e2e:	00 00 00 00 
 1724e32:	48 89 85 b0 fa ff ff 	mov    %rax,-0x550(%rbp)
 1724e39:	48 c7 85 b8 fa ff ff 	movq   $0x0,-0x548(%rbp)
 1724e40:	00 00 00 00 
 1724e44:	48 c7 85 c0 fa ff ff 	movq   $0x0,-0x540(%rbp)
 1724e4b:	00 00 00 00 
 1724e4f:	48 c7 85 d0 fa ff ff 	movq   $0x0,-0x530(%rbp)
 1724e56:	00 00 00 00 
 1724e5a:	48 c7 85 d8 fa ff ff 	movq   $0x0,-0x528(%rbp)
 1724e61:	00 00 00 00 
 1724e65:	c5 f9 7f 85 10 fa ff 	vmovdqa %xmm0,-0x5f0(%rbp)
 1724e6c:	ff 
 1724e6d:	e8 ee 43 4d ff       	call   bf9260 <void synapse::hbAssert<char const*>(bool, synStatus, synapse::LogManager::LogType, unsigned int, char const*, char const*, char const*, int, char const*, char const*&&)@@Base+0x2080>
 1724e72:	48 83 ec 08          	sub    $0x8,%rsp
 1724e76:	48 8b 8d c0 fd ff ff 	mov    -0x240(%rbp),%rcx
 1724e7d:	31 d2                	xor    %edx,%edx
 1724e7f:	4c 89 e7             	mov    %r12,%rdi
 1724e82:	6a 00                	push   $0x0
 1724e84:	41 b9 9f 00 00 00    	mov    $0x9f,%r9d
 1724e8a:	be 04 00 00 00       	mov    $0x4,%esi
 1724e8f:	4c 8b 85 b8 fd ff ff 	mov    -0x248(%rbp),%r8
 1724e96:	41 57                	push   %r15
 1724e98:	41 56                	push   %r14
 1724e9a:	e8 11 12 f8 fe       	call   6a60b0 <hl_logger::v1_0::log(std::shared_ptr<hl_logger::Logger> const&, int, bool, std::basic_string_view<char, std::char_traits<char> >, std::basic_string_view<char, std::char_traits<char> >, int, bool)@plt>
 1724e9f:	48 8d 05 e2 73 39 02 	lea    0x23973e2(%rip),%rax        # 3abc288 <typeinfo for hl_gcfg::inline_ver_1_11::GcfgItem@@Base+0x18>
 1724ea6:	48 8b bd b8 fd ff ff 	mov    -0x248(%rbp),%rdi
 1724ead:	48 83 c4 20          	add    $0x20,%rsp
 1724eb1:	48 83 c0 10          	add    $0x10,%rax
 1724eb5:	48 89 85 b0 fd ff ff 	mov    %rax,-0x250(%rbp)
 1724ebc:	48 39 df             	cmp    %rbx,%rdi
 1724ebf:	74 0c                	je     1724ecd <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x6bd>
 1724ec1:	48 8b b5 c8 fd ff ff 	mov    -0x238(%rbp),%rsi
 1724ec8:	e8 b3 dd f7 fe       	call   6a2c80 <operator delete(void*, unsigned long)@plt>
 1724ecd:	48 8b bd 98 f9 ff ff 	mov    -0x668(%rbp),%rdi
 1724ed4:	48 85 ff             	test   %rdi,%rdi
 1724ed7:	74 05                	je     1724ede <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x6ce>
 1724ed9:	e8 12 5d f8 fe       	call   6aabf0 <std::_Sp_counted_base<(__gnu_cxx::_Lock_policy)2>::_M_release()@plt>
 1724ede:	48 8b 05 7b 00 44 02 	mov    0x244007b(%rip),%rax        # 3b64f60 <hl_logger::v12_inline_fmt_compile::ModuleLoggerData<synapse::LogManager::LogType>::levels@@Base-0x7540>
 1724ee5:	80 78 19 04          	cmpb   $0x4,0x19(%rax)
 1724ee9:	0f 87 65 fc ff ff    	ja     1724b54 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x344>
 1724eef:	90                   	nop
 1724ef0:	48 8b 05 91 68 44 02 	mov    0x2446891(%rip),%rax        # 3b6b788 <hl_logger::v12_inline_fmt_compile::moduleLoggerData<synapse::LogManager::LogType>@@Base-0x25ca58>
 1724ef7:	be 01 00 00 00       	mov    $0x1,%esi
 1724efc:	48 8b 80 d0 0b 00 00 	mov    0xbd0(%rax),%rax
 1724f03:	f0 48 0f c1 70 10    	lock xadd %rsi,0x10(%rax)
 1724f09:	0f b6 48 08          	movzbl 0x8(%rax),%ecx
 1724f0d:	49 89 f6             	mov    %rsi,%r14
 1724f10:	4c 89 e7             	mov    %r12,%rdi
 1724f13:	49 d3 ee             	shr    %cl,%r14
 1724f16:	48 8b 08             	mov    (%rax),%rcx
 1724f19:	48 8d 51 ff          	lea    -0x1(%rcx),%rdx
 1724f1d:	48 21 f2             	and    %rsi,%rdx
 1724f20:	48 8d 1c d5 00 00 00 	lea    0x0(,%rdx,8),%rbx
 1724f27:	00 
 1724f28:	48 29 d3             	sub    %rdx,%rbx
 1724f2b:	48 c1 e3 04          	shl    $0x4,%rbx
 1724f2f:	48 03 58 18          	add    0x18(%rax),%rbx
 1724f33:	48 89 de             	mov    %rbx,%rsi
 1724f36:	e8 25 f8 f8 fe       	call   6b4760 <containers::hllog_1_0::ConcurrentRecentElementsQueue<hl_logger::v12_inline_fmt_compile::LazyLogItem<64u> >::acquireItem(containers::hllog_1_0::ConcurrentRecentElementsQueue<hl_logger::v12_inline_fmt_compile::LazyLogItem<64u> >::InternalItem const&)@plt>
 1724f3b:	44 89 f0             	mov    %r14d,%eax
 1724f3e:	2a 43 01             	sub    0x1(%rbx),%al
 1724f41:	3c 03                	cmp    $0x3,%al
 1724f43:	77 5c                	ja     1724fa1 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x791>
 1724f45:	80 7b 68 00          	cmpb   $0x0,0x68(%rbx)
 1724f49:	4c 8d 63 08          	lea    0x8(%rbx),%r12
 1724f4d:	0f 85 01 04 00 00    	jne    1725354 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0xb44>
 1724f53:	c6 43 08 04          	movb   $0x4,0x8(%rbx)
 1724f57:	e8 34 49 f8 fe       	call   6a9890 <hl_logger::internal::v1_0::getThreadId()@plt>
 1724f5c:	48 89 43 10          	mov    %rax,0x10(%rbx)
 1724f60:	e8 2b 23 f9 fe       	call   6b7290 <std::chrono::_V2::system_clock::now()@plt>
 1724f65:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
 1724f69:	49 89 44 24 10       	mov    %rax,0x10(%r12)
 1724f6e:	48 8d 43 20          	lea    0x20(%rbx),%rax
 1724f72:	48 89 43 60          	mov    %rax,0x60(%rbx)
 1724f76:	48 8d 05 fb 7c 36 02 	lea    0x2367cfb(%rip),%rax        # 3a8cc78 <MappedMemMgr::enumNamePoints@@Base+0x1b8>
 1724f7d:	c5 fa 7f 43 20       	vmovdqu %xmm0,0x20(%rbx)
 1724f82:	c6 43 68 01          	movb   $0x1,0x68(%rbx)
 1724f86:	48 89 43 20          	mov    %rax,0x20(%rbx)
 1724f8a:	44 89 6b 28          	mov    %r13d,0x28(%rbx)
 1724f8e:	44 88 73 01          	mov    %r14b,0x1(%rbx)
 1724f92:	c5 fa 7f 43 30       	vmovdqu %xmm0,0x30(%rbx)
 1724f97:	c5 fa 7f 43 40       	vmovdqu %xmm0,0x40(%rbx)
 1724f9c:	c5 fa 7f 43 50       	vmovdqu %xmm0,0x50(%rbx)
 1724fa1:	80 bd 98 f9 ff ff 00 	cmpb   $0x0,-0x668(%rbp)
 1724fa8:	0f 84 a6 fb ff ff    	je     1724b54 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x344>
 1724fae:	48 8b 85 90 f9 ff ff 	mov    -0x670(%rbp),%rax
 1724fb5:	c6 00 00             	movb   $0x0,(%rax)
 1724fb8:	e9 97 fb ff ff       	jmp    1724b54 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x344>
 1724fbd:	0f 1f 00             	nopl   (%rax)
 1724fc0:	e8 5b e9 f9 fe       	call   6c3920 <ScalUtils::isCompareAfterDownLoad()@plt>
 1724fc5:	84 c0                	test   %al,%al
 1724fc7:	0f 84 b5 00 00 00    	je     1725082 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x872>
 1724fcd:	48 8d 85 00 fa ff ff 	lea    -0x600(%rbp),%rax
 1724fd4:	4c 8d a5 b0 fd ff ff 	lea    -0x250(%rbp),%r12
 1724fdb:	31 d2                	xor    %edx,%edx
 1724fdd:	48 c7 85 00 fa ff ff 	movq   $0x12,-0x600(%rbp)
 1724fe4:	12 00 00 00 
 1724fe8:	48 89 c6             	mov    %rax,%rsi
 1724feb:	4c 89 e7             	mov    %r12,%rdi
 1724fee:	48 8d 9d c0 fd ff ff 	lea    -0x240(%rbp),%rbx
 1724ff5:	48 89 85 80 f9 ff ff 	mov    %rax,-0x680(%rbp)
 1724ffc:	48 89 9d b0 fd ff ff 	mov    %rbx,-0x250(%rbp)
 1725003:	e8 a8 09 f9 fe       	call   6b59b0 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_create(unsigned long&, unsigned long)@plt>
 1725008:	48 8b 95 00 fa ff ff 	mov    -0x600(%rbp),%rdx
 172500f:	b9 70 79 00 00       	mov    $0x7970,%ecx
 1725014:	4c 89 ff             	mov    %r15,%rdi
 1725017:	c5 f9 6f 05 a1 31 ad 	vmovdqa 0x1ad31a1(%rip),%xmm0        # 31f81c0 <std::__detail::__to_chars_10_impl<unsigned long>(char*, unsigned int, unsigned long)::__digits@@Base+0xb80>
 172501e:	01 
 172501f:	48 89 85 b0 fd ff ff 	mov    %rax,-0x250(%rbp)
 1725026:	48 89 95 c0 fd ff ff 	mov    %rdx,-0x240(%rbp)
 172502d:	66 89 48 10          	mov    %cx,0x10(%rax)
 1725031:	c5 fa 7f 00          	vmovdqu %xmm0,(%rax)
 1725035:	48 8b 85 00 fa ff ff 	mov    -0x600(%rbp),%rax
 172503c:	48 8b 95 b0 fd ff ff 	mov    -0x250(%rbp),%rdx
 1725043:	48 89 85 b8 fd ff ff 	mov    %rax,-0x248(%rbp)
 172504a:	c6 04 02 00          	movb   $0x0,(%rdx,%rax,1)
 172504e:	49 8b 76 18          	mov    0x18(%r14),%rsi
 1725052:	4c 89 e2             	mov    %r12,%rdx
 1725055:	e8 86 75 f9 fe       	call   6bc5e0 <RecipeLauncher::debugCompareRecipeOnDev(InternalRecipeHandle const*, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)@plt>
 172505a:	48 8b bd b0 fd ff ff 	mov    -0x250(%rbp),%rdi
 1725061:	41 89 c4             	mov    %eax,%r12d
 1725064:	48 39 df             	cmp    %rbx,%rdi
 1725067:	74 10                	je     1725079 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x869>
 1725069:	48 8b 85 c0 fd ff ff 	mov    -0x240(%rbp),%rax
 1725070:	48 8d 70 01          	lea    0x1(%rax),%rsi
 1725074:	e8 07 dc f7 fe       	call   6a2c80 <operator delete(void*, unsigned long)@plt>
 1725079:	45 85 e4             	test   %r12d,%r12d
 172507c:	0f 85 4f 02 00 00    	jne    17252d1 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0xac1>
 1725082:	49 8b 46 18          	mov    0x18(%r14),%rax
 1725086:	49 8b 56 30          	mov    0x30(%r14),%rdx
 172508a:	4c 89 ff             	mov    %r15,%rdi
 172508d:	45 31 e4             	xor    %r12d,%r12d
 1725090:	4c 8b a8 08 06 00 00 	mov    0x608(%rax),%r13
 1725097:	4c 89 ee             	mov    %r13,%rsi
 172509a:	e8 31 0c fa fe       	call   6c5cd0 <RecipeLauncher::updateSfgLongSos(unsigned long, std::vector<SlotMapItemSptr<EventWithMappedTensor>, std::allocator<SlotMapItemSptr<EventWithMappedTensor> > >&)@plt>
 172509f:	48 8b 1d 5a 62 44 02 	mov    0x244625a(%rip),%rbx        # 3b6b300 <g_globalStat@@Base-0x25c560>
 17250a6:	80 7b 2c 00          	cmpb   $0x0,0x2c(%rbx)
 17250aa:	0f 85 d8 01 00 00    	jne    1725288 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0xa78>
 17250b0:	4c 89 ee             	mov    %r13,%rsi
 17250b3:	4c 89 ff             	mov    %r15,%rdi
 17250b6:	e8 35 37 fa fe       	call   6c87f0 <RecipeLauncher::scalEnqueue(unsigned long)@plt>
 17250bb:	41 89 c5             	mov    %eax,%r13d
 17250be:	85 c0                	test   %eax,%eax
 17250c0:	0f 85 8e fa ff ff    	jne    1724b54 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x344>
 17250c6:	80 7b 2c 00          	cmpb   $0x0,0x2c(%rbx)
 17250ca:	0f 85 c0 00 00 00    	jne    1725190 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x980>
 17250d0:	49 8b 7e 18          	mov    0x18(%r14),%rdi
 17250d4:	48 8b 07             	mov    (%rdi),%rax
 17250d7:	8b 80 f8 00 00 00    	mov    0xf8(%rax),%eax
 17250dd:	85 c0                	test   %eax,%eax
 17250df:	74 4c                	je     172512d <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x91d>
 17250e1:	45 31 e4             	xor    %r12d,%r12d
 17250e4:	4c 89 f6             	mov    %r14,%rsi
 17250e7:	4c 89 ff             	mov    %r15,%rdi
 17250ea:	e8 01 0f f9 fe       	call   6b5ff0 <RecipeLauncher::validateSectionsInfo(LaunchInfo const&)@plt>
 17250ef:	85 c0                	test   %eax,%eax
 17250f1:	0f 85 9e 01 00 00    	jne    1725295 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0xa85>
 17250f7:	80 7b 2c 00          	cmpb   $0x0,0x2c(%rbx)
 17250fb:	74 2c                	je     1725129 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x919>
 17250fd:	e8 de bd f9 fe       	call   6c0ee0 <std::chrono::_V2::steady_clock::now()@plt>
 1725102:	48 8b 53 40          	mov    0x40(%rbx),%rdx
 1725106:	4c 29 e0             	sub    %r12,%rax
 1725109:	f0 48 01 82 00 02 00 	lock add %rax,0x200(%rdx)
 1725110:	00 
 1725111:	48 8b 43 40          	mov    0x40(%rbx),%rax
 1725115:	f0 48 83 80 08 02 00 	lock addq $0x1,0x208(%rax)
 172511c:	00 01 
 172511e:	48 8b 43 40          	mov    0x40(%rbx),%rax
 1725122:	48 8b 80 08 02 00 00 	mov    0x208(%rax),%rax
 1725129:	49 8b 7e 18          	mov    0x18(%r14),%rdi
 172512d:	e8 7e ba f9 fe       	call   6c0bb0 <RecipeUtils::isKernelPrintf(InternalRecipeHandle const&)@plt>
 1725132:	84 c0                	test   %al,%al
 1725134:	0f 84 1a fa ff ff    	je     1724b54 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x344>
 172513a:	4c 89 f6             	mov    %r14,%rsi
 172513d:	4c 89 ff             	mov    %r15,%rdi
 1725140:	e8 9b 02 f8 fe       	call   6a53e0 <RecipeLauncher::handleKernelsPrintf(LaunchInfo const&)@plt>
 1725145:	e9 0a fa ff ff       	jmp    1724b54 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x344>
 172514a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
 1725150:	4c 89 ff             	mov    %r15,%rdi
 1725153:	e8 98 1d f9 fe       	call   6b6ef0 <RecipeLauncher::clearMemMgrs()@plt>
 1725158:	e9 f7 f9 ff ff       	jmp    1724b54 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x344>
 172515d:	0f 1f 00             	nopl   (%rax)
 1725160:	80 78 19 04          	cmpb   $0x4,0x19(%rax)
 1725164:	4c 8d a5 90 f9 ff ff 	lea    -0x670(%rbp),%r12
 172516b:	0f 86 7f fd ff ff    	jbe    1724ef0 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x6e0>
 1725171:	e9 de f9 ff ff       	jmp    1724b54 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x344>
 1725176:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
 172517d:	00 00 00 
 1725180:	4c 89 f6             	mov    %r14,%rsi
 1725183:	4c 89 ff             	mov    %r15,%rdi
 1725186:	e8 65 0e f9 fe       	call   6b5ff0 <RecipeLauncher::validateSectionsInfo(LaunchInfo const&)@plt>
 172518b:	e9 c4 f9 ff ff       	jmp    1724b54 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x344>
 1725190:	e8 4b bd f9 fe       	call   6c0ee0 <std::chrono::_V2::steady_clock::now()@plt>
 1725195:	48 8b 53 40          	mov    0x40(%rbx),%rdx
 1725199:	4c 29 e0             	sub    %r12,%rax
 172519c:	f0 48 01 82 40 08 00 	lock add %rax,0x840(%rdx)
 17251a3:	00 
 17251a4:	48 8b 43 40          	mov    0x40(%rbx),%rax
 17251a8:	f0 48 83 80 48 08 00 	lock addq $0x1,0x848(%rax)
 17251af:	00 01 
 17251b1:	48 8b 43 40          	mov    0x40(%rbx),%rax
 17251b5:	48 8b 80 48 08 00 00 	mov    0x848(%rax),%rax
 17251bc:	80 7b 2c 00          	cmpb   $0x0,0x2c(%rbx)
 17251c0:	0f 84 0a ff ff ff    	je     17250d0 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x8c0>
 17251c6:	e8 15 bd f9 fe       	call   6c0ee0 <std::chrono::_V2::steady_clock::now()@plt>
 17251cb:	49 89 c4             	mov    %rax,%r12
 17251ce:	49 8b 46 18          	mov    0x18(%r14),%rax
 17251d2:	48 8b 00             	mov    (%rax),%rax
 17251d5:	8b 90 f8 00 00 00    	mov    0xf8(%rax),%edx
 17251db:	85 d2                	test   %edx,%edx
 17251dd:	0f 85 01 ff ff ff    	jne    17250e4 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x8d4>
 17251e3:	e9 0f ff ff ff       	jmp    17250f7 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x8e7>
 17251e8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
 17251ef:	00 
 17251f0:	48 89 d0             	mov    %rdx,%rax
 17251f3:	48 d1 e8             	shr    $1,%rax
 17251f6:	48 01 d0             	add    %rdx,%rax
 17251f9:	48 89 85 38 f9 ff ff 	mov    %rax,-0x6c8(%rbp)
 1725200:	48 39 c1             	cmp    %rax,%rcx
 1725203:	0f 87 94 00 00 00    	ja     172529d <RecipeLauncher::launch(LaunchInfo const&)@@Base+0xa8d>
 1725209:	48 85 c0             	test   %rax,%rax
 172520c:	0f 88 25 01 00 00    	js     1725337 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0xb27>
 1725212:	48 8b bd 38 f9 ff ff 	mov    -0x6c8(%rbp),%rdi
 1725219:	e8 12 f0 f7 fe       	call   6a4230 <operator new(unsigned long)@plt>
 172521e:	49 8b 55 10          	mov    0x10(%r13),%rdx
 1725222:	48 89 c1             	mov    %rax,%rcx
 1725225:	48 85 d2             	test   %rdx,%rdx
 1725228:	0f 85 f2 00 00 00    	jne    1725320 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0xb10>
 172522e:	48 8b 85 38 f9 ff ff 	mov    -0x6c8(%rbp),%rax
 1725235:	48 8b bd 40 f9 ff ff 	mov    -0x6c0(%rbp),%rdi
 172523c:	49 89 4d 08          	mov    %rcx,0x8(%r13)
 1725240:	49 89 45 18          	mov    %rax,0x18(%r13)
 1725244:	49 8d 45 20          	lea    0x20(%r13),%rax
 1725248:	48 39 c7             	cmp    %rax,%rdi
 172524b:	74 6a                	je     17252b7 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0xaa7>
 172524d:	48 8b b5 60 f9 ff ff 	mov    -0x6a0(%rbp),%rsi
 1725254:	e8 27 da f7 fe       	call   6a2c80 <operator delete(void*, unsigned long)@plt>
 1725259:	49 8b 45 18          	mov    0x18(%r13),%rax
 172525d:	48 89 85 60 f9 ff ff 	mov    %rax,-0x6a0(%rbp)
 1725264:	49 8b 45 08          	mov    0x8(%r13),%rax
 1725268:	48 89 85 40 f9 ff ff 	mov    %rax,-0x6c0(%rbp)
 172526f:	e9 ab f7 ff ff       	jmp    1724a1f <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x20f>
 1725274:	c6 43 68 00          	movb   $0x0,0x68(%rbx)
 1725278:	48 89 c7             	mov    %rax,%rdi
 172527b:	e8 10 ce f8 fe       	call   6b2090 <hl_logger::v12_inline_fmt_compile::LazyLogItem<64u>::~LazyLogItem()@plt>
 1725280:	e9 9d f9 ff ff       	jmp    1724c22 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x412>
 1725285:	0f 1f 00             	nopl   (%rax)
 1725288:	e8 53 bc f9 fe       	call   6c0ee0 <std::chrono::_V2::steady_clock::now()@plt>
 172528d:	49 89 c4             	mov    %rax,%r12
 1725290:	e9 1b fe ff ff       	jmp    17250b0 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x8a0>
 1725295:	41 89 c5             	mov    %eax,%r13d
 1725298:	e9 b7 f8 ff ff       	jmp    1724b54 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x344>
 172529d:	48 89 8d 38 f9 ff ff 	mov    %rcx,-0x6c8(%rbp)
 17252a4:	48 83 bd 38 f9 ff ff 	cmpq   $0x0,-0x6c8(%rbp)
 17252ab:	00 
 17252ac:	0f 89 60 ff ff ff    	jns    1725212 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0xa02>
 17252b2:	e8 79 cd f7 fe       	call   6a2030 <std::__throw_bad_alloc()@plt>
 17252b7:	48 8b 85 38 f9 ff ff 	mov    -0x6c8(%rbp),%rax
 17252be:	48 89 8d 40 f9 ff ff 	mov    %rcx,-0x6c0(%rbp)
 17252c5:	48 89 85 60 f9 ff ff 	mov    %rax,-0x6a0(%rbp)
 17252cc:	e9 4e f7 ff ff       	jmp    1724a1f <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x20f>
 17252d1:	48 8d 05 8c 5d a0 01 	lea    0x1a05d8c(%rip),%rax        # 312b064 <void std::deque<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > >, std::allocator<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > > >::_M_push_back_aux<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&>(std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&)@@Base+0x7874>
 17252d8:	ff b5 80 f9 ff ff    	push   -0x680(%rbp)
 17252de:	31 d2                	xor    %edx,%edx
 17252e0:	4c 8d 0d f1 de a4 01 	lea    0x1a4def1(%rip),%r9        # 31731d8 <void std::deque<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > >, std::allocator<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > > >::_M_push_back_aux<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&>(std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&)@@Base+0x4f9e8>
 17252e7:	50                   	push   %rax
 17252e8:	b9 05 00 00 00       	mov    $0x5,%ecx
 17252ed:	be 1a 00 00 00       	mov    $0x1a,%esi
 17252f2:	bf 01 00 00 00       	mov    $0x1,%edi
 17252f7:	48 8d 05 92 d8 a4 01 	lea    0x1a4d892(%rip),%rax        # 3172b90 <void std::deque<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > >, std::allocator<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > > >::_M_push_back_aux<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&>(std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&)@@Base+0x4f3a0>
 17252fe:	68 a8 00 00 00       	push   $0xa8
 1725303:	4c 8d 05 1d 80 a0 01 	lea    0x1a0801d(%rip),%r8        # 312d327 <void std::deque<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > >, std::allocator<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > > >::_M_push_back_aux<std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&>(std::queue<unsigned int, std::deque<unsigned int, std::allocator<unsigned int> > > const&)@@Base+0x9b37>
 172530a:	50                   	push   %rax
 172530b:	44 89 a5 00 fa ff ff 	mov    %r12d,-0x600(%rbp)
 1725312:	e8 19 15 f9 fe       	call   6b6830 <void synapse::hbAssert<hl_logger::v12_inline_fmt_compile::details::EnumPack<synStatus> >(bool, synStatus, synapse::LogManager::LogType, unsigned int, char const*, char const*, char const*, int, char const*, hl_logger::v12_inline_fmt_compile::details::EnumPack<synStatus>&&)@plt>
 1725317:	48 83 c4 20          	add    $0x20,%rsp
 172531b:	e8 50 5a f9 fe       	call   6bad70 <__cxa_rethrow@plt>
 1725320:	48 8b b5 40 f9 ff ff 	mov    -0x6c0(%rbp),%rsi
 1725327:	48 89 c7             	mov    %rax,%rdi
 172532a:	e8 91 ef f7 fe       	call   6a42c0 <memmove@plt>
 172532f:	48 89 c1             	mov    %rax,%rcx
 1725332:	e9 f7 fe ff ff       	jmp    172522e <RecipeLauncher::launch(LaunchInfo const&)@@Base+0xa1e>
 1725337:	48 b8 ff ff ff ff ff 	movabs $0x7fffffffffffffff,%rax
 172533e:	ff ff 7f 
 1725341:	48 39 c1             	cmp    %rax,%rcx
 1725344:	48 0f 43 c1          	cmovae %rcx,%rax
 1725348:	48 89 85 38 f9 ff ff 	mov    %rax,-0x6c8(%rbp)
 172534f:	e9 50 ff ff ff       	jmp    17252a4 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0xa94>
 1725354:	c6 43 68 00          	movb   $0x0,0x68(%rbx)
 1725358:	4c 89 e7             	mov    %r12,%rdi
 172535b:	e8 30 cd f8 fe       	call   6b2090 <hl_logger::v12_inline_fmt_compile::LazyLogItem<64u>::~LazyLogItem()@plt>
 1725360:	e9 ee fb ff ff       	jmp    1724f53 <RecipeLauncher::launch(LaunchInfo const&)@@Base+0x743>
 1725365:	e8 d6 3b f8 fe       	call   6a8f40 <__stack_chk_fail@plt>
 172536a:	f3 0f 1e fa          	endbr64
 172536e:	48 89 95 70 f9 ff ff 	mov    %rdx,-0x690(%rbp)
 1725375:	49 89 c5             	mov    %rax,%r13
 1725378:	e9 a5 3e 15 ff       	jmp    879222 <gaudi::QmansDefinition::getArbitrationMaster(internalStreamType) const@@Base+0x3a15a>
 172537d:	f3 0f 1e fa          	endbr64
 1725381:	49 89 c7             	mov    %rax,%r15
 1725384:	49 89 d6             	mov    %rdx,%r14
 1725387:	e9 5c 3f 15 ff       	jmp    8792e8 <gaudi::QmansDefinition::getArbitrationMaster(internalStreamType) const@@Base+0x3a220>
 172538c:	f3 0f 1e fa          	endbr64
 1725390:	48 89 c7             	mov    %rax,%rdi
 1725393:	e9 c3 40 15 ff       	jmp    87945b <gaudi::QmansDefinition::getArbitrationMaster(internalStreamType) const@@Base+0x3a393>
 1725398:	f3 0f 1e fa          	endbr64
 172539c:	49 89 c4             	mov    %rax,%r12
 172539f:	e9 fd 3f 15 ff       	jmp    8793a1 <gaudi::QmansDefinition::getArbitrationMaster(internalStreamType) const@@Base+0x3a2d9>
 17253a4:	f3 0f 1e fa          	endbr64
 17253a8:	48 89 c7             	mov    %rax,%rdi
 17253ab:	e9 c6 40 15 ff       	jmp    879476 <gaudi::QmansDefinition::getArbitrationMaster(internalStreamType) const@@Base+0x3a3ae>
