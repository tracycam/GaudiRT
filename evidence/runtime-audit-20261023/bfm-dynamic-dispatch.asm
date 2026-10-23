
/home/rate/llm/Gaudi-LLM/GaudiRT/evidence/runtime-audit-20261023/binaries/libeng_mme_cmpt_bfm.so:     file format elf64-x86-64


Disassembly of section .text:

00000000000604e0 <EngineBFM::process_compute_dynamic_ecb(engine_ctxt_t*, unsigned int)@@Base>:
   604e0:	f3 0f 1e fa          	endbr64 
   604e4:	55                   	push   %rbp
   604e5:	48 89 e5             	mov    %rsp,%rbp
   604e8:	41 57                	push   %r15
   604ea:	41 56                	push   %r14
   604ec:	4c 8d 7d a0          	lea    -0x60(%rbp),%r15
   604f0:	49 89 f6             	mov    %rsi,%r14
   604f3:	41 55                	push   %r13
   604f5:	41 54                	push   %r12
   604f7:	4c 8d 25 c2 f0 00 00 	lea    0xf0c2(%rip),%r12        # 6f5c0 <typeinfo name for std::_Nested_exception<boost::context::ontop_error>@@Base+0x1a0>
   604fe:	53                   	push   %rbx
   604ff:	48 81 ec a8 00 00 00 	sub    $0xa8,%rsp
   60506:	48 89 bd 68 ff ff ff 	mov    %rdi,-0x98(%rbp)
   6050d:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
   60514:	00 00 
   60516:	48 89 45 c8          	mov    %rax,-0x38(%rbp)
   6051a:	48 89 f8             	mov    %rdi,%rax
   6051d:	48 81 c7 38 0a 00 00 	add    $0xa38,%rdi
   60524:	48 89 bd 40 ff ff ff 	mov    %rdi,-0xc0(%rbp)
   6052b:	44 8b a8 14 0b 00 00 	mov    0xb14(%rax),%r13d
   60532:	48 8d 45 80          	lea    -0x80(%rbp),%rax
   60536:	48 89 85 58 ff ff ff 	mov    %rax,-0xa8(%rbp)
   6053d:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
   60541:	48 89 85 60 ff ff ff 	mov    %rax,-0xa0(%rbp)
   60548:	41 83 e5 01          	and    $0x1,%r13d
   6054c:	41 83 f5 01          	xor    $0x1,%r13d
   60550:	48 8b 85 68 ff ff ff 	mov    -0x98(%rbp),%rax
   60557:	8b 98 68 0a 00 00    	mov    0xa68(%rax),%ebx
   6055d:	3b 98 64 0a 00 00    	cmp    0xa64(%rax),%ebx
   60563:	0f 84 b7 03 00 00    	je     60920 <EngineBFM::process_compute_dynamic_ecb(engine_ctxt_t*, unsigned int)@@Base+0x440>
   60569:	48 03 98 58 0a 00 00 	add    0xa58(%rax),%rbx
   60570:	48 8b bd 68 ff ff ff 	mov    -0x98(%rbp),%rdi
   60577:	4c 89 f6             	mov    %r14,%rsi
   6057a:	0f b6 03             	movzbl (%rbx),%eax
   6057d:	83 e0 0f             	and    $0xf,%eax
   60580:	3c 08                	cmp    $0x8,%al
   60582:	0f 87 30 01 00 00    	ja     606b8 <EngineBFM::process_compute_dynamic_ecb(engine_ctxt_t*, unsigned int)@@Base+0x1d8>
   60588:	0f b6 c0             	movzbl %al,%eax
   6058b:	49 63 04 84          	movslq (%r12,%rax,4),%rax
   6058f:	4c 01 e0             	add    %r12,%rax
   60592:	3e ff e0             	notrack jmp *%rax
   60595:	0f 1f 00             	nopl   (%rax)
   60598:	e8 03 fc fb ff       	call   201a0 <spdlog::details::os::thread_id()::tid@@Base+0x20198>
   6059d:	0f 1f 00             	nopl   (%rax)
   605a0:	85 c0                	test   %eax,%eax
   605a2:	74 09                	je     605ad <EngineBFM::process_compute_dynamic_ecb(engine_ctxt_t*, unsigned int)@@Base+0xcd>
   605a4:	45 84 ed             	test   %r13b,%r13b
   605a7:	0f 85 43 04 00 00    	jne    609f0 <EngineBFM::process_compute_dynamic_ecb(engine_ctxt_t*, unsigned int)@@Base+0x510>
   605ad:	48 8b b5 68 ff ff ff 	mov    -0x98(%rbp),%rsi
   605b4:	8b 96 68 0a 00 00    	mov    0xa68(%rsi),%edx
   605ba:	89 d1                	mov    %edx,%ecx
   605bc:	81 e1 ff 01 00 00    	and    $0x1ff,%ecx
   605c2:	89 8e 68 0a 00 00    	mov    %ecx,0xa68(%rsi)
   605c8:	84 d2                	test   %dl,%dl
   605ca:	75 12                	jne    605de <EngineBFM::process_compute_dynamic_ecb(engine_ctxt_t*, unsigned int)@@Base+0xfe>
   605cc:	8b b6 60 0a 00 00    	mov    0xa60(%rsi),%esi
   605d2:	85 f6                	test   %esi,%esi
   605d4:	74 08                	je     605de <EngineBFM::process_compute_dynamic_ecb(engine_ctxt_t*, unsigned int)@@Base+0xfe>
   605d6:	85 c0                	test   %eax,%eax
   605d8:	0f 84 ea 00 00 00    	je     606c8 <EngineBFM::process_compute_dynamic_ecb(engine_ctxt_t*, unsigned int)@@Base+0x1e8>
   605de:	f6 03 10             	testb  $0x10,(%rbx)
   605e1:	0f 84 69 ff ff ff    	je     60550 <EngineBFM::process_compute_dynamic_ecb(engine_ctxt_t*, unsigned int)@@Base+0x70>
   605e7:	45 84 ed             	test   %r13b,%r13b
   605ea:	0f 84 60 ff ff ff    	je     60550 <EngineBFM::process_compute_dynamic_ecb(engine_ctxt_t*, unsigned int)@@Base+0x70>
   605f0:	48 8b 9d 68 ff ff ff 	mov    -0x98(%rbp),%rbx
   605f7:	4c 8d 25 69 c9 00 00 	lea    0xc969(%rip),%r12        # 6cf67 <EngineBFM::dma_wait_for_idle()@@Base+0x3e47>
   605fe:	4c 89 ff             	mov    %r15,%rdi
   60601:	4c 89 e6             	mov    %r12,%rsi
   60604:	c7 83 18 0b 00 00 00 	movl   $0x0,0xb18(%rbx)
   6060b:	00 00 00 
   6060e:	e8 3d d9 ff ff       	call   5df50 <EngineBFM::config_and_arm_monitor(hw_qman_msg_short_t*, hw_qman_msg_short_t*, virt_sob_ids_t*, unsigned char, unsigned int*)@@Base+0xa10>
   60613:	48 8b b3 18 02 00 00 	mov    0x218(%rbx),%rsi
   6061a:	48 8b 9d 58 ff ff ff 	mov    -0xa8(%rbp),%rbx
   60621:	48 89 df             	mov    %rbx,%rdi
   60624:	e8 27 d9 ff ff       	call   5df50 <EngineBFM::config_and_arm_monitor(hw_qman_msg_short_t*, hw_qman_msg_short_t*, virt_sob_ids_t*, unsigned char, unsigned int*)@@Base+0xa10>
   60629:	48 8d bd 70 ff ff ff 	lea    -0x90(%rbp),%rdi
   60630:	4c 89 fa             	mov    %r15,%rdx
   60633:	48 89 de             	mov    %rbx,%rsi
   60636:	e8 35 f4 fb ff       	call   1fa70 <spdlog::details::os::thread_id()::tid@@Base+0x1fa68>
   6063b:	48 8b bd 70 ff ff ff 	mov    -0x90(%rbp),%rdi
   60642:	4c 89 e2             	mov    %r12,%rdx
   60645:	be 02 00 00 00       	mov    $0x2,%esi
   6064a:	e8 91 fb fb ff       	call   201e0 <spdlog::details::os::thread_id()::tid@@Base+0x201d8>
   6064f:	e9 2e 03 00 00       	jmp    60982 <EngineBFM::process_compute_dynamic_ecb(engine_ctxt_t*, unsigned int)@@Base+0x4a2>
   60654:	0f 1f 40 00          	nopl   0x0(%rax)
   60658:	e8 d3 f7 fb ff       	call   1fe30 <spdlog::details::os::thread_id()::tid@@Base+0x1fe28>
   6065d:	e9 3e ff ff ff       	jmp    605a0 <EngineBFM::process_compute_dynamic_ecb(engine_ctxt_t*, unsigned int)@@Base+0xc0>
   60662:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
   60668:	e8 a3 f9 fb ff       	call   20010 <spdlog::details::os::thread_id()::tid@@Base+0x20008>
   6066d:	e9 2e ff ff ff       	jmp    605a0 <EngineBFM::process_compute_dynamic_ecb(engine_ctxt_t*, unsigned int)@@Base+0xc0>
   60672:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
   60678:	e8 a3 ef fb ff       	call   1f620 <spdlog::details::os::thread_id()::tid@@Base+0x1f618>
   6067d:	e9 1e ff ff ff       	jmp    605a0 <EngineBFM::process_compute_dynamic_ecb(engine_ctxt_t*, unsigned int)@@Base+0xc0>
   60682:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
   60688:	e8 13 ed fb ff       	call   1f3a0 <spdlog::details::os::thread_id()::tid@@Base+0x1f398>
   6068d:	e9 0e ff ff ff       	jmp    605a0 <EngineBFM::process_compute_dynamic_ecb(engine_ctxt_t*, unsigned int)@@Base+0xc0>
   60692:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
   60698:	e8 d3 f4 fb ff       	call   1fb70 <spdlog::details::os::thread_id()::tid@@Base+0x1fb68>
   6069d:	e9 fe fe ff ff       	jmp    605a0 <EngineBFM::process_compute_dynamic_ecb(engine_ctxt_t*, unsigned int)@@Base+0xc0>
   606a2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
   606a8:	e8 a3 f3 fb ff       	call   1fa50 <spdlog::details::os::thread_id()::tid@@Base+0x1fa48>
   606ad:	e9 ee fe ff ff       	jmp    605a0 <EngineBFM::process_compute_dynamic_ecb(engine_ctxt_t*, unsigned int)@@Base+0xc0>
   606b2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
   606b8:	e8 43 ec fb ff       	call   1f300 <spdlog::details::os::thread_id()::tid@@Base+0x1f2f8>
   606bd:	e9 de fe ff ff       	jmp    605a0 <EngineBFM::process_compute_dynamic_ecb(engine_ctxt_t*, unsigned int)@@Base+0xc0>
   606c2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
   606c8:	48 8b 85 60 ff ff ff 	mov    -0xa0(%rbp),%rax
   606cf:	31 d2                	xor    %edx,%edx
   606d1:	4c 89 ff             	mov    %r15,%rdi
   606d4:	48 c7 85 70 ff ff ff 	movq   $0xb6,-0x90(%rbp)
   606db:	b6 00 00 00 
   606df:	48 89 45 a0          	mov    %rax,-0x60(%rbp)
   606e3:	48 8d 85 70 ff ff ff 	lea    -0x90(%rbp),%rax
   606ea:	48 89 c6             	mov    %rax,%rsi
   606ed:	48 89 85 48 ff ff ff 	mov    %rax,-0xb8(%rbp)
   606f4:	e8 f7 f9 fb ff       	call   200f0 <spdlog::details::os::thread_id()::tid@@Base+0x200e8>
   606f9:	48 8b 95 70 ff ff ff 	mov    -0x90(%rbp),%rdx
   60700:	b9 7b 7d 00 00       	mov    $0x7d7b,%ecx
   60705:	66 0f 6f 05 43 e9 00 	movdqa 0xe943(%rip),%xmm0        # 6f050 <std::_Sp_make_shared_tag::_S_ti()::__tag@@Base+0x7a0>
   6070c:	00 
   6070d:	48 89 45 a0          	mov    %rax,-0x60(%rbp)
   60711:	48 8d 7d 90          	lea    -0x70(%rbp),%rdi
   60715:	48 89 55 b0          	mov    %rdx,-0x50(%rbp)
   60719:	0f 11 00             	movups %xmm0,(%rax)
   6071c:	66 0f 6f 05 3c e9 00 	movdqa 0xe93c(%rip),%xmm0        # 6f060 <std::_Sp_make_shared_tag::_S_ti()::__tag@@Base+0x7b0>
   60723:	00 
   60724:	c7 80 b0 00 00 00 6e 	movl   $0x203a746e,0xb0(%rax)
   6072b:	74 3a 20 
   6072e:	0f 11 40 10          	movups %xmm0,0x10(%rax)
   60732:	66 0f 6f 05 36 e9 00 	movdqa 0xe936(%rip),%xmm0        # 6f070 <std::_Sp_make_shared_tag::_S_ti()::__tag@@Base+0x7c0>
   60739:	00 
   6073a:	66 89 88 b4 00 00 00 	mov    %cx,0xb4(%rax)
   60741:	0f 11 40 20          	movups %xmm0,0x20(%rax)
   60745:	66 0f 6f 05 33 e9 00 	movdqa 0xe933(%rip),%xmm0        # 6f080 <std::_Sp_make_shared_tag::_S_ti()::__tag@@Base+0x7d0>
   6074c:	00 
   6074d:	0f 11 40 30          	movups %xmm0,0x30(%rax)
   60751:	66 0f 6f 05 37 e9 00 	movdqa 0xe937(%rip),%xmm0        # 6f090 <std::_Sp_make_shared_tag::_S_ti()::__tag@@Base+0x7e0>
   60758:	00 
   60759:	0f 11 40 40          	movups %xmm0,0x40(%rax)
   6075d:	66 0f 6f 05 3b e9 00 	movdqa 0xe93b(%rip),%xmm0        # 6f0a0 <std::_Sp_make_shared_tag::_S_ti()::__tag@@Base+0x7f0>
   60764:	00 
   60765:	0f 11 40 50          	movups %xmm0,0x50(%rax)
   60769:	66 0f 6f 05 3f e9 00 	movdqa 0xe93f(%rip),%xmm0        # 6f0b0 <std::_Sp_make_shared_tag::_S_ti()::__tag@@Base+0x800>
   60770:	00 
   60771:	0f 11 40 60          	movups %xmm0,0x60(%rax)
   60775:	66 0f 6f 05 43 e9 00 	movdqa 0xe943(%rip),%xmm0        # 6f0c0 <std::_Sp_make_shared_tag::_S_ti()::__tag@@Base+0x810>
   6077c:	00 
   6077d:	0f 11 40 70          	movups %xmm0,0x70(%rax)
   60781:	66 0f 6f 05 47 e9 00 	movdqa 0xe947(%rip),%xmm0        # 6f0d0 <std::_Sp_make_shared_tag::_S_ti()::__tag@@Base+0x820>
   60788:	00 
   60789:	0f 11 80 80 00 00 00 	movups %xmm0,0x80(%rax)
   60790:	66 0f 6f 05 48 e9 00 	movdqa 0xe948(%rip),%xmm0        # 6f0e0 <std::_Sp_make_shared_tag::_S_ti()::__tag@@Base+0x830>
   60797:	00 
   60798:	0f 11 80 90 00 00 00 	movups %xmm0,0x90(%rax)
   6079f:	66 0f 6f 05 49 e9 00 	movdqa 0xe949(%rip),%xmm0        # 6f0f0 <std::_Sp_make_shared_tag::_S_ti()::__tag@@Base+0x840>
   607a6:	00 
   607a7:	0f 11 80 a0 00 00 00 	movups %xmm0,0xa0(%rax)
   607ae:	48 8b 85 70 ff ff ff 	mov    -0x90(%rbp),%rax
   607b5:	48 8b 55 a0          	mov    -0x60(%rbp),%rdx
   607b9:	48 89 45 a8          	mov    %rax,-0x58(%rbp)
   607bd:	c6 04 02 00          	movb   $0x0,(%rdx,%rax,1)
   607c1:	48 8b 85 68 ff ff ff 	mov    -0x98(%rbp),%rax
   607c8:	48 8b 80 18 02 00 00 	mov    0x218(%rax),%rax
   607cf:	48 89 bd 50 ff ff ff 	mov    %rdi,-0xb0(%rbp)
   607d6:	48 89 7d 80          	mov    %rdi,-0x80(%rbp)
   607da:	48 89 85 38 ff ff ff 	mov    %rax,-0xc8(%rbp)
   607e1:	48 85 c0             	test   %rax,%rax
   607e4:	0f 84 c6 02 00 00    	je     60ab0 <EngineBFM::process_compute_dynamic_ecb(engine_ctxt_t*, unsigned int)@@Base+0x5d0>
   607ea:	48 89 c7             	mov    %rax,%rdi
   607ed:	e8 7e ec fb ff       	call   1f470 <spdlog::details::os::thread_id()::tid@@Base+0x1f468>
   607f2:	48 89 85 30 ff ff ff 	mov    %rax,-0xd0(%rbp)
   607f9:	48 89 85 70 ff ff ff 	mov    %rax,-0x90(%rbp)
   60800:	48 83 f8 0f          	cmp    $0xf,%rax
   60804:	0f 87 4e 02 00 00    	ja     60a58 <EngineBFM::process_compute_dynamic_ecb(engine_ctxt_t*, unsigned int)@@Base+0x578>
   6080a:	48 83 f8 01          	cmp    $0x1,%rax
   6080e:	0f 85 ec 00 00 00    	jne    60900 <EngineBFM::process_compute_dynamic_ecb(engine_ctxt_t*, unsigned int)@@Base+0x420>
   60814:	48 8b bd 38 ff ff ff 	mov    -0xc8(%rbp),%rdi
   6081b:	0f b6 17             	movzbl (%rdi),%edx
   6081e:	88 55 90             	mov    %dl,-0x70(%rbp)
   60821:	48 8b 95 50 ff ff ff 	mov    -0xb0(%rbp),%rdx
   60828:	48 89 45 88          	mov    %rax,-0x78(%rbp)
   6082c:	48 8b b5 58 ff ff ff 	mov    -0xa8(%rbp),%rsi
   60833:	c6 04 02 00          	movb   $0x0,(%rdx,%rax,1)
   60837:	48 8b bd 48 ff ff ff 	mov    -0xb8(%rbp),%rdi
   6083e:	4c 89 fa             	mov    %r15,%rdx
   60841:	e8 2a f2 fb ff       	call   1fa70 <spdlog::details::os::thread_id()::tid@@Base+0x1fa68>
   60846:	48 8b 95 68 ff ff ff 	mov    -0x98(%rbp),%rdx
   6084d:	48 8b bd 70 ff ff ff 	mov    -0x90(%rbp),%rdi
   60854:	be 02 00 00 00       	mov    $0x2,%esi
   60859:	48 8d 82 e4 03 00 00 	lea    0x3e4(%rdx),%rax
   60860:	48 8d 8a d4 03 00 00 	lea    0x3d4(%rdx),%rcx
   60867:	50                   	push   %rax
   60868:	48 8d 82 e0 03 00 00 	lea    0x3e0(%rdx),%rax
   6086f:	4c 8d 8a dc 03 00 00 	lea    0x3dc(%rdx),%r9
   60876:	50                   	push   %rax
   60877:	4c 8d 82 d8 03 00 00 	lea    0x3d8(%rdx),%r8
   6087e:	48 8d 15 43 b2 00 00 	lea    0xb243(%rip),%rdx        # 6bac8 <EngineBFM::dma_wait_for_idle()@@Base+0x29a8>
   60885:	e8 86 f1 fb ff       	call   1fa10 <spdlog::details::os::thread_id()::tid@@Base+0x1fa08>
   6088a:	48 8b bd 78 ff ff ff 	mov    -0x88(%rbp),%rdi
   60891:	58                   	pop    %rax
   60892:	5a                   	pop    %rdx
   60893:	48 85 ff             	test   %rdi,%rdi
   60896:	74 05                	je     6089d <EngineBFM::process_compute_dynamic_ecb(engine_ctxt_t*, unsigned int)@@Base+0x3bd>
   60898:	e8 63 ef fb ff       	call   1f800 <spdlog::details::os::thread_id()::tid@@Base+0x1f7f8>
   6089d:	48 8b 7d 80          	mov    -0x80(%rbp),%rdi
   608a1:	48 3b bd 50 ff ff ff 	cmp    -0xb0(%rbp),%rdi
   608a8:	74 0d                	je     608b7 <EngineBFM::process_compute_dynamic_ecb(engine_ctxt_t*, unsigned int)@@Base+0x3d7>
   608aa:	48 8b 45 90          	mov    -0x70(%rbp),%rax
   608ae:	48 8d 70 01          	lea    0x1(%rax),%rsi
   608b2:	e8 69 f1 fb ff       	call   1fa20 <spdlog::details::os::thread_id()::tid@@Base+0x1fa18>
   608b7:	48 8b 7d a0          	mov    -0x60(%rbp),%rdi
   608bb:	48 3b bd 60 ff ff ff 	cmp    -0xa0(%rbp),%rdi
   608c2:	74 0d                	je     608d1 <EngineBFM::process_compute_dynamic_ecb(engine_ctxt_t*, unsigned int)@@Base+0x3f1>
   608c4:	48 8b 45 b0          	mov    -0x50(%rbp),%rax
   608c8:	48 8d 70 01          	lea    0x1(%rax),%rsi
   608cc:	e8 4f f1 fb ff       	call   1fa20 <spdlog::details::os::thread_id()::tid@@Base+0x1fa18>
   608d1:	48 8b b5 40 ff ff ff 	mov    -0xc0(%rbp),%rsi
   608d8:	48 8b bd 68 ff ff ff 	mov    -0x98(%rbp),%rdi
   608df:	e8 1c f2 fb ff       	call   1fb00 <spdlog::details::os::thread_id()::tid@@Base+0x1faf8>
   608e4:	48 8b 85 68 ff ff ff 	mov    -0x98(%rbp),%rax
   608eb:	83 80 d4 03 00 00 01 	addl   $0x1,0x3d4(%rax)
   608f2:	e9 e7 fc ff ff       	jmp    605de <EngineBFM::process_compute_dynamic_ecb(engine_ctxt_t*, unsigned int)@@Base+0xfe>
   608f7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
   608fe:	00 00 
   60900:	48 83 bd 30 ff ff ff 	cmpq   $0x0,-0xd0(%rbp)
   60907:	00 
   60908:	0f 85 94 01 00 00    	jne    60aa2 <EngineBFM::process_compute_dynamic_ecb(engine_ctxt_t*, unsigned int)@@Base+0x5c2>
   6090e:	48 8b 95 50 ff ff ff 	mov    -0xb0(%rbp),%rdx
   60915:	e9 0e ff ff ff       	jmp    60828 <EngineBFM::process_compute_dynamic_ecb(engine_ctxt_t*, unsigned int)@@Base+0x348>
   6091a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
   60920:	83 88 14 0b 00 00 02 	orl    $0x2,0xb14(%rax)
   60927:	4c 8d 25 55 c6 00 00 	lea    0xc655(%rip),%r12        # 6cf83 <EngineBFM::dma_wait_for_idle()@@Base+0x3e63>
   6092e:	48 89 c3             	mov    %rax,%rbx
   60931:	4c 89 ff             	mov    %r15,%rdi
   60934:	c7 80 18 0b 00 00 00 	movl   $0x0,0xb18(%rax)
   6093b:	00 00 00 
   6093e:	4c 89 e6             	mov    %r12,%rsi
   60941:	e8 0a d6 ff ff       	call   5df50 <EngineBFM::config_and_arm_monitor(hw_qman_msg_short_t*, hw_qman_msg_short_t*, virt_sob_ids_t*, unsigned char, unsigned int*)@@Base+0xa10>
   60946:	48 8b b3 18 02 00 00 	mov    0x218(%rbx),%rsi
   6094d:	48 8b 9d 58 ff ff ff 	mov    -0xa8(%rbp),%rbx
   60954:	48 89 df             	mov    %rbx,%rdi
   60957:	e8 f4 d5 ff ff       	call   5df50 <EngineBFM::config_and_arm_monitor(hw_qman_msg_short_t*, hw_qman_msg_short_t*, virt_sob_ids_t*, unsigned char, unsigned int*)@@Base+0xa10>
   6095c:	48 8d bd 70 ff ff ff 	lea    -0x90(%rbp),%rdi
   60963:	4c 89 fa             	mov    %r15,%rdx
   60966:	48 89 de             	mov    %rbx,%rsi
   60969:	e8 02 f1 fb ff       	call   1fa70 <spdlog::details::os::thread_id()::tid@@Base+0x1fa68>
   6096e:	48 8b bd 70 ff ff ff 	mov    -0x90(%rbp),%rdi
   60975:	4c 89 e2             	mov    %r12,%rdx
   60978:	be 02 00 00 00       	mov    $0x2,%esi
   6097d:	e8 5e f8 fb ff       	call   201e0 <spdlog::details::os::thread_id()::tid@@Base+0x201d8>
   60982:	48 8b bd 78 ff ff ff 	mov    -0x88(%rbp),%rdi
   60989:	48 85 ff             	test   %rdi,%rdi
   6098c:	74 05                	je     60993 <EngineBFM::process_compute_dynamic_ecb(engine_ctxt_t*, unsigned int)@@Base+0x4b3>
   6098e:	e8 6d ee fb ff       	call   1f800 <spdlog::details::os::thread_id()::tid@@Base+0x1f7f8>
   60993:	48 8b 7d 80          	mov    -0x80(%rbp),%rdi
   60997:	48 8d 45 90          	lea    -0x70(%rbp),%rax
   6099b:	48 39 c7             	cmp    %rax,%rdi
   6099e:	74 0d                	je     609ad <EngineBFM::process_compute_dynamic_ecb(engine_ctxt_t*, unsigned int)@@Base+0x4cd>
   609a0:	48 8b 45 90          	mov    -0x70(%rbp),%rax
   609a4:	48 8d 70 01          	lea    0x1(%rax),%rsi
   609a8:	e8 73 f0 fb ff       	call   1fa20 <spdlog::details::os::thread_id()::tid@@Base+0x1fa18>
   609ad:	48 8b 7d a0          	mov    -0x60(%rbp),%rdi
   609b1:	48 3b bd 60 ff ff ff 	cmp    -0xa0(%rbp),%rdi
   609b8:	74 0d                	je     609c7 <EngineBFM::process_compute_dynamic_ecb(engine_ctxt_t*, unsigned int)@@Base+0x4e7>
   609ba:	48 8b 45 b0          	mov    -0x50(%rbp),%rax
   609be:	48 8d 70 01          	lea    0x1(%rax),%rsi
   609c2:	e8 59 f0 fb ff       	call   1fa20 <spdlog::details::os::thread_id()::tid@@Base+0x1fa18>
   609c7:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
   609cb:	64 48 2b 04 25 28 00 	sub    %fs:0x28,%rax
   609d2:	00 00 
   609d4:	0f 85 d1 00 00 00    	jne    60aab <EngineBFM::process_compute_dynamic_ecb(engine_ctxt_t*, unsigned int)@@Base+0x5cb>
   609da:	48 8d 65 d8          	lea    -0x28(%rbp),%rsp
   609de:	5b                   	pop    %rbx
   609df:	41 5c                	pop    %r12
   609e1:	41 5d                	pop    %r13
   609e3:	41 5e                	pop    %r14
   609e5:	41 5f                	pop    %r15
   609e7:	5d                   	pop    %rbp
   609e8:	c3                   	ret    
   609e9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
   609f0:	48 8b 9d 68 ff ff ff 	mov    -0x98(%rbp),%rbx
   609f7:	4c 8d 25 92 b0 00 00 	lea    0xb092(%rip),%r12        # 6ba90 <EngineBFM::dma_wait_for_idle()@@Base+0x2970>
   609fe:	4c 89 ff             	mov    %r15,%rdi
   60a01:	4c 89 e6             	mov    %r12,%rsi
   60a04:	c7 83 18 0b 00 00 00 	movl   $0x0,0xb18(%rbx)
   60a0b:	00 00 00 
   60a0e:	e8 3d d5 ff ff       	call   5df50 <EngineBFM::config_and_arm_monitor(hw_qman_msg_short_t*, hw_qman_msg_short_t*, virt_sob_ids_t*, unsigned char, unsigned int*)@@Base+0xa10>
   60a13:	48 8b b3 18 02 00 00 	mov    0x218(%rbx),%rsi
   60a1a:	48 8b 9d 58 ff ff ff 	mov    -0xa8(%rbp),%rbx
   60a21:	48 89 df             	mov    %rbx,%rdi
   60a24:	e8 27 d5 ff ff       	call   5df50 <EngineBFM::config_and_arm_monitor(hw_qman_msg_short_t*, hw_qman_msg_short_t*, virt_sob_ids_t*, unsigned char, unsigned int*)@@Base+0xa10>
   60a29:	48 8d bd 70 ff ff ff 	lea    -0x90(%rbp),%rdi
   60a30:	4c 89 fa             	mov    %r15,%rdx
   60a33:	48 89 de             	mov    %rbx,%rsi
   60a36:	e8 35 f0 fb ff       	call   1fa70 <spdlog::details::os::thread_id()::tid@@Base+0x1fa68>
   60a3b:	48 8b bd 70 ff ff ff 	mov    -0x90(%rbp),%rdi
   60a42:	4c 89 e2             	mov    %r12,%rdx
   60a45:	31 f6                	xor    %esi,%esi
   60a47:	e8 94 f7 fb ff       	call   201e0 <spdlog::details::os::thread_id()::tid@@Base+0x201d8>
   60a4c:	e9 31 ff ff ff       	jmp    60982 <EngineBFM::process_compute_dynamic_ecb(engine_ctxt_t*, unsigned int)@@Base+0x4a2>
   60a51:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
   60a58:	48 8b b5 48 ff ff ff 	mov    -0xb8(%rbp),%rsi
   60a5f:	48 8b bd 58 ff ff ff 	mov    -0xa8(%rbp),%rdi
   60a66:	31 d2                	xor    %edx,%edx
   60a68:	e8 83 f6 fb ff       	call   200f0 <spdlog::details::os::thread_id()::tid@@Base+0x200e8>
   60a6d:	48 89 45 80          	mov    %rax,-0x80(%rbp)
   60a71:	48 89 c7             	mov    %rax,%rdi
   60a74:	48 8b 85 70 ff ff ff 	mov    -0x90(%rbp),%rax
   60a7b:	48 89 45 90          	mov    %rax,-0x70(%rbp)
   60a7f:	48 8b 95 30 ff ff ff 	mov    -0xd0(%rbp),%rdx
   60a86:	48 8b b5 38 ff ff ff 	mov    -0xc8(%rbp),%rsi
   60a8d:	e8 7e ed fb ff       	call   1f810 <spdlog::details::os::thread_id()::tid@@Base+0x1f808>
   60a92:	48 8b 85 70 ff ff ff 	mov    -0x90(%rbp),%rax
   60a99:	48 8b 55 80          	mov    -0x80(%rbp),%rdx
   60a9d:	e9 86 fd ff ff       	jmp    60828 <EngineBFM::process_compute_dynamic_ecb(engine_ctxt_t*, unsigned int)@@Base+0x348>
   60aa2:	48 8b bd 50 ff ff ff 	mov    -0xb0(%rbp),%rdi
   60aa9:	eb d4                	jmp    60a7f <EngineBFM::process_compute_dynamic_ecb(engine_ctxt_t*, unsigned int)@@Base+0x59f>
   60aab:	e8 20 f0 fb ff       	call   1fad0 <spdlog::details::os::thread_id()::tid@@Base+0x1fac8>
   60ab0:	48 8d 3d 49 95 00 00 	lea    0x9549(%rip),%rdi        # 6a000 <EngineBFM::dma_wait_for_idle()@@Base+0xee0>
   60ab7:	e8 84 ec fb ff       	call   1f740 <spdlog::details::os::thread_id()::tid@@Base+0x1f738>
   60abc:	f3 0f 1e fa          	endbr64 
   60ac0:	48 89 c3             	mov    %rax,%rbx
   60ac3:	e9 9b 35 fc ff       	jmp    24063 <boost::context::detail::context_unwind(boost::context::detail::transfer_t)@@Base+0x3636>
   60ac8:	f3 0f 1e fa          	endbr64 
   60acc:	48 89 c3             	mov    %rax,%rbx
   60acf:	e9 dc 35 fc ff       	jmp    240b0 <boost::context::detail::context_unwind(boost::context::detail::transfer_t)@@Base+0x3683>
   60ad4:	f3 0f 1e fa          	endbr64 
   60ad8:	48 89 c3             	mov    %rax,%rbx
   60adb:	e9 b6 35 fc ff       	jmp    24096 <boost::context::detail::context_unwind(boost::context::detail::transfer_t)@@Base+0x3669>
   60ae0:	f3 0f 1e fa          	endbr64 
   60ae4:	48 89 c3             	mov    %rax,%rbx
   60ae7:	e9 99 35 fc ff       	jmp    24085 <boost::context::detail::context_unwind(boost::context::detail::transfer_t)@@Base+0x3658>
   60aec:	f3 0f 1e fa          	endbr64 
   60af0:	48 89 c3             	mov    %rax,%rbx
   60af3:	e9 40 35 fc ff       	jmp    24038 <boost::context::detail::context_unwind(boost::context::detail::transfer_t)@@Base+0x360b>
   60af8:	f3 0f 1e fa          	endbr64 
   60afc:	48 89 c3             	mov    %rax,%rbx
   60aff:	e9 45 35 fc ff       	jmp    24049 <boost::context::detail::context_unwind(boost::context::detail::transfer_t)@@Base+0x361c>
   60b04:	f3 0f 1e fa          	endbr64 
   60b08:	48 89 c3             	mov    %rax,%rbx
   60b0b:	e9 c2 35 fc ff       	jmp    240d2 <boost::context::detail::context_unwind(boost::context::detail::transfer_t)@@Base+0x36a5>
   60b10:	f3 0f 1e fa          	endbr64 
   60b14:	48 89 c3             	mov    %rax,%rbx
   60b17:	e9 03 36 fc ff       	jmp    2411f <boost::context::detail::context_unwind(boost::context::detail::transfer_t)@@Base+0x36f2>
   60b1c:	f3 0f 1e fa          	endbr64 
   60b20:	48 89 c3             	mov    %rax,%rbx
   60b23:	e9 d5 35 fc ff       	jmp    240fd <boost::context::detail::context_unwind(boost::context::detail::transfer_t)@@Base+0x36d0>
   60b28:	f3 0f 1e fa          	endbr64 
   60b2c:	48 89 c3             	mov    %rax,%rbx
   60b2f:	e9 16 36 fc ff       	jmp    2414a <boost::context::detail::context_unwind(boost::context::detail::transfer_t)@@Base+0x371d>
   60b34:	f3 0f 1e fa          	endbr64 
   60b38:	48 89 c3             	mov    %rax,%rbx
   60b3b:	e9 f0 35 fc ff       	jmp    24130 <boost::context::detail::context_unwind(boost::context::detail::transfer_t)@@Base+0x3703>
   60b40:	f3 0f 1e fa          	endbr64 
   60b44:	48 89 c3             	mov    %rax,%rbx
   60b47:	e9 97 35 fc ff       	jmp    240e3 <boost::context::detail::context_unwind(boost::context::detail::transfer_t)@@Base+0x36b6>
