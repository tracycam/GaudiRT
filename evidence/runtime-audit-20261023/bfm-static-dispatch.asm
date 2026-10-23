
/home/rate/llm/Gaudi-LLM/GaudiRT/evidence/runtime-audit-20261023/binaries/libeng_mme_cmpt_bfm.so:     file format elf64-x86-64


Disassembly of section .text:

0000000000063200 <EngineBFM::process_compute_static_ecb(engine_ctxt_t*, unsigned int)@@Base>:
   63200:	f3 0f 1e fa          	endbr64 
   63204:	55                   	push   %rbp
   63205:	48 89 e5             	mov    %rsp,%rbp
   63208:	41 57                	push   %r15
   6320a:	41 56                	push   %r14
   6320c:	4c 8d 7d a0          	lea    -0x60(%rbp),%r15
   63210:	49 89 f6             	mov    %rsi,%r14
   63213:	41 55                	push   %r13
   63215:	41 54                	push   %r12
   63217:	53                   	push   %rbx
   63218:	48 81 ec a8 00 00 00 	sub    $0xa8,%rsp
   6321f:	48 89 bd 68 ff ff ff 	mov    %rdi,-0x98(%rbp)
   63226:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
   6322d:	00 00 
   6322f:	48 89 45 c8          	mov    %rax,-0x38(%rbp)
   63233:	48 89 f8             	mov    %rdi,%rax
   63236:	48 81 c7 38 0a 00 00 	add    $0xa38,%rdi
   6323d:	48 89 bd 40 ff ff ff 	mov    %rdi,-0xc0(%rbp)
   63244:	44 8b a8 14 0b 00 00 	mov    0xb14(%rax),%r13d
   6324b:	48 8d 45 80          	lea    -0x80(%rbp),%rax
   6324f:	48 89 85 58 ff ff ff 	mov    %rax,-0xa8(%rbp)
   63256:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
   6325a:	48 89 85 60 ff ff ff 	mov    %rax,-0xa0(%rbp)
   63261:	41 83 e5 02          	and    $0x2,%r13d
   63265:	0f 1f 00             	nopl   (%rax)
   63268:	48 8b 85 68 ff ff ff 	mov    -0x98(%rbp),%rax
   6326f:	8b 98 48 0a 00 00    	mov    0xa48(%rax),%ebx
   63275:	3b 98 44 0a 00 00    	cmp    0xa44(%rax),%ebx
   6327b:	0f 84 3f 03 00 00    	je     635c0 <EngineBFM::process_compute_static_ecb(engine_ctxt_t*, unsigned int)@@Base+0x3c0>
   63281:	48 03 98 38 0a 00 00 	add    0xa38(%rax),%rbx
   63288:	0f b6 03             	movzbl (%rbx),%eax
   6328b:	83 e0 0f             	and    $0xf,%eax
   6328e:	3c 05                	cmp    $0x5,%al
   63290:	0f 84 f2 00 00 00    	je     63388 <EngineBFM::process_compute_static_ecb(engine_ctxt_t*, unsigned int)@@Base+0x188>
   63296:	3c 08                	cmp    $0x8,%al
   63298:	0f 84 d2 00 00 00    	je     63370 <EngineBFM::process_compute_static_ecb(engine_ctxt_t*, unsigned int)@@Base+0x170>
   6329e:	3c 01                	cmp    $0x1,%al
   632a0:	0f 84 fa 00 00 00    	je     633a0 <EngineBFM::process_compute_static_ecb(engine_ctxt_t*, unsigned int)@@Base+0x1a0>
   632a6:	48 8b bd 68 ff ff ff 	mov    -0x98(%rbp),%rdi
   632ad:	4c 89 f6             	mov    %r14,%rsi
   632b0:	e8 6b cb fb ff       	call   1fe20 <spdlog::details::os::thread_id()::tid@@Base+0x1fe18>
   632b5:	45 85 ed             	test   %r13d,%r13d
   632b8:	41 0f 94 c4          	sete   %r12b
   632bc:	85 c0                	test   %eax,%eax
   632be:	74 09                	je     632c9 <EngineBFM::process_compute_static_ecb(engine_ctxt_t*, unsigned int)@@Base+0xc9>
   632c0:	45 84 e4             	test   %r12b,%r12b
   632c3:	0f 85 c7 03 00 00    	jne    63690 <EngineBFM::process_compute_static_ecb(engine_ctxt_t*, unsigned int)@@Base+0x490>
   632c9:	48 8b b5 68 ff ff ff 	mov    -0x98(%rbp),%rsi
   632d0:	8b 96 48 0a 00 00    	mov    0xa48(%rsi),%edx
   632d6:	89 d1                	mov    %edx,%ecx
   632d8:	81 e1 ff 01 00 00    	and    $0x1ff,%ecx
   632de:	89 8e 48 0a 00 00    	mov    %ecx,0xa48(%rsi)
   632e4:	84 d2                	test   %dl,%dl
   632e6:	75 12                	jne    632fa <EngineBFM::process_compute_static_ecb(engine_ctxt_t*, unsigned int)@@Base+0xfa>
   632e8:	8b 8e 40 0a 00 00    	mov    0xa40(%rsi),%ecx
   632ee:	85 c9                	test   %ecx,%ecx
   632f0:	74 08                	je     632fa <EngineBFM::process_compute_static_ecb(engine_ctxt_t*, unsigned int)@@Base+0xfa>
   632f2:	85 c0                	test   %eax,%eax
   632f4:	0f 84 be 00 00 00    	je     633b8 <EngineBFM::process_compute_static_ecb(engine_ctxt_t*, unsigned int)@@Base+0x1b8>
   632fa:	f6 03 10             	testb  $0x10,(%rbx)
   632fd:	0f 84 65 ff ff ff    	je     63268 <EngineBFM::process_compute_static_ecb(engine_ctxt_t*, unsigned int)@@Base+0x68>
   63303:	45 84 e4             	test   %r12b,%r12b
   63306:	0f 84 5c ff ff ff    	je     63268 <EngineBFM::process_compute_static_ecb(engine_ctxt_t*, unsigned int)@@Base+0x68>
   6330c:	48 8b 9d 68 ff ff ff 	mov    -0x98(%rbp),%rbx
   63313:	4c 8d 25 b5 9c 00 00 	lea    0x9cb5(%rip),%r12        # 6cfcf <EngineBFM::dma_wait_for_idle()@@Base+0x3eaf>
   6331a:	4c 89 ff             	mov    %r15,%rdi
   6331d:	4c 89 e6             	mov    %r12,%rsi
   63320:	c7 83 18 0b 00 00 01 	movl   $0x1,0xb18(%rbx)
   63327:	00 00 00 
   6332a:	e8 71 dc ff ff       	call   60fa0 <void spdlog::logger::log<unsigned int, unsigned int, unsigned int, unsigned int, unsigned int>(spdlog::level::level_enum, char const*, unsigned int const&, unsigned int const&, unsigned int const&, unsigned int const&, unsigned int const&)@@Base+0x450>
   6332f:	48 8b b3 18 02 00 00 	mov    0x218(%rbx),%rsi
   63336:	48 8b 9d 58 ff ff ff 	mov    -0xa8(%rbp),%rbx
   6333d:	48 89 df             	mov    %rbx,%rdi
   63340:	e8 5b dc ff ff       	call   60fa0 <void spdlog::logger::log<unsigned int, unsigned int, unsigned int, unsigned int, unsigned int>(spdlog::level::level_enum, char const*, unsigned int const&, unsigned int const&, unsigned int const&, unsigned int const&, unsigned int const&)@@Base+0x450>
   63345:	48 8d bd 70 ff ff ff 	lea    -0x90(%rbp),%rdi
   6334c:	4c 89 fa             	mov    %r15,%rdx
   6334f:	48 89 de             	mov    %rbx,%rsi
   63352:	e8 19 c7 fb ff       	call   1fa70 <spdlog::details::os::thread_id()::tid@@Base+0x1fa68>
   63357:	48 8b bd 70 ff ff ff 	mov    -0x90(%rbp),%rdi
   6335e:	4c 89 e2             	mov    %r12,%rdx
   63361:	be 02 00 00 00       	mov    $0x2,%esi
   63366:	e8 75 ce fb ff       	call   201e0 <spdlog::details::os::thread_id()::tid@@Base+0x201d8>
   6336b:	e9 b2 02 00 00       	jmp    63622 <EngineBFM::process_compute_static_ecb(engine_ctxt_t*, unsigned int)@@Base+0x422>
   63370:	48 8b bd 68 ff ff ff 	mov    -0x98(%rbp),%rdi
   63377:	4c 89 f6             	mov    %r14,%rsi
   6337a:	e8 01 c6 fb ff       	call   1f980 <spdlog::details::os::thread_id()::tid@@Base+0x1f978>
   6337f:	e9 31 ff ff ff       	jmp    632b5 <EngineBFM::process_compute_static_ecb(engine_ctxt_t*, unsigned int)@@Base+0xb5>
   63384:	0f 1f 40 00          	nopl   0x0(%rax)
   63388:	48 8b bd 68 ff ff ff 	mov    -0x98(%rbp),%rdi
   6338f:	4c 89 f6             	mov    %r14,%rsi
   63392:	e8 79 c3 fb ff       	call   1f710 <spdlog::details::os::thread_id()::tid@@Base+0x1f708>
   63397:	e9 19 ff ff ff       	jmp    632b5 <EngineBFM::process_compute_static_ecb(engine_ctxt_t*, unsigned int)@@Base+0xb5>
   6339c:	0f 1f 40 00          	nopl   0x0(%rax)
   633a0:	48 8b bd 68 ff ff ff 	mov    -0x98(%rbp),%rdi
   633a7:	4c 89 f6             	mov    %r14,%rsi
   633aa:	e8 c1 cd fb ff       	call   20170 <spdlog::details::os::thread_id()::tid@@Base+0x20168>
   633af:	e9 01 ff ff ff       	jmp    632b5 <EngineBFM::process_compute_static_ecb(engine_ctxt_t*, unsigned int)@@Base+0xb5>
   633b4:	0f 1f 40 00          	nopl   0x0(%rax)
   633b8:	48 8b 85 60 ff ff ff 	mov    -0xa0(%rbp),%rax
   633bf:	31 d2                	xor    %edx,%edx
   633c1:	4c 89 ff             	mov    %r15,%rdi
   633c4:	48 c7 85 70 ff ff ff 	movq   $0x80,-0x90(%rbp)
   633cb:	80 00 00 00 
   633cf:	48 89 45 a0          	mov    %rax,-0x60(%rbp)
   633d3:	48 8d 85 70 ff ff ff 	lea    -0x90(%rbp),%rax
   633da:	48 89 c6             	mov    %rax,%rsi
   633dd:	48 89 85 48 ff ff ff 	mov    %rax,-0xb8(%rbp)
   633e4:	e8 07 cd fb ff       	call   200f0 <spdlog::details::os::thread_id()::tid@@Base+0x200e8>
   633e9:	48 8b 95 70 ff ff ff 	mov    -0x90(%rbp),%rdx
   633f0:	48 8d 7d 90          	lea    -0x70(%rbp),%rdi
   633f4:	66 0f 6f 05 54 bc 00 	movdqa 0xbc54(%rip),%xmm0        # 6f050 <std::_Sp_make_shared_tag::_S_ti()::__tag@@Base+0x7a0>
   633fb:	00 
   633fc:	48 89 45 a0          	mov    %rax,-0x60(%rbp)
   63400:	48 89 55 b0          	mov    %rdx,-0x50(%rbp)
   63404:	0f 11 00             	movups %xmm0,(%rax)
   63407:	66 0f 6f 05 41 be 00 	movdqa 0xbe41(%rip),%xmm0        # 6f250 <std::_Sp_make_shared_tag::_S_ti()::__tag@@Base+0x9a0>
   6340e:	00 
   6340f:	48 89 bd 50 ff ff ff 	mov    %rdi,-0xb0(%rbp)
   63416:	0f 11 40 10          	movups %xmm0,0x10(%rax)
   6341a:	66 0f 6f 05 3e be 00 	movdqa 0xbe3e(%rip),%xmm0        # 6f260 <std::_Sp_make_shared_tag::_S_ti()::__tag@@Base+0x9b0>
   63421:	00 
   63422:	0f 11 40 20          	movups %xmm0,0x20(%rax)
   63426:	66 0f 6f 05 42 be 00 	movdqa 0xbe42(%rip),%xmm0        # 6f270 <std::_Sp_make_shared_tag::_S_ti()::__tag@@Base+0x9c0>
   6342d:	00 
   6342e:	0f 11 40 30          	movups %xmm0,0x30(%rax)
   63432:	66 0f 6f 05 46 be 00 	movdqa 0xbe46(%rip),%xmm0        # 6f280 <std::_Sp_make_shared_tag::_S_ti()::__tag@@Base+0x9d0>
   63439:	00 
   6343a:	0f 11 40 40          	movups %xmm0,0x40(%rax)
   6343e:	66 0f 6f 05 4a be 00 	movdqa 0xbe4a(%rip),%xmm0        # 6f290 <std::_Sp_make_shared_tag::_S_ti()::__tag@@Base+0x9e0>
   63445:	00 
   63446:	0f 11 40 50          	movups %xmm0,0x50(%rax)
   6344a:	66 0f 6f 05 4e be 00 	movdqa 0xbe4e(%rip),%xmm0        # 6f2a0 <std::_Sp_make_shared_tag::_S_ti()::__tag@@Base+0x9f0>
   63451:	00 
   63452:	0f 11 40 60          	movups %xmm0,0x60(%rax)
   63456:	66 0f 6f 05 52 be 00 	movdqa 0xbe52(%rip),%xmm0        # 6f2b0 <std::_Sp_make_shared_tag::_S_ti()::__tag@@Base+0xa00>
   6345d:	00 
   6345e:	0f 11 40 70          	movups %xmm0,0x70(%rax)
   63462:	48 8b 85 70 ff ff ff 	mov    -0x90(%rbp),%rax
   63469:	48 8b 55 a0          	mov    -0x60(%rbp),%rdx
   6346d:	48 89 45 a8          	mov    %rax,-0x58(%rbp)
   63471:	c6 04 02 00          	movb   $0x0,(%rdx,%rax,1)
   63475:	48 8b 85 68 ff ff ff 	mov    -0x98(%rbp),%rax
   6347c:	48 89 7d 80          	mov    %rdi,-0x80(%rbp)
   63480:	48 8b 80 18 02 00 00 	mov    0x218(%rax),%rax
   63487:	48 89 85 38 ff ff ff 	mov    %rax,-0xc8(%rbp)
   6348e:	48 85 c0             	test   %rax,%rax
   63491:	0f 84 b9 02 00 00    	je     63750 <EngineBFM::process_compute_static_ecb(engine_ctxt_t*, unsigned int)@@Base+0x550>
   63497:	48 89 c7             	mov    %rax,%rdi
   6349a:	e8 d1 bf fb ff       	call   1f470 <spdlog::details::os::thread_id()::tid@@Base+0x1f468>
   6349f:	48 89 85 30 ff ff ff 	mov    %rax,-0xd0(%rbp)
   634a6:	48 89 85 70 ff ff ff 	mov    %rax,-0x90(%rbp)
   634ad:	48 83 f8 0f          	cmp    $0xf,%rax
   634b1:	0f 87 41 02 00 00    	ja     636f8 <EngineBFM::process_compute_static_ecb(engine_ctxt_t*, unsigned int)@@Base+0x4f8>
   634b7:	48 83 f8 01          	cmp    $0x1,%rax
   634bb:	0f 85 df 00 00 00    	jne    635a0 <EngineBFM::process_compute_static_ecb(engine_ctxt_t*, unsigned int)@@Base+0x3a0>
   634c1:	48 8b bd 38 ff ff ff 	mov    -0xc8(%rbp),%rdi
   634c8:	0f b6 17             	movzbl (%rdi),%edx
   634cb:	88 55 90             	mov    %dl,-0x70(%rbp)
   634ce:	48 8b 95 50 ff ff ff 	mov    -0xb0(%rbp),%rdx
   634d5:	48 89 45 88          	mov    %rax,-0x78(%rbp)
   634d9:	48 8b b5 58 ff ff ff 	mov    -0xa8(%rbp),%rsi
   634e0:	c6 04 02 00          	movb   $0x0,(%rdx,%rax,1)
   634e4:	48 8b bd 48 ff ff ff 	mov    -0xb8(%rbp),%rdi
   634eb:	4c 89 fa             	mov    %r15,%rdx
   634ee:	e8 7d c5 fb ff       	call   1fa70 <spdlog::details::os::thread_id()::tid@@Base+0x1fa68>
   634f3:	48 8b 95 68 ff ff ff 	mov    -0x98(%rbp),%rdx
   634fa:	48 83 ec 08          	sub    $0x8,%rsp
   634fe:	be 02 00 00 00       	mov    $0x2,%esi
   63503:	48 8b bd 70 ff ff ff 	mov    -0x90(%rbp),%rdi
   6350a:	48 8d 82 d0 03 00 00 	lea    0x3d0(%rdx),%rax
   63511:	48 8d 8a c4 03 00 00 	lea    0x3c4(%rdx),%rcx
   63518:	50                   	push   %rax
   63519:	4c 8d 8a cc 03 00 00 	lea    0x3cc(%rdx),%r9
   63520:	4c 8d 82 c8 03 00 00 	lea    0x3c8(%rdx),%r8
   63527:	48 8d 15 9a 8a 00 00 	lea    0x8a9a(%rip),%rdx        # 6bfc8 <EngineBFM::dma_wait_for_idle()@@Base+0x2ea8>
   6352e:	e8 9d c1 fb ff       	call   1f6d0 <spdlog::details::os::thread_id()::tid@@Base+0x1f6c8>
   63533:	48 8b bd 78 ff ff ff 	mov    -0x88(%rbp),%rdi
   6353a:	58                   	pop    %rax
   6353b:	5a                   	pop    %rdx
   6353c:	48 85 ff             	test   %rdi,%rdi
   6353f:	74 05                	je     63546 <EngineBFM::process_compute_static_ecb(engine_ctxt_t*, unsigned int)@@Base+0x346>
   63541:	e8 ba c2 fb ff       	call   1f800 <spdlog::details::os::thread_id()::tid@@Base+0x1f7f8>
   63546:	48 8b 7d 80          	mov    -0x80(%rbp),%rdi
   6354a:	48 3b bd 50 ff ff ff 	cmp    -0xb0(%rbp),%rdi
   63551:	74 0d                	je     63560 <EngineBFM::process_compute_static_ecb(engine_ctxt_t*, unsigned int)@@Base+0x360>
   63553:	48 8b 45 90          	mov    -0x70(%rbp),%rax
   63557:	48 8d 70 01          	lea    0x1(%rax),%rsi
   6355b:	e8 c0 c4 fb ff       	call   1fa20 <spdlog::details::os::thread_id()::tid@@Base+0x1fa18>
   63560:	48 8b 7d a0          	mov    -0x60(%rbp),%rdi
   63564:	48 3b bd 60 ff ff ff 	cmp    -0xa0(%rbp),%rdi
   6356b:	74 0d                	je     6357a <EngineBFM::process_compute_static_ecb(engine_ctxt_t*, unsigned int)@@Base+0x37a>
   6356d:	48 8b 45 b0          	mov    -0x50(%rbp),%rax
   63571:	48 8d 70 01          	lea    0x1(%rax),%rsi
   63575:	e8 a6 c4 fb ff       	call   1fa20 <spdlog::details::os::thread_id()::tid@@Base+0x1fa18>
   6357a:	48 8b b5 40 ff ff ff 	mov    -0xc0(%rbp),%rsi
   63581:	48 8b bd 68 ff ff ff 	mov    -0x98(%rbp),%rdi
   63588:	e8 73 ca fb ff       	call   20000 <spdlog::details::os::thread_id()::tid@@Base+0x1fff8>
   6358d:	48 8b 85 68 ff ff ff 	mov    -0x98(%rbp),%rax
   63594:	83 80 c4 03 00 00 01 	addl   $0x1,0x3c4(%rax)
   6359b:	e9 5a fd ff ff       	jmp    632fa <EngineBFM::process_compute_static_ecb(engine_ctxt_t*, unsigned int)@@Base+0xfa>
   635a0:	48 83 bd 30 ff ff ff 	cmpq   $0x0,-0xd0(%rbp)
   635a7:	00 
   635a8:	0f 85 94 01 00 00    	jne    63742 <EngineBFM::process_compute_static_ecb(engine_ctxt_t*, unsigned int)@@Base+0x542>
   635ae:	48 8b 95 50 ff ff ff 	mov    -0xb0(%rbp),%rdx
   635b5:	e9 1b ff ff ff       	jmp    634d5 <EngineBFM::process_compute_static_ecb(engine_ctxt_t*, unsigned int)@@Base+0x2d5>
   635ba:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
   635c0:	83 88 14 0b 00 00 01 	orl    $0x1,0xb14(%rax)
   635c7:	4c 8d 25 1e 9a 00 00 	lea    0x9a1e(%rip),%r12        # 6cfec <EngineBFM::dma_wait_for_idle()@@Base+0x3ecc>
   635ce:	48 89 c3             	mov    %rax,%rbx
   635d1:	4c 89 ff             	mov    %r15,%rdi
   635d4:	c7 80 18 0b 00 00 01 	movl   $0x1,0xb18(%rax)
   635db:	00 00 00 
   635de:	4c 89 e6             	mov    %r12,%rsi
   635e1:	e8 ba d9 ff ff       	call   60fa0 <void spdlog::logger::log<unsigned int, unsigned int, unsigned int, unsigned int, unsigned int>(spdlog::level::level_enum, char const*, unsigned int const&, unsigned int const&, unsigned int const&, unsigned int const&, unsigned int const&)@@Base+0x450>
   635e6:	48 8b b3 18 02 00 00 	mov    0x218(%rbx),%rsi
   635ed:	48 8b 9d 58 ff ff ff 	mov    -0xa8(%rbp),%rbx
   635f4:	48 89 df             	mov    %rbx,%rdi
   635f7:	e8 a4 d9 ff ff       	call   60fa0 <void spdlog::logger::log<unsigned int, unsigned int, unsigned int, unsigned int, unsigned int>(spdlog::level::level_enum, char const*, unsigned int const&, unsigned int const&, unsigned int const&, unsigned int const&, unsigned int const&)@@Base+0x450>
   635fc:	48 8d bd 70 ff ff ff 	lea    -0x90(%rbp),%rdi
   63603:	4c 89 fa             	mov    %r15,%rdx
   63606:	48 89 de             	mov    %rbx,%rsi
   63609:	e8 62 c4 fb ff       	call   1fa70 <spdlog::details::os::thread_id()::tid@@Base+0x1fa68>
   6360e:	48 8b bd 70 ff ff ff 	mov    -0x90(%rbp),%rdi
   63615:	4c 89 e2             	mov    %r12,%rdx
   63618:	be 02 00 00 00       	mov    $0x2,%esi
   6361d:	e8 be cb fb ff       	call   201e0 <spdlog::details::os::thread_id()::tid@@Base+0x201d8>
   63622:	48 8b bd 78 ff ff ff 	mov    -0x88(%rbp),%rdi
   63629:	48 85 ff             	test   %rdi,%rdi
   6362c:	74 05                	je     63633 <EngineBFM::process_compute_static_ecb(engine_ctxt_t*, unsigned int)@@Base+0x433>
   6362e:	e8 cd c1 fb ff       	call   1f800 <spdlog::details::os::thread_id()::tid@@Base+0x1f7f8>
   63633:	48 8b 7d 80          	mov    -0x80(%rbp),%rdi
   63637:	48 8d 45 90          	lea    -0x70(%rbp),%rax
   6363b:	48 39 c7             	cmp    %rax,%rdi
   6363e:	74 0d                	je     6364d <EngineBFM::process_compute_static_ecb(engine_ctxt_t*, unsigned int)@@Base+0x44d>
   63640:	48 8b 45 90          	mov    -0x70(%rbp),%rax
   63644:	48 8d 70 01          	lea    0x1(%rax),%rsi
   63648:	e8 d3 c3 fb ff       	call   1fa20 <spdlog::details::os::thread_id()::tid@@Base+0x1fa18>
   6364d:	48 8b 7d a0          	mov    -0x60(%rbp),%rdi
   63651:	48 3b bd 60 ff ff ff 	cmp    -0xa0(%rbp),%rdi
   63658:	74 0d                	je     63667 <EngineBFM::process_compute_static_ecb(engine_ctxt_t*, unsigned int)@@Base+0x467>
   6365a:	48 8b 45 b0          	mov    -0x50(%rbp),%rax
   6365e:	48 8d 70 01          	lea    0x1(%rax),%rsi
   63662:	e8 b9 c3 fb ff       	call   1fa20 <spdlog::details::os::thread_id()::tid@@Base+0x1fa18>
   63667:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
   6366b:	64 48 2b 04 25 28 00 	sub    %fs:0x28,%rax
   63672:	00 00 
   63674:	0f 85 d1 00 00 00    	jne    6374b <EngineBFM::process_compute_static_ecb(engine_ctxt_t*, unsigned int)@@Base+0x54b>
   6367a:	48 8d 65 d8          	lea    -0x28(%rbp),%rsp
   6367e:	5b                   	pop    %rbx
   6367f:	41 5c                	pop    %r12
   63681:	41 5d                	pop    %r13
   63683:	41 5e                	pop    %r14
   63685:	41 5f                	pop    %r15
   63687:	5d                   	pop    %rbp
   63688:	c3                   	ret    
   63689:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
   63690:	48 8b 9d 68 ff ff ff 	mov    -0x98(%rbp),%rbx
   63697:	4c 8d 25 f2 88 00 00 	lea    0x88f2(%rip),%r12        # 6bf90 <EngineBFM::dma_wait_for_idle()@@Base+0x2e70>
   6369e:	4c 89 ff             	mov    %r15,%rdi
   636a1:	4c 89 e6             	mov    %r12,%rsi
   636a4:	c7 83 18 0b 00 00 01 	movl   $0x1,0xb18(%rbx)
   636ab:	00 00 00 
   636ae:	e8 ed d8 ff ff       	call   60fa0 <void spdlog::logger::log<unsigned int, unsigned int, unsigned int, unsigned int, unsigned int>(spdlog::level::level_enum, char const*, unsigned int const&, unsigned int const&, unsigned int const&, unsigned int const&, unsigned int const&)@@Base+0x450>
   636b3:	48 8b b3 18 02 00 00 	mov    0x218(%rbx),%rsi
   636ba:	48 8b 9d 58 ff ff ff 	mov    -0xa8(%rbp),%rbx
   636c1:	48 89 df             	mov    %rbx,%rdi
   636c4:	e8 d7 d8 ff ff       	call   60fa0 <void spdlog::logger::log<unsigned int, unsigned int, unsigned int, unsigned int, unsigned int>(spdlog::level::level_enum, char const*, unsigned int const&, unsigned int const&, unsigned int const&, unsigned int const&, unsigned int const&)@@Base+0x450>
   636c9:	48 8d bd 70 ff ff ff 	lea    -0x90(%rbp),%rdi
   636d0:	4c 89 fa             	mov    %r15,%rdx
   636d3:	48 89 de             	mov    %rbx,%rsi
   636d6:	e8 95 c3 fb ff       	call   1fa70 <spdlog::details::os::thread_id()::tid@@Base+0x1fa68>
   636db:	48 8b bd 70 ff ff ff 	mov    -0x90(%rbp),%rdi
   636e2:	4c 89 e2             	mov    %r12,%rdx
   636e5:	31 f6                	xor    %esi,%esi
   636e7:	e8 f4 ca fb ff       	call   201e0 <spdlog::details::os::thread_id()::tid@@Base+0x201d8>
   636ec:	e9 31 ff ff ff       	jmp    63622 <EngineBFM::process_compute_static_ecb(engine_ctxt_t*, unsigned int)@@Base+0x422>
   636f1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
   636f8:	48 8b b5 48 ff ff ff 	mov    -0xb8(%rbp),%rsi
   636ff:	48 8b bd 58 ff ff ff 	mov    -0xa8(%rbp),%rdi
   63706:	31 d2                	xor    %edx,%edx
   63708:	e8 e3 c9 fb ff       	call   200f0 <spdlog::details::os::thread_id()::tid@@Base+0x200e8>
   6370d:	48 89 45 80          	mov    %rax,-0x80(%rbp)
   63711:	48 89 c7             	mov    %rax,%rdi
   63714:	48 8b 85 70 ff ff ff 	mov    -0x90(%rbp),%rax
   6371b:	48 89 45 90          	mov    %rax,-0x70(%rbp)
   6371f:	48 8b 95 30 ff ff ff 	mov    -0xd0(%rbp),%rdx
   63726:	48 8b b5 38 ff ff ff 	mov    -0xc8(%rbp),%rsi
   6372d:	e8 de c0 fb ff       	call   1f810 <spdlog::details::os::thread_id()::tid@@Base+0x1f808>
   63732:	48 8b 85 70 ff ff ff 	mov    -0x90(%rbp),%rax
   63739:	48 8b 55 80          	mov    -0x80(%rbp),%rdx
   6373d:	e9 93 fd ff ff       	jmp    634d5 <EngineBFM::process_compute_static_ecb(engine_ctxt_t*, unsigned int)@@Base+0x2d5>
   63742:	48 8b bd 50 ff ff ff 	mov    -0xb0(%rbp),%rdi
   63749:	eb d4                	jmp    6371f <EngineBFM::process_compute_static_ecb(engine_ctxt_t*, unsigned int)@@Base+0x51f>
   6374b:	e8 80 c3 fb ff       	call   1fad0 <spdlog::details::os::thread_id()::tid@@Base+0x1fac8>
   63750:	48 8d 3d a9 68 00 00 	lea    0x68a9(%rip),%rdi        # 6a000 <EngineBFM::dma_wait_for_idle()@@Base+0xee0>
   63757:	e8 e4 bf fb ff       	call   1f740 <spdlog::details::os::thread_id()::tid@@Base+0x1f738>
   6375c:	f3 0f 1e fa          	endbr64 
   63760:	48 89 c3             	mov    %rax,%rbx
   63763:	e9 d1 13 fc ff       	jmp    24b39 <boost::context::detail::context_unwind(boost::context::detail::transfer_t)@@Base+0x410c>
   63768:	f3 0f 1e fa          	endbr64 
   6376c:	48 89 c3             	mov    %rax,%rbx
   6376f:	e9 12 14 fc ff       	jmp    24b86 <boost::context::detail::context_unwind(boost::context::detail::transfer_t)@@Base+0x4159>
   63774:	f3 0f 1e fa          	endbr64 
   63778:	48 89 c3             	mov    %rax,%rbx
   6377b:	e9 ec 13 fc ff       	jmp    24b6c <boost::context::detail::context_unwind(boost::context::detail::transfer_t)@@Base+0x413f>
   63780:	f3 0f 1e fa          	endbr64 
   63784:	48 89 c3             	mov    %rax,%rbx
   63787:	e9 cf 13 fc ff       	jmp    24b5b <boost::context::detail::context_unwind(boost::context::detail::transfer_t)@@Base+0x412e>
   6378c:	f3 0f 1e fa          	endbr64 
   63790:	48 89 c3             	mov    %rax,%rbx
   63793:	e9 76 13 fc ff       	jmp    24b0e <boost::context::detail::context_unwind(boost::context::detail::transfer_t)@@Base+0x40e1>
   63798:	f3 0f 1e fa          	endbr64 
   6379c:	48 89 c3             	mov    %rax,%rbx
   6379f:	e9 7b 13 fc ff       	jmp    24b1f <boost::context::detail::context_unwind(boost::context::detail::transfer_t)@@Base+0x40f2>
   637a4:	f3 0f 1e fa          	endbr64 
   637a8:	48 89 c3             	mov    %rax,%rbx
   637ab:	e9 f8 13 fc ff       	jmp    24ba8 <boost::context::detail::context_unwind(boost::context::detail::transfer_t)@@Base+0x417b>
   637b0:	f3 0f 1e fa          	endbr64 
   637b4:	48 89 c3             	mov    %rax,%rbx
   637b7:	e9 39 14 fc ff       	jmp    24bf5 <boost::context::detail::context_unwind(boost::context::detail::transfer_t)@@Base+0x41c8>
   637bc:	f3 0f 1e fa          	endbr64 
   637c0:	48 89 c3             	mov    %rax,%rbx
   637c3:	e9 0b 14 fc ff       	jmp    24bd3 <boost::context::detail::context_unwind(boost::context::detail::transfer_t)@@Base+0x41a6>
   637c8:	f3 0f 1e fa          	endbr64 
   637cc:	48 89 c3             	mov    %rax,%rbx
   637cf:	e9 4c 14 fc ff       	jmp    24c20 <boost::context::detail::context_unwind(boost::context::detail::transfer_t)@@Base+0x41f3>
   637d4:	f3 0f 1e fa          	endbr64 
   637d8:	48 89 c3             	mov    %rax,%rbx
   637db:	e9 26 14 fc ff       	jmp    24c06 <boost::context::detail::context_unwind(boost::context::detail::transfer_t)@@Base+0x41d9>
   637e0:	f3 0f 1e fa          	endbr64 
   637e4:	48 89 c3             	mov    %rax,%rbx
   637e7:	e9 cd 13 fc ff       	jmp    24bb9 <boost::context::detail::context_unwind(boost::context::detail::transfer_t)@@Base+0x418c>
