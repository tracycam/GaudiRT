
GaudiRT/evidence/runtime-audit-20261023/binaries/libeng_mme_cmpt_bfm.so:     file format elf64-x86-64


Disassembly of section .text:

0000000000057a20 <EngineBFM::eng_process_compute_ecb_list(unsigned int)@@Base>:
   57a20:	f3 0f 1e fa          	endbr64 
   57a24:	c3                   	ret    
   57a25:	90                   	nop
   57a26:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
   57a2d:	00 00 00 

0000000000057a30 <EngineBFM::process_compute_eng_barrier(unsigned int, unsigned int)@@Base>:
   57a30:	f3 0f 1e fa          	endbr64 
   57a34:	41 57                	push   %r15
   57a36:	41 56                	push   %r14
   57a38:	41 55                	push   %r13
   57a3a:	41 54                	push   %r12
   57a3c:	55                   	push   %rbp
   57a3d:	48 89 fd             	mov    %rdi,%rbp
   57a40:	53                   	push   %rbx
   57a41:	48 bb 73 6f 62 5f 69 	movabs $0x3d2064695f626f73,%rbx
   57a48:	64 20 3d 
   57a4b:	48 81 ec 98 00 00 00 	sub    $0x98,%rsp
   57a52:	89 74 24 2c          	mov    %esi,0x2c(%rsp)
   57a56:	4c 8d 6c 24 60       	lea    0x60(%rsp),%r13
   57a5b:	4c 8d 64 24 30       	lea    0x30(%rsp),%r12
   57a60:	89 54 24 28          	mov    %edx,0x28(%rsp)
   57a64:	4c 89 e6             	mov    %r12,%rsi
   57a67:	31 d2                	xor    %edx,%edx
   57a69:	4c 89 ef             	mov    %r13,%rdi
   57a6c:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
   57a73:	00 00 
   57a75:	48 89 84 24 88 00 00 	mov    %rax,0x88(%rsp)
   57a7c:	00 
   57a7d:	31 c0                	xor    %eax,%eax
   57a7f:	48 8d 44 24 70       	lea    0x70(%rsp),%rax
   57a84:	48 c7 44 24 30 1b 00 	movq   $0x1b,0x30(%rsp)
   57a8b:	00 00 
   57a8d:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
   57a92:	48 89 44 24 60       	mov    %rax,0x60(%rsp)
   57a97:	e8 54 86 fc ff       	call   200f0 <spdlog::details::os::thread_id()::tid@@Base+0x200e8>
   57a9c:	48 8b 54 24 30       	mov    0x30(%rsp),%rdx
   57aa1:	66 0f 6f 05 77 71 01 	movdqa 0x17177(%rip),%xmm0        # 6ec20 <std::_Sp_make_shared_tag::_S_ti()::__tag@@Base+0x370>
   57aa8:	00 
   57aa9:	48 89 44 24 60       	mov    %rax,0x60(%rsp)
   57aae:	48 89 54 24 70       	mov    %rdx,0x70(%rsp)
   57ab3:	ba 20 7b 00 00       	mov    $0x7b20,%edx
   57ab8:	48 89 58 10          	mov    %rbx,0x10(%rax)
   57abc:	48 8d 5c 24 50       	lea    0x50(%rsp),%rbx
   57ac1:	66 89 50 18          	mov    %dx,0x18(%rax)
   57ac5:	c6 40 1a 7d          	movb   $0x7d,0x1a(%rax)
   57ac9:	0f 11 00             	movups %xmm0,(%rax)
   57acc:	48 8b 44 24 30       	mov    0x30(%rsp),%rax
   57ad1:	48 8b 54 24 60       	mov    0x60(%rsp),%rdx
   57ad6:	48 89 44 24 68       	mov    %rax,0x68(%rsp)
   57adb:	c6 04 02 00          	movb   $0x0,(%rdx,%rax,1)
   57adf:	4c 8b bd 18 02 00 00 	mov    0x218(%rbp),%r15
   57ae6:	48 89 5c 24 40       	mov    %rbx,0x40(%rsp)
   57aeb:	4d 85 ff             	test   %r15,%r15
   57aee:	0f 84 77 03 00 00    	je     57e6b <EngineBFM::process_compute_eng_barrier(unsigned int, unsigned int)@@Base+0x43b>
   57af4:	4c 89 ff             	mov    %r15,%rdi
   57af7:	4c 8d 74 24 40       	lea    0x40(%rsp),%r14
   57afc:	e8 6f 79 fc ff       	call   1f470 <spdlog::details::os::thread_id()::tid@@Base+0x1f468>
   57b01:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
   57b06:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
   57b0b:	48 83 f8 0f          	cmp    $0xf,%rax
   57b0f:	0f 87 bb 02 00 00    	ja     57dd0 <EngineBFM::process_compute_eng_barrier(unsigned int, unsigned int)@@Base+0x3a0>
   57b15:	48 83 f8 01          	cmp    $0x1,%rax
   57b19:	0f 85 81 02 00 00    	jne    57da0 <EngineBFM::process_compute_eng_barrier(unsigned int, unsigned int)@@Base+0x370>
   57b1f:	41 0f b6 17          	movzbl (%r15),%edx
   57b23:	88 54 24 50          	mov    %dl,0x50(%rsp)
   57b27:	48 89 da             	mov    %rbx,%rdx
   57b2a:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
   57b2f:	4c 89 f6             	mov    %r14,%rsi
   57b32:	4c 89 e7             	mov    %r12,%rdi
   57b35:	c6 04 02 00          	movb   $0x0,(%rdx,%rax,1)
   57b39:	4c 89 ea             	mov    %r13,%rdx
   57b3c:	e8 2f 7f fc ff       	call   1fa70 <spdlog::details::os::thread_id()::tid@@Base+0x1fa68>
   57b41:	48 8d 44 24 28       	lea    0x28(%rsp),%rax
   57b46:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
   57b4b:	be 02 00 00 00       	mov    $0x2,%esi
   57b50:	48 8d 15 c8 53 01 00 	lea    0x153c8(%rip),%rdx        # 6cf1f <EngineBFM::dma_wait_for_idle()@@Base+0x3dff>
   57b57:	48 89 c1             	mov    %rax,%rcx
   57b5a:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
   57b5f:	e8 9c 81 fc ff       	call   1fd00 <spdlog::details::os::thread_id()::tid@@Base+0x1fcf8>
   57b64:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
   57b69:	48 85 ff             	test   %rdi,%rdi
   57b6c:	74 05                	je     57b73 <EngineBFM::process_compute_eng_barrier(unsigned int, unsigned int)@@Base+0x143>
   57b6e:	e8 8d 7c fc ff       	call   1f800 <spdlog::details::os::thread_id()::tid@@Base+0x1f7f8>
   57b73:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
   57b78:	48 39 df             	cmp    %rbx,%rdi
   57b7b:	74 0e                	je     57b8b <EngineBFM::process_compute_eng_barrier(unsigned int, unsigned int)@@Base+0x15b>
   57b7d:	48 8b 44 24 50       	mov    0x50(%rsp),%rax
   57b82:	48 8d 70 01          	lea    0x1(%rax),%rsi
   57b86:	e8 95 7e fc ff       	call   1fa20 <spdlog::details::os::thread_id()::tid@@Base+0x1fa18>
   57b8b:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
   57b90:	48 3b 7c 24 08       	cmp    0x8(%rsp),%rdi
   57b95:	74 0e                	je     57ba5 <EngineBFM::process_compute_eng_barrier(unsigned int, unsigned int)@@Base+0x175>
   57b97:	48 8b 44 24 70       	mov    0x70(%rsp),%rax
   57b9c:	48 8d 70 01          	lea    0x1(%rax),%rsi
   57ba0:	e8 7b 7e fc ff       	call   1fa20 <spdlog::details::os::thread_id()::tid@@Base+0x1fa18>
   57ba5:	8b 85 20 03 00 00    	mov    0x320(%rbp),%eax
   57bab:	48 8b 95 f8 0c 00 00 	mov    0xcf8(%rbp),%rdx
   57bb2:	be 01 00 00 00       	mov    $0x1,%esi
   57bb7:	48 89 ef             	mov    %rbp,%rdi
   57bba:	48 8d 0c c0          	lea    (%rax,%rax,8),%rcx
   57bbe:	48 89 54 24 20       	mov    %rdx,0x20(%rsp)
   57bc3:	4c 8d 3c cd 00 00 00 	lea    0x0(,%rcx,8),%r15
   57bca:	00 
   57bcb:	4a 8d 84 3a 08 05 00 	lea    0x508(%rdx,%r15,1),%rax
   57bd2:	00 
   57bd3:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
   57bd8:	e8 03 7f fc ff       	call   1fae0 <spdlog::details::os::thread_id()::tid@@Base+0x1fad8>
   57bdd:	8b 4c 24 2c          	mov    0x2c(%rsp),%ecx
   57be1:	8b 44 24 28          	mov    0x28(%rsp),%eax
   57be5:	4c 89 ef             	mov    %r13,%rdi
   57be8:	48 8b 54 24 20       	mov    0x20(%rsp),%rdx
   57bed:	48 8d 0c 49          	lea    (%rcx,%rcx,2),%rcx
   57bf1:	c1 e0 02             	shl    $0x2,%eax
   57bf4:	48 8d 4c cd 00       	lea    0x0(%rbp,%rcx,8),%rcx
   57bf9:	4c 01 fa             	add    %r15,%rdx
   57bfc:	8b b1 5c 05 00 00    	mov    0x55c(%rcx),%esi
   57c02:	01 c6                	add    %eax,%esi
   57c04:	89 b2 18 05 00 00    	mov    %esi,0x518(%rdx)
   57c0a:	03 81 5c 05 00 00    	add    0x55c(%rcx),%eax
   57c10:	4c 89 e6             	mov    %r12,%rsi
   57c13:	89 82 28 05 00 00    	mov    %eax,0x528(%rdx)
   57c19:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
   57c1e:	31 d2                	xor    %edx,%edx
   57c20:	48 c7 44 24 30 29 00 	movq   $0x29,0x30(%rsp)
   57c27:	00 00 
   57c29:	48 89 44 24 60       	mov    %rax,0x60(%rsp)
   57c2e:	e8 bd 84 fc ff       	call   200f0 <spdlog::details::os::thread_id()::tid@@Base+0x200e8>
   57c33:	48 8b 54 24 30       	mov    0x30(%rsp),%rdx
   57c38:	66 0f 6f 05 f0 6f 01 	movdqa 0x16ff0(%rip),%xmm0        # 6ec30 <std::_Sp_make_shared_tag::_S_ti()::__tag@@Base+0x380>
   57c3f:	00 
   57c40:	48 b9 6f 62 5f 69 64 	movabs $0x7b203a64695f626f,%rcx
   57c47:	3a 20 7b 
   57c4a:	48 89 44 24 60       	mov    %rax,0x60(%rsp)
   57c4f:	48 89 54 24 70       	mov    %rdx,0x70(%rsp)
   57c54:	0f 11 00             	movups %xmm0,(%rax)
   57c57:	66 0f 6f 05 e1 6f 01 	movdqa 0x16fe1(%rip),%xmm0        # 6ec40 <std::_Sp_make_shared_tag::_S_ti()::__tag@@Base+0x390>
   57c5e:	00 
   57c5f:	48 89 48 20          	mov    %rcx,0x20(%rax)
   57c63:	c6 40 28 7d          	movb   $0x7d,0x28(%rax)
   57c67:	0f 11 40 10          	movups %xmm0,0x10(%rax)
   57c6b:	48 8b 44 24 30       	mov    0x30(%rsp),%rax
   57c70:	48 8b 54 24 60       	mov    0x60(%rsp),%rdx
   57c75:	48 89 44 24 68       	mov    %rax,0x68(%rsp)
   57c7a:	c6 04 02 00          	movb   $0x0,(%rdx,%rax,1)
   57c7e:	4c 8b bd 18 02 00 00 	mov    0x218(%rbp),%r15
   57c85:	48 89 5c 24 40       	mov    %rbx,0x40(%rsp)
   57c8a:	4d 85 ff             	test   %r15,%r15
   57c8d:	0f 84 cc 01 00 00    	je     57e5f <EngineBFM::process_compute_eng_barrier(unsigned int, unsigned int)@@Base+0x42f>
   57c93:	4c 89 ff             	mov    %r15,%rdi
   57c96:	e8 d5 77 fc ff       	call   1f470 <spdlog::details::os::thread_id()::tid@@Base+0x1f468>
   57c9b:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
   57ca0:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
   57ca5:	48 83 f8 0f          	cmp    $0xf,%rax
   57ca9:	0f 87 61 01 00 00    	ja     57e10 <EngineBFM::process_compute_eng_barrier(unsigned int, unsigned int)@@Base+0x3e0>
   57caf:	48 83 f8 01          	cmp    $0x1,%rax
   57cb3:	0f 85 ff 00 00 00    	jne    57db8 <EngineBFM::process_compute_eng_barrier(unsigned int, unsigned int)@@Base+0x388>
   57cb9:	41 0f b6 17          	movzbl (%r15),%edx
   57cbd:	88 54 24 50          	mov    %dl,0x50(%rsp)
   57cc1:	48 89 da             	mov    %rbx,%rdx
   57cc4:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
   57cc9:	4c 89 f6             	mov    %r14,%rsi
   57ccc:	4c 89 e7             	mov    %r12,%rdi
   57ccf:	c6 04 02 00          	movb   $0x0,(%rdx,%rax,1)
   57cd3:	4c 89 ea             	mov    %r13,%rdx
   57cd6:	e8 95 7d fc ff       	call   1fa70 <spdlog::details::os::thread_id()::tid@@Base+0x1fa68>
   57cdb:	4c 8b 44 24 18       	mov    0x18(%rsp),%r8
   57ce0:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
   57ce5:	48 8d 4c 24 2c       	lea    0x2c(%rsp),%rcx
   57cea:	48 8d 15 0f 31 01 00 	lea    0x1310f(%rip),%rdx        # 6ae00 <EngineBFM::dma_wait_for_idle()@@Base+0x1ce0>
   57cf1:	be 02 00 00 00       	mov    $0x2,%esi
   57cf6:	e8 95 7e fc ff       	call   1fb90 <spdlog::details::os::thread_id()::tid@@Base+0x1fb88>
   57cfb:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
   57d00:	48 85 ff             	test   %rdi,%rdi
   57d03:	74 05                	je     57d0a <EngineBFM::process_compute_eng_barrier(unsigned int, unsigned int)@@Base+0x2da>
   57d05:	e8 f6 7a fc ff       	call   1f800 <spdlog::details::os::thread_id()::tid@@Base+0x1f7f8>
   57d0a:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
   57d0f:	48 39 df             	cmp    %rbx,%rdi
   57d12:	74 0e                	je     57d22 <EngineBFM::process_compute_eng_barrier(unsigned int, unsigned int)@@Base+0x2f2>
   57d14:	48 8b 44 24 50       	mov    0x50(%rsp),%rax
   57d19:	48 8d 70 01          	lea    0x1(%rax),%rsi
   57d1d:	e8 fe 7c fc ff       	call   1fa20 <spdlog::details::os::thread_id()::tid@@Base+0x1fa18>
   57d22:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
   57d27:	48 3b 7c 24 08       	cmp    0x8(%rsp),%rdi
   57d2c:	74 0e                	je     57d3c <EngineBFM::process_compute_eng_barrier(unsigned int, unsigned int)@@Base+0x30c>
   57d2e:	48 8b 44 24 70       	mov    0x70(%rsp),%rax
   57d33:	48 8d 70 01          	lea    0x1(%rax),%rsi
   57d37:	e8 e4 7c fc ff       	call   1fa20 <spdlog::details::os::thread_id()::tid@@Base+0x1fa18>
   57d3c:	48 8b 45 00          	mov    0x0(%rbp),%rax
   57d40:	8b 9d e8 02 00 00    	mov    0x2e8(%rbp),%ebx
   57d46:	48 8b 40 20          	mov    0x20(%rax),%rax
   57d4a:	48 3b 05 2f de 02 00 	cmp    0x2de2f(%rip),%rax        # 85b80 <EngineBFM::get_dccm_base()@@Base+0x5e2d0>
   57d51:	0f 85 f9 00 00 00    	jne    57e50 <EngineBFM::process_compute_eng_barrier(unsigned int, unsigned int)@@Base+0x420>
   57d57:	48 8b 85 f8 0c 00 00 	mov    0xcf8(%rbp),%rax
   57d5e:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
   57d63:	48 89 ef             	mov    %rbp,%rdi
   57d66:	48 29 c2             	sub    %rax,%rdx
   57d69:	8d 34 13             	lea    (%rbx,%rdx,1),%esi
   57d6c:	ba 48 00 00 00       	mov    $0x48,%edx
   57d71:	e8 ea 7f fc ff       	call   1fd60 <spdlog::details::os::thread_id()::tid@@Base+0x1fd58>
   57d76:	48 8b 84 24 88 00 00 	mov    0x88(%rsp),%rax
   57d7d:	00 
   57d7e:	64 48 2b 04 25 28 00 	sub    %fs:0x28,%rax
   57d85:	00 00 
   57d87:	0f 85 cd 00 00 00    	jne    57e5a <EngineBFM::process_compute_eng_barrier(unsigned int, unsigned int)@@Base+0x42a>
   57d8d:	48 81 c4 98 00 00 00 	add    $0x98,%rsp
   57d94:	5b                   	pop    %rbx
   57d95:	5d                   	pop    %rbp
   57d96:	41 5c                	pop    %r12
   57d98:	41 5d                	pop    %r13
   57d9a:	41 5e                	pop    %r14
   57d9c:	41 5f                	pop    %r15
   57d9e:	c3                   	ret    
   57d9f:	90                   	nop
   57da0:	48 83 7c 24 10 00    	cmpq   $0x0,0x10(%rsp)
   57da6:	0f 85 d0 00 00 00    	jne    57e7c <EngineBFM::process_compute_eng_barrier(unsigned int, unsigned int)@@Base+0x44c>
   57dac:	48 89 da             	mov    %rbx,%rdx
   57daf:	e9 76 fd ff ff       	jmp    57b2a <EngineBFM::process_compute_eng_barrier(unsigned int, unsigned int)@@Base+0xfa>
   57db4:	0f 1f 40 00          	nopl   0x0(%rax)
   57db8:	48 83 7c 24 20 00    	cmpq   $0x0,0x20(%rsp)
   57dbe:	0f 85 b3 00 00 00    	jne    57e77 <EngineBFM::process_compute_eng_barrier(unsigned int, unsigned int)@@Base+0x447>
   57dc4:	48 89 da             	mov    %rbx,%rdx
   57dc7:	e9 f8 fe ff ff       	jmp    57cc4 <EngineBFM::process_compute_eng_barrier(unsigned int, unsigned int)@@Base+0x294>
   57dcc:	0f 1f 40 00          	nopl   0x0(%rax)
   57dd0:	31 d2                	xor    %edx,%edx
   57dd2:	4c 89 e6             	mov    %r12,%rsi
   57dd5:	4c 89 f7             	mov    %r14,%rdi
   57dd8:	e8 13 83 fc ff       	call   200f0 <spdlog::details::os::thread_id()::tid@@Base+0x200e8>
   57ddd:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
   57de2:	48 89 c7             	mov    %rax,%rdi
   57de5:	48 8b 44 24 30       	mov    0x30(%rsp),%rax
   57dea:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
   57def:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
   57df4:	4c 89 fe             	mov    %r15,%rsi
   57df7:	e8 14 7a fc ff       	call   1f810 <spdlog::details::os::thread_id()::tid@@Base+0x1f808>
   57dfc:	48 8b 44 24 30       	mov    0x30(%rsp),%rax
   57e01:	48 8b 54 24 40       	mov    0x40(%rsp),%rdx
   57e06:	e9 1f fd ff ff       	jmp    57b2a <EngineBFM::process_compute_eng_barrier(unsigned int, unsigned int)@@Base+0xfa>
   57e0b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
   57e10:	31 d2                	xor    %edx,%edx
   57e12:	4c 89 e6             	mov    %r12,%rsi
   57e15:	4c 89 f7             	mov    %r14,%rdi
   57e18:	e8 d3 82 fc ff       	call   200f0 <spdlog::details::os::thread_id()::tid@@Base+0x200e8>
   57e1d:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
   57e22:	48 89 c7             	mov    %rax,%rdi
   57e25:	48 8b 44 24 30       	mov    0x30(%rsp),%rax
   57e2a:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
   57e2f:	48 8b 54 24 20       	mov    0x20(%rsp),%rdx
   57e34:	4c 89 fe             	mov    %r15,%rsi
   57e37:	e8 d4 79 fc ff       	call   1f810 <spdlog::details::os::thread_id()::tid@@Base+0x1f808>
   57e3c:	48 8b 44 24 30       	mov    0x30(%rsp),%rax
   57e41:	48 8b 54 24 40       	mov    0x40(%rsp),%rdx
   57e46:	e9 79 fe ff ff       	jmp    57cc4 <EngineBFM::process_compute_eng_barrier(unsigned int, unsigned int)@@Base+0x294>
   57e4b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
   57e50:	48 89 ef             	mov    %rbp,%rdi
   57e53:	ff d0                	call   *%rax
   57e55:	e9 04 ff ff ff       	jmp    57d5e <EngineBFM::process_compute_eng_barrier(unsigned int, unsigned int)@@Base+0x32e>
   57e5a:	e8 71 7c fc ff       	call   1fad0 <spdlog::details::os::thread_id()::tid@@Base+0x1fac8>
   57e5f:	48 8d 3d 9a 21 01 00 	lea    0x1219a(%rip),%rdi        # 6a000 <EngineBFM::dma_wait_for_idle()@@Base+0xee0>
   57e66:	e8 d5 78 fc ff       	call   1f740 <spdlog::details::os::thread_id()::tid@@Base+0x1f738>
   57e6b:	48 8d 3d 8e 21 01 00 	lea    0x1218e(%rip),%rdi        # 6a000 <EngineBFM::dma_wait_for_idle()@@Base+0xee0>
   57e72:	e8 c9 78 fc ff       	call   1f740 <spdlog::details::os::thread_id()::tid@@Base+0x1f738>
   57e77:	48 89 df             	mov    %rbx,%rdi
   57e7a:	eb b3                	jmp    57e2f <EngineBFM::process_compute_eng_barrier(unsigned int, unsigned int)@@Base+0x3ff>
   57e7c:	48 89 df             	mov    %rbx,%rdi
   57e7f:	e9 6b ff ff ff       	jmp    57def <EngineBFM::process_compute_eng_barrier(unsigned int, unsigned int)@@Base+0x3bf>
   57e84:	f3 0f 1e fa          	endbr64 
   57e88:	48 89 c5             	mov    %rax,%rbp
   57e8b:	e9 74 ab fc ff       	jmp    22a04 <boost::context::detail::context_unwind(boost::context::detail::transfer_t)@@Base+0x1fd7>
   57e90:	f3 0f 1e fa          	endbr64 
   57e94:	48 89 c5             	mov    %rax,%rbp
   57e97:	e9 c0 ab fc ff       	jmp    22a5c <boost::context::detail::context_unwind(boost::context::detail::transfer_t)@@Base+0x202f>
   57e9c:	f3 0f 1e fa          	endbr64 
   57ea0:	48 89 c5             	mov    %rax,%rbp
   57ea3:	e9 cc ab fc ff       	jmp    22a74 <boost::context::detail::context_unwind(boost::context::detail::transfer_t)@@Base+0x2047>
   57ea8:	f3 0f 1e fa          	endbr64 
   57eac:	48 89 c5             	mov    %rax,%rbp
   57eaf:	e9 5f ab fc ff       	jmp    22a13 <boost::context::detail::context_unwind(boost::context::detail::transfer_t)@@Base+0x1fe6>
   57eb4:	f3 0f 1e fa          	endbr64 
   57eb8:	48 89 c5             	mov    %rax,%rbp
   57ebb:	e9 6b ab fc ff       	jmp    22a2b <boost::context::detail::context_unwind(boost::context::detail::transfer_t)@@Base+0x1ffe>
   57ec0:	f3 0f 1e fa          	endbr64 
   57ec4:	48 89 c5             	mov    %rax,%rbp
   57ec7:	e9 81 ab fc ff       	jmp    22a4d <boost::context::detail::context_unwind(boost::context::detail::transfer_t)@@Base+0x2020>
   57ecc:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000057ed0 <EngineBFM::process_eng_wd_fence_and_exec(engine_ctxt_t*)@@Base>:
   57ed0:	f3 0f 1e fa          	endbr64 
   57ed4:	41 57                	push   %r15
   57ed6:	41 56                	push   %r14
   57ed8:	41 55                	push   %r13
   57eda:	41 54                	push   %r12
   57edc:	55                   	push   %rbp
   57edd:	53                   	push   %rbx
   57ede:	48 89 fb             	mov    %rdi,%rbx
   57ee1:	48 81 ec 98 08 00 00 	sub    $0x898,%rsp
   57ee8:	48 8b 8f f8 0c 00 00 	mov    0xcf8(%rdi),%rcx
   57eef:	8b af 68 0a 00 00    	mov    0xa68(%rdi),%ebp
   57ef5:	48 89 74 24 38       	mov    %rsi,0x38(%rsp)
   57efa:	48                   	rex.W
   57efb:	03                   	.byte 0x3
   57efc:	af                   	scas   %es:(%rdi),%eax
   57efd:	58                   	pop    %rax
   57efe:	0a 00                	or     (%rax),%al
