//! Emulator that boots the Horizon OS kernel.
//!
//! The kernel image runs unmodified at EL1, and its initial processes at
//! EL0; this crate supplies what lies outside them.
//!
//! * [`boot`] and [`secmon`] define the contract the kernel expects at entry
//!   and from the secure monitor.
//! * [`image`] decodes the kernel blob, and [`package2`] verifies and
//!   decrypts it out of package2 as the secure monitor does on cold boot.
//! * [`gic`], [`timer`] and [`uart`] are the devices the kernel itself
//!   drives, reached through [`bus`].
//! * [`hvf`] runs guest code on the host CPU, and [`hvf::debug`] adds
//!   breakpoints and single-step that the guest cannot see.
//! * [`bpmp`] interprets the boot and power-management processor, the
//!   ARM7TDMI the boot chain runs on before the CCPLEX starts.
//! * [`soc`] holds the rest of the system-on-chip and the board devices
//!   the initial processes drive; [`keys`] loads the console keys the
//!   secure monitor derives from; [`nand`] makes the eMMC image of a
//!   console that has the firmware installed, [`cdn`] fetches that
//!   firmware from Nintendo as the console it is for, and [`sdcard`] makes
//!   the image of a microSD card with files on it; [`homebrew`] runs
//!   programs Nintendo did not sign on it.
//! * [`machine`] ties these into a console and executes it.
//! * [`symbols`] and [`svc`] name things for traces, [`ipc`] decodes the
//!   messages processes send each other, and [`log_manager`] the log
//!   packets they send the log manager.

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
compile_error!("the only CPU backend is Hypervisor.framework on Apple silicon");

pub mod air;
pub mod arm64;
pub mod boot;
pub mod bootrom;
pub mod bpmp;
pub mod bus;
pub mod cdn;
pub mod clock;
pub mod falcon;
pub mod gic;
pub mod homebrew;
pub mod hvf;
pub mod image;
pub mod ipc;
pub mod keys;
pub mod log_manager;
pub mod machine;
pub mod maxwell;
pub mod memory;
pub mod nand;
pub mod net;
pub mod package2;
pub mod pmu;
pub mod sdcard;
pub mod secmon;
pub mod snapshot;
pub mod soc;
pub mod svc;
pub mod symbols;
pub mod timer;
pub mod uart;
