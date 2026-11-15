//! Emulator that boots the Horizon OS kernel.
//!
//! The kernel image runs unmodified at EL1, and its initial processes at
//! EL0; this crate supplies what lies outside them.
//!
//! * [`boot`] and [`secmon`] define the contract the kernel expects at entry
//!   and from the secure monitor.
//! * [`image`] decodes the kernel blob.
//! * [`gic`], [`timer`] and [`uart`] are the devices the kernel itself
//!   drives, reached through [`bus`].
//! * [`hvf`] runs guest code on the host CPU, and [`hvf::debug`] adds
//!   breakpoints and single-step that the guest cannot see.
//! * [`soc`] holds the rest of the system-on-chip and the board devices
//!   the initial processes drive; [`keys`] loads the console keys the
//!   secure monitor derives from; [`nand`] makes the eMMC image of a
//!   console that has the firmware installed, and [`sdcard`] the image of
//!   a microSD card with files on it.
//! * [`machine`] ties these into a console and executes it.
//! * [`symbols`] and [`svc`] name things for traces, and [`ipc`] decodes the
//!   messages processes send each other.

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
compile_error!("the only CPU backend is Hypervisor.framework on Apple silicon");

pub mod arm64;
pub mod boot;
pub mod bus;
pub mod gic;
pub mod hvf;
pub mod image;
pub mod ipc;
pub mod keys;
pub mod machine;
pub mod maxwell;
pub mod memory;
pub mod nand;
pub mod sdcard;
pub mod secmon;
pub mod soc;
pub mod svc;
pub mod symbols;
pub mod timer;
pub mod uart;
