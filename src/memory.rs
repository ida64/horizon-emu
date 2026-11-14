//! Guest DRAM backed by anonymous host memory.

use std::{io, ptr};

/// One contiguous block of guest-physical memory.
///
/// The guest's vCPUs write this memory behind the compiler's back while host
/// threads inspect it, so it is only ever accessed through raw-pointer copies;
/// no reference into it is handed out. Host-side reads may observe a guest
/// write in progress, which is inherent to looking at a running machine.
pub struct GuestRam {
    base: u64,
    len: usize,
    host: *mut u8,
}

// SAFETY: `host` is an exclusively owned mapping that lives until drop, and
// all access goes through `ptr::copy` on raw pointers, never references.
unsafe impl Send for GuestRam {}
// SAFETY: as above; concurrent access is unsynchronized by design, exactly as
// between the guest's own cores.
unsafe impl Sync for GuestRam {}

impl GuestRam {
    /// Reserves `len` zeroed bytes at guest-physical `base`. Pages are
    /// committed lazily, so a multi-gigabyte block costs what the guest touches.
    pub fn new(base: u64, len: usize) -> io::Result<Self> {
        // SAFETY: anonymous private mapping; no existing memory is affected.
        let host = unsafe {
            libc::mmap(
                ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        if host == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(Self {
            base,
            len,
            host: host.cast(),
        })
    }

    pub fn base(&self) -> u64 {
        self.base
    }

    pub fn size(&self) -> usize {
        self.len
    }

    /// Host address of the block, for mapping into a VM.
    pub fn host_ptr(&self) -> *mut u8 {
        self.host
    }

    /// Host pointer to `len` bytes at guest-physical `address`, if in range.
    fn locate(&self, address: u64, len: usize) -> Option<*mut u8> {
        let offset = usize::try_from(address.checked_sub(self.base)?).ok()?;
        (offset.checked_add(len)? <= self.len).then(|| self.host.wrapping_add(offset))
    }

    /// Copies guest memory into `buffer`; `None` if the range is not all RAM.
    pub fn read(&self, address: u64, buffer: &mut [u8]) -> Option<()> {
        let source = self.locate(address, buffer.len())?;
        // SAFETY: `locate` bounds-checked the range against the mapping, and
        // `buffer` is a distinct host allocation.
        unsafe { ptr::copy_nonoverlapping(source, buffer.as_mut_ptr(), buffer.len()) };
        Some(())
    }

    /// Copies `data` into guest memory; `None` if the range is not all RAM.
    pub fn write(&self, address: u64, data: &[u8]) -> Option<()> {
        let destination = self.locate(address, data.len())?;
        // SAFETY: as in `read`.
        unsafe { ptr::copy_nonoverlapping(data.as_ptr(), destination, data.len()) };
        Some(())
    }

    pub fn read_u64(&self, address: u64) -> Option<u64> {
        let mut raw = [0; 8];
        self.read(address, &mut raw)?;
        Some(u64::from_le_bytes(raw))
    }
}

impl Drop for GuestRam {
    fn drop(&mut self) {
        // SAFETY: `host`/`len` are exactly what mmap returned.
        unsafe { libc::munmap(self.host.cast(), self.len) };
    }
}
