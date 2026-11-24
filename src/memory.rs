//! Guest DRAM backed by anonymous host memory.
//!
//! # Write watch
//!
//! Once mapped into a VM ([`GuestRam::watch_writes`]), host caches of guest
//! memory can learn whether it was written since they last read it without
//! reading it again: [`GuestRam::watch`] takes away the guest's write
//! permission on the pages of a range (the VM's stage-2 tables) and gives a
//! *mark*; a guest write to such a page faults to the host
//! ([`GuestRam::write_fault`]), which notes the page written and gives the
//! permission back, and a host write ([`GuestRam::write`]) notes its pages
//! written too. [`GuestRam::unwritten_since`] then says whether no page of
//! a range was written since a mark. Pages are the host's (the granule
//! `hv_vm_protect` takes), so a write next to a watched range in the same
//! page counts as a write to it.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
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
    watch: OnceLock<Watch>,
}

/// Sets the guest's stage-2 permissions of `size` bytes at guest-physical
/// `ipa`: read and execute, and write if `writable`; whether it could.
pub type Protect = fn(ipa: u64, size: usize, writable: bool) -> bool;

/// The write watch's state (see the module documentation).
struct Watch {
    /// Bytes per page.
    page: u64,
    protect: Protect,
    /// Held while pages' permissions change and while a mark is given, so
    /// a page is watched from before its mark on until a fault notes it.
    lock: Mutex<()>,
    /// One bit per page: whether the guest may not write it.
    protected: Box<[AtomicU64]>,
    /// Per page, the mark current when it was last written: at least every
    /// mark given before the write.
    written: Box<[AtomicU64]>,
    /// The last mark given.
    mark: AtomicU64,
}

impl Watch {
    fn note_written(&self, page: usize) {
        self.written[page].fetch_max(self.mark.load(Ordering::SeqCst), Ordering::SeqCst);
    }

    fn is_protected(&self, page: usize) -> bool {
        self.protected[page / 64].load(Ordering::SeqCst) & (1 << (page % 64)) != 0
    }
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
            watch: OnceLock::new(),
        })
    }

    /// Starts the write watch (see the module documentation), for a block
    /// just mapped writable into a VM whose stage-2 permissions `protect`
    /// sets, in pages of `page` bytes. Only the first call counts.
    pub fn watch_writes(&self, page: u64, protect: Protect) {
        let pages = self.len.div_ceil(page as usize);
        let _ = self.watch.set(Watch {
            page,
            protect,
            lock: Mutex::new(()),
            protected: (0..pages.div_ceil(64)).map(|_| AtomicU64::new(0)).collect(),
            written: (0..pages).map(|_| AtomicU64::new(0)).collect(),
            mark: AtomicU64::new(1),
        });
    }

    /// The pages `len` bytes at `address` lie in, if all RAM: first and
    /// one past the last (none for no bytes).
    fn pages(&self, watch: &Watch, address: u64, len: usize) -> Option<(usize, usize)> {
        self.locate(address, len)?;
        let offset = address - self.base;
        let first = offset / watch.page;
        let end = (offset + len as u64).div_ceil(watch.page);
        Some((first as usize, end.max(first) as usize))
    }

    /// Watches the pages of `len` bytes at `address` for writes and gives
    /// a mark for [`unwritten_since`](Self::unwritten_since): a write to
    /// them after this call returns counts as one since the mark. `None`
    /// without a watch, outside RAM, or where the permission could not be
    /// taken.
    pub fn watch(&self, address: u64, len: usize) -> Option<u64> {
        let watch = self.watch.get()?;
        let (first, end) = self.pages(watch, address, len)?;
        let _held = watch.lock.lock().unwrap();
        let mut page = first;
        while page < end {
            if watch.is_protected(page) {
                page += 1;
                continue;
            }
            let run = (page..end)
                .find(|&p| watch.is_protected(p))
                .unwrap_or(end);
            let ipa = self.base + page as u64 * watch.page;
            let size = ((run - page) as u64 * watch.page).min(self.base + self.len as u64 - ipa);
            if !(watch.protect)(ipa, size as usize, false) {
                return None;
            }
            for p in page..run {
                watch.protected[p / 64].fetch_or(1 << (p % 64), Ordering::SeqCst);
            }
            page = run;
        }
        Some(watch.mark.fetch_add(1, Ordering::SeqCst) + 1)
    }

    /// Whether no page of `len` bytes at `address` was written since
    /// `mark` (given by [`watch`](Self::watch) for a range covering
    /// them); `false` without a watch or outside RAM.
    pub fn unwritten_since(&self, address: u64, len: usize, mark: u64) -> bool {
        let Some(watch) = self.watch.get() else {
            return false;
        };
        self.pages(watch, address, len).is_some_and(|(first, end)| {
            (first..end).all(|page| watch.written[page].load(Ordering::SeqCst) < mark)
        })
    }

    /// A guest write at guest-physical `ipa` faulted: if a watched page,
    /// it is noted written and given back the write permission, and the
    /// write is to be retried (`true`). `false` for an address the watch
    /// does not cover.
    pub fn write_fault(&self, ipa: u64) -> bool {
        let Some(watch) = self.watch.get() else {
            return false;
        };
        let Some((page, _)) = self.pages(watch, ipa, 1) else {
            return false;
        };
        let _held = watch.lock.lock().unwrap();
        if watch.is_protected(page) {
            let at = self.base + page as u64 * watch.page;
            let size = watch.page.min(self.base + self.len as u64 - at);
            if !(watch.protect)(at, size as usize, true) {
                return false;
            }
            watch.protected[page / 64].fetch_and(!(1 << (page % 64)), Ordering::SeqCst);
        }
        // A fault on a page another core's fault gave back is retried too.
        watch.note_written(page);
        true
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

    /// Whether guest memory holds `expected`, compared in place; `None` if
    /// the range is not all RAM.
    pub fn equals(&self, address: u64, expected: &[u8]) -> Option<bool> {
        let source = self.locate(address, expected.len())?;
        // SAFETY: as in `read`; the guest may write the bytes meanwhile,
        // which a copy would race with alike.
        let held = unsafe { std::slice::from_raw_parts(source.cast_const(), expected.len()) };
        Some(held == expected)
    }

    /// Copies `data` into guest memory; `None` if the range is not all RAM.
    /// The write watch notes its pages written.
    pub fn write(&self, address: u64, data: &[u8]) -> Option<()> {
        let destination = self.locate(address, data.len())?;
        // SAFETY: as in `read`.
        unsafe { ptr::copy_nonoverlapping(data.as_ptr(), destination, data.len()) };
        if let Some(watch) = self.watch.get()
            && let Some((first, end)) = self.pages(watch, address, data.len())
        {
            for page in first..end {
                watch.note_written(page);
            }
        }
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

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    /// The permission changes the watch asked for: (ipa, size, writable).
    static CHANGES: Mutex<Vec<(u64, usize, bool)>> = Mutex::new(Vec::new());

    fn record(ipa: u64, size: usize, writable: bool) -> bool {
        CHANGES.lock().unwrap().push((ipa, size, writable));
        true
    }

    #[test]
    fn watched_ranges_see_guest_and_host_writes() {
        const BASE: u64 = 0x8000_0000;
        const PAGE: u64 = 0x4000;
        let ram = GuestRam::new(BASE, 8 * PAGE as usize).unwrap();
        // Without a watch nothing is known unwritten.
        assert_eq!(ram.watch(BASE, 16), None);
        assert!(!ram.unwritten_since(BASE, 16, 1));
        ram.watch_writes(PAGE, record);

        // Pages 1 and 2 lose their write permission in one change.
        let mark = ram.watch(BASE + PAGE + 8, PAGE as usize).unwrap();
        assert_eq!(CHANGES.lock().unwrap().as_slice(), [(BASE + PAGE, 2 * PAGE as usize, false)]);
        assert!(ram.unwritten_since(BASE + PAGE + 8, PAGE as usize, mark));

        // A host write to page 0 leaves the range unwritten.
        ram.write(BASE, &[1; 8]).unwrap();
        assert!(ram.unwritten_since(BASE + PAGE + 8, PAGE as usize, mark));

        // A guest write to page 2 faults: the page is writable again and
        // the range written.
        assert!(ram.write_fault(BASE + 2 * PAGE + 4));
        assert_eq!(CHANGES.lock().unwrap().last(), Some(&(BASE + 2 * PAGE, PAGE as usize, true)));
        assert!(!ram.unwritten_since(BASE + PAGE + 8, PAGE as usize, mark));
        // Page 1 alone is still unwritten.
        assert!(ram.unwritten_since(BASE + PAGE, 8, mark));

        // Watched again: only page 2 needs its permission taken, and the
        // new mark sees the range unwritten until a host write to it.
        let again = ram.watch(BASE + PAGE + 8, PAGE as usize).unwrap();
        assert!(again > mark);
        assert_eq!(CHANGES.lock().unwrap().last(), Some(&(BASE + 2 * PAGE, PAGE as usize, false)));
        assert!(ram.unwritten_since(BASE + PAGE + 8, PAGE as usize, again));
        ram.write(BASE + 2 * PAGE, &[2; 4]).unwrap();
        assert!(!ram.unwritten_since(BASE + PAGE + 8, PAGE as usize, again));

        // Outside RAM the watch has nothing to say.
        assert!(!ram.write_fault(BASE + 8 * PAGE));
        assert_eq!(ram.watch(BASE + 8 * PAGE, 1), None);
    }
}
