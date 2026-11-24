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
//!
//! # Read guard
//!
//! A host cache newer than guest memory (the GPU's render targets kept on
//! the host GPU) can have memory brought up to date only when someone
//! looks: [`GuestRam::guard`] takes every guest permission on the pages of
//! a range and gives a [`DramGuard`] that holds them so until dropped
//! (pages count their guards). A guest access to such a page faults to the
//! host ([`GuestRam::guard_fault`]), and a host read of one
//! ([`GuestRam::read`], [`GuestRam::equals`]) looks first: either calls
//! the hook ([`GuestRam::set_read_hook`]) with the guarded pages touched,
//! which is to write what the caches hold there and drop their guards,
//! before the access goes on. The hook runs [`exempt`]: so do the threads
//! of the agent keeping the caches, which bring memory up to date
//! themselves before they read it (a hook called from them could wait on
//! their own work). Host writes need no hook: the write watch notes them,
//! and the caches give way to what was written.

use std::cell::Cell;
use std::sync::atomic::{AtomicU16, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
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
    /// The read guard's hook ([`GuestRam::set_read_hook`]).
    hook: OnceLock<ReadHook>,
}

/// What the guest may do with a page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    /// Nothing: every access faults ([`GuestRam::guard`]).
    None,
    /// Read and execute ([`GuestRam::watch`]).
    Read,
    /// Read, write and execute.
    All,
}

/// Sets the guest's stage-2 permissions of `size` bytes at guest-physical
/// `ipa`; whether it could.
pub type Protect = fn(ipa: u64, size: usize, access: Access) -> bool;

/// Brings guest-physical memory `start..end` up to date for a reader (see
/// the module documentation).
pub type ReadHook = Box<dyn Fn(u64, u64) + Send + Sync>;

/// The write watch's and read guard's state (see the module
/// documentation).
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
    /// Per page, the guards holding it.
    guards: Box<[AtomicU16]>,
    /// Pages with a guard.
    guarded: AtomicUsize,
}

impl Watch {
    fn note_written(&self, page: usize) {
        self.written[page].fetch_max(self.mark.load(Ordering::SeqCst), Ordering::SeqCst);
    }

    fn is_protected(&self, page: usize) -> bool {
        self.protected[page / 64].load(Ordering::SeqCst) & (1 << (page % 64)) != 0
    }

    fn is_guarded(&self, page: usize) -> bool {
        self.guards[page].load(Ordering::SeqCst) != 0
    }

    /// What the guest may do with an unguarded page.
    fn unguarded_access(&self, page: usize) -> Access {
        if self.is_protected(page) {
            Access::Read
        } else {
            Access::All
        }
    }
}

thread_local! {
    /// Nesting of [`exempt`] on this thread.
    static EXEMPT: Cell<u32> = const { Cell::new(0) };
}

/// Host reads on this thread skip the read guard while the value lives
/// (see the module documentation).
pub struct Exempt(());

/// Exempts this thread's host reads from the read guard until the value
/// is dropped.
pub fn exempt() -> Exempt {
    EXEMPT.with(|depth| depth.set(depth.get() + 1));
    Exempt(())
}

impl Drop for Exempt {
    fn drop(&mut self) {
        EXEMPT.with(|depth| depth.set(depth.get() - 1));
    }
}

fn is_exempt() -> bool {
    EXEMPT.with(|depth| depth.get() != 0)
}

/// Guest-physical ranges kept from the guest's access, and looked at
/// before host reads, until dropped ([`GuestRam::guard`]).
pub struct DramGuard {
    ram: Arc<GuestRam>,
    ranges: Vec<(u64, u64)>,
}

impl DramGuard {
    /// The guarded ranges, `start..end` each.
    pub fn ranges(&self) -> &[(u64, u64)] {
        &self.ranges
    }

    /// Whether a guarded range overlaps `start..end`, page-wise.
    pub fn touches(&self, start: u64, end: u64) -> bool {
        let page = self.ram.watch.get().map_or(1, |watch| watch.page);
        self.ranges.iter().any(|&(s, e)| {
            s / page * page < end.next_multiple_of(page) && start / page * page < e.next_multiple_of(page)
        })
    }

    /// Takes `other`'s ranges too (both of one block).
    pub fn merge(&mut self, mut other: DramGuard) {
        self.ranges.append(&mut other.ranges);
    }
}

impl Drop for DramGuard {
    fn drop(&mut self) {
        for &(start, end) in &self.ranges {
            self.ram.unguard(start, (end - start) as usize);
        }
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
            hook: OnceLock::new(),
        })
    }

    /// Starts the write watch and the read guard (see the module
    /// documentation), for a block just mapped writable into a VM whose
    /// stage-2 permissions `protect` sets, in pages of `page` bytes. Only
    /// the first call counts.
    pub fn watch_writes(&self, page: u64, protect: Protect) {
        let pages = self.len.div_ceil(page as usize);
        let _ = self.watch.set(Watch {
            page,
            protect,
            lock: Mutex::new(()),
            protected: (0..pages.div_ceil(64)).map(|_| AtomicU64::new(0)).collect(),
            written: (0..pages).map(|_| AtomicU64::new(0)).collect(),
            mark: AtomicU64::new(1),
            guards: (0..pages).map(|_| AtomicU16::new(0)).collect(),
            guarded: AtomicUsize::new(0),
        });
    }

    /// Sets what brings guarded memory up to date for a reader; only the
    /// first call counts.
    pub fn set_read_hook(&self, hook: ReadHook) {
        let _ = self.hook.set(hook);
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

    /// Sets the permissions of each run of pages in `first..end` that
    /// `pick` gives an access for, the same for the whole run.
    fn protect_runs(
        &self,
        watch: &Watch,
        first: usize,
        end: usize,
        pick: impl Fn(usize) -> Option<Access>,
    ) -> bool {
        let mut page = first;
        while page < end {
            let Some(access) = pick(page) else {
                page += 1;
                continue;
            };
            let run = (page + 1..end)
                .find(|&p| pick(p) != Some(access))
                .unwrap_or(end);
            let ipa = self.base + page as u64 * watch.page;
            let size = ((run - page) as u64 * watch.page).min(self.base + self.len as u64 - ipa);
            if !(watch.protect)(ipa, size as usize, access) {
                return false;
            }
            page = run;
        }
        true
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
        // A guarded page keeps no permission at all meanwhile.
        let changed = self.protect_runs(watch, first, end, |page| {
            (!watch.is_protected(page) && !watch.is_guarded(page)).then_some(Access::Read)
        });
        if !changed {
            return None;
        }
        for page in first..end {
            watch.protected[page / 64].fetch_or(1 << (page % 64), Ordering::SeqCst);
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
            if !watch.is_guarded(page) {
                let at = self.base + page as u64 * watch.page;
                let size = watch.page.min(self.base + self.len as u64 - at);
                if !(watch.protect)(at, size as usize, Access::All) {
                    return false;
                }
            }
            watch.protected[page / 64].fetch_and(!(1 << (page % 64)), Ordering::SeqCst);
        }
        // A fault on a page another core's fault gave back is retried too.
        watch.note_written(page);
        true
    }

    /// Guards the pages of `len` bytes at `address` (see the module
    /// documentation) until the guard is dropped; `None` without a watch,
    /// outside RAM, or where the permissions could not be taken.
    pub fn guard(self: &Arc<Self>, address: u64, len: usize) -> Option<DramGuard> {
        let watch = self.watch.get()?;
        let (first, end) = self.pages(watch, address, len)?;
        let _held = watch.lock.lock().unwrap();
        if !self.protect_runs(watch, first, end, |page| {
            (!watch.is_guarded(page)).then_some(Access::None)
        }) {
            return None;
        }
        for page in first..end {
            if watch.guards[page].fetch_add(1, Ordering::SeqCst) == 0 {
                watch.guarded.fetch_add(1, Ordering::SeqCst);
            }
        }
        Some(DramGuard {
            ram: Arc::clone(self),
            ranges: vec![(address, address + len as u64)],
        })
    }

    /// Drops one guard of the pages of `len` bytes at `address`; a page
    /// left with none gets the permissions the write watch gives it.
    fn unguard(&self, address: u64, len: usize) {
        let Some(watch) = self.watch.get() else {
            return;
        };
        let Some((first, end)) = self.pages(watch, address, len) else {
            return;
        };
        let _held = watch.lock.lock().unwrap();
        for page in first..end {
            let held = watch.guards[page].load(Ordering::SeqCst);
            if held != 0 {
                watch.guards[page].store(held - 1, Ordering::SeqCst);
                if held == 1 {
                    watch.guarded.fetch_sub(1, Ordering::SeqCst);
                }
            }
        }
        self.protect_runs(watch, first, end, |page| {
            (!watch.is_guarded(page)).then(|| watch.unguarded_access(page))
        });
    }

    /// A guest access at guest-physical `ipa` faulted: if a guarded page,
    /// the hook brings it up to date and the access is to be retried
    /// (`true`); `false` for a page with no guard.
    pub fn guard_fault(&self, ipa: u64) -> bool {
        let Some(watch) = self.watch.get() else {
            return false;
        };
        let Some((page, _)) = self.pages(watch, ipa, 1) else {
            return false;
        };
        if !watch.is_guarded(page) {
            return false;
        }
        self.reveal(watch, page, page + 1);
        true
    }

    /// Calls the hook for pages `first..end` and makes sure none of them
    /// keeps a guard: a cache that could not tell what it held there
    /// lets the reader see memory as it is.
    fn reveal(&self, watch: &Watch, first: usize, end: usize) {
        let start = self.base + first as u64 * watch.page;
        let stop = (self.base + end as u64 * watch.page).min(self.base + self.len as u64);
        if let Some(hook) = self.hook.get() {
            let _exempt = exempt();
            hook(start, stop);
        }
        let _held = watch.lock.lock().unwrap();
        let mut left = false;
        for page in first..end {
            if watch.guards[page].swap(0, Ordering::SeqCst) != 0 {
                watch.guarded.fetch_sub(1, Ordering::SeqCst);
                left = true;
            }
        }
        if left {
            self.protect_runs(watch, first, end, |page| Some(watch.unguarded_access(page)));
        }
    }

    /// Before a host read of `len` bytes at `address`: guarded pages in it
    /// are brought up to date, unless this thread is exempt.
    fn before_read(&self, address: u64, len: usize) {
        let Some(watch) = self.watch.get() else {
            return;
        };
        if watch.guarded.load(Ordering::SeqCst) == 0 || len == 0 || is_exempt() {
            return;
        }
        let Some((first, end)) = self.pages(watch, address, len) else {
            return;
        };
        let touched: Vec<usize> = (first..end).filter(|&page| watch.is_guarded(page)).collect();
        if let (Some(&low), Some(&high)) = (touched.first(), touched.last()) {
            self.reveal(watch, low, high + 1);
        }
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

    /// Copies guest memory into `buffer`; `None` if the range is not all
    /// RAM. Guarded pages are brought up to date first.
    pub fn read(&self, address: u64, buffer: &mut [u8]) -> Option<()> {
        let source = self.locate(address, buffer.len())?;
        self.before_read(address, buffer.len());
        // SAFETY: `locate` bounds-checked the range against the mapping, and
        // `buffer` is a distinct host allocation.
        unsafe { ptr::copy_nonoverlapping(source, buffer.as_mut_ptr(), buffer.len()) };
        Some(())
    }

    /// Whether guest memory holds `expected`, compared in place; `None` if
    /// the range is not all RAM. Guarded pages are brought up to date
    /// first.
    pub fn equals(&self, address: u64, expected: &[u8]) -> Option<bool> {
        let source = self.locate(address, expected.len())?;
        self.before_read(address, expected.len());
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

    /// The permission changes asked for, by test: (ipa, size, access).
    static CHANGES: Mutex<Vec<(u64, usize, Access)>> = Mutex::new(Vec::new());
    static GUARD_CHANGES: Mutex<Vec<(u64, usize, Access)>> = Mutex::new(Vec::new());

    fn record(ipa: u64, size: usize, access: Access) -> bool {
        CHANGES.lock().unwrap().push((ipa, size, access));
        true
    }

    fn record_guard(ipa: u64, size: usize, access: Access) -> bool {
        GUARD_CHANGES.lock().unwrap().push((ipa, size, access));
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
        assert_eq!(
            CHANGES.lock().unwrap().as_slice(),
            [(BASE + PAGE, 2 * PAGE as usize, Access::Read)]
        );
        assert!(ram.unwritten_since(BASE + PAGE + 8, PAGE as usize, mark));

        // A host write to page 0 leaves the range unwritten.
        ram.write(BASE, &[1; 8]).unwrap();
        assert!(ram.unwritten_since(BASE + PAGE + 8, PAGE as usize, mark));

        // A guest write to page 2 faults: the page is writable again and
        // the range written.
        assert!(ram.write_fault(BASE + 2 * PAGE + 4));
        assert_eq!(
            CHANGES.lock().unwrap().last(),
            Some(&(BASE + 2 * PAGE, PAGE as usize, Access::All))
        );
        assert!(!ram.unwritten_since(BASE + PAGE + 8, PAGE as usize, mark));
        // Page 1 alone is still unwritten.
        assert!(ram.unwritten_since(BASE + PAGE, 8, mark));

        // Watched again: only page 2 needs its permission taken, and the
        // new mark sees the range unwritten until a host write to it.
        let again = ram.watch(BASE + PAGE + 8, PAGE as usize).unwrap();
        assert!(again > mark);
        assert_eq!(
            CHANGES.lock().unwrap().last(),
            Some(&(BASE + 2 * PAGE, PAGE as usize, Access::Read))
        );
        assert!(ram.unwritten_since(BASE + PAGE + 8, PAGE as usize, again));
        ram.write(BASE + 2 * PAGE, &[2; 4]).unwrap();
        assert!(!ram.unwritten_since(BASE + PAGE + 8, PAGE as usize, again));

        // Outside RAM the watch has nothing to say.
        assert!(!ram.write_fault(BASE + 8 * PAGE));
        assert_eq!(ram.watch(BASE + 8 * PAGE, 1), None);
    }

    /// Guarded pages keep no permission until their last guard goes; a
    /// host read or a guest fault there calls the hook first, which is
    /// to drop the guards, and the pages come back with what the write
    /// watch gives them; exempt threads read without the hook.
    #[test]
    fn guarded_pages_are_brought_up_to_date_before_anyone_looks() {
        const BASE: u64 = 0x8000_0000;
        const PAGE: u64 = 0x4000;
        let ram = Arc::new(GuestRam::new(BASE, 8 * PAGE as usize).unwrap());
        ram.watch_writes(PAGE, record_guard);
        let held: Arc<Mutex<Vec<DramGuard>>> = Arc::default();
        let calls: Arc<Mutex<Vec<(u64, u64)>>> = Arc::default();
        {
            let (held, calls, ram) = (Arc::clone(&held), Arc::clone(&calls), Arc::clone(&ram));
            ram.clone().set_read_hook(Box::new(move |start, end| {
                calls.lock().unwrap().push((start, end));
                // The cache writes what it holds, and lets go.
                ram.write(start, &[7; 4]).unwrap();
                held.lock().unwrap().retain(|guard| !guard.touches(start, end));
            }));
        }
        // Page 2 is watched; pages 1-2 then guarded twice over page 2.
        ram.watch(BASE + 2 * PAGE, 8).unwrap();
        held.lock().unwrap().push(ram.guard(BASE + PAGE, 2 * PAGE as usize).unwrap());
        held.lock().unwrap().push(ram.guard(BASE + 2 * PAGE, 16).unwrap());
        assert_eq!(
            GUARD_CHANGES.lock().unwrap().last(),
            Some(&(BASE + PAGE, 2 * PAGE as usize, Access::None))
        );

        // Reads elsewhere, and exempt ones, call nothing.
        let mut bytes = [0; 4];
        ram.read(BASE + 4 * PAGE, &mut bytes).unwrap();
        {
            let _exempt = exempt();
            ram.read(BASE + PAGE, &mut bytes).unwrap();
        }
        assert!(calls.lock().unwrap().is_empty());

        // A read of page 2 calls the hook for it before reading.
        ram.read(BASE + 2 * PAGE, &mut bytes).unwrap();
        assert_eq!(bytes, [7; 4]);
        assert_eq!(calls.lock().unwrap().as_slice(), [(BASE + 2 * PAGE, BASE + 3 * PAGE)]);
        // Both guards went: pages 1 and 2 have their permissions back,
        // page 2 still without write.
        assert!(held.lock().unwrap().is_empty());
        let changes = GUARD_CHANGES.lock().unwrap().clone();
        assert!(changes.contains(&(BASE + PAGE, PAGE as usize, Access::All)));
        assert!(changes.contains(&(BASE + 2 * PAGE, PAGE as usize, Access::Read)));

        // A guest access to a guarded page faults to the hook; with no
        // guard left it is the watch's again.
        held.lock().unwrap().push(ram.guard(BASE + 5 * PAGE, 4).unwrap());
        assert!(ram.guard_fault(BASE + 5 * PAGE + 100));
        assert_eq!(calls.lock().unwrap().last(), Some(&(BASE + 5 * PAGE, BASE + 6 * PAGE)));
        assert!(!ram.guard_fault(BASE + 5 * PAGE + 100));
    }
}
