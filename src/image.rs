//! Kernel image as package2 places it in DRAM.
//!
//! The image is one contiguous blob: kernel proper, embedded INI1 (initial
//! process archive) and the appended kernel loader. Nothing in it is described
//! by a header; the only self-description is the entry stub, which passes the
//! loader three PC-relative pointers. This module decodes that stub so the
//! layout the loader will see is known before the guest runs.

use std::{fmt, io, ops::Range, path::Path};

pub const PAGE_SIZE: u64 = 0x1000;

const EM_AARCH64: u16 = 183;
const PT_LOAD: u32 = 1;

/// The stub is short straight-line code; the loader call is within this many
/// instructions of the branch target or the image is not a kernel we know.
const ENTRY_STUB_SCAN_INSNS: u64 = 64;

/// The loader ignores an INI1 larger than this (it substitutes an empty
/// header), so such a blob is not an INI1 for our purposes either.
const INI1_MAX_SIZE: u32 = 0xC0_0000;

#[derive(Debug)]
pub enum ImageError {
    Io(io::Error),
    Elf(&'static str),
    EntryStub(&'static str),
    Layout(&'static str),
}

impl fmt::Display for ImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "cannot read kernel image: {error}"),
            Self::Elf(what) => write!(f, "not a loadable AArch64 ELF: {what}"),
            Self::EntryStub(what) => write!(f, "unrecognized kernel entry stub: {what}"),
            Self::Layout(what) => write!(f, "kernel layout rejected: {what}"),
        }
    }
}

impl std::error::Error for ImageError {}

impl From<io::Error> for ImageError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// Section bounds from the kernel layout block, as physical addresses at the
/// image's load address (the block itself stores `i32` offsets relative to its
/// own address; the loader rebases them the same way).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelLayout {
    pub text: Range<u64>,
    pub rodata: Range<u64>,
    pub data: Range<u64>,
    pub bss: Range<u64>,
    pub image_end: u64,
    pub dynamic: u64,
    pub init_array: Range<u64>,
    /// Where the loader stores TTBR0/TTBR1/TCR/MAIR/SCTLR for secondary cores.
    pub translation_registers: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ini1Header {
    pub address: u64,
    pub size: u32,
    pub process_count: u32,
}

#[derive(Debug)]
pub struct KernelImage {
    blob: Blob,
    pub early_entry: u64,
    pub layout_address: u64,
    pub layout: KernelLayout,
    /// `None` when the embedded INI1 slot holds no valid archive.
    pub ini1: Option<Ini1Header>,
    pub loader_entry: u64,
}

impl KernelImage {
    pub fn load(path: &Path) -> Result<Self, ImageError> {
        Self::from_elf(&std::fs::read(path)?)
    }

    /// Accepts the ELF wrapping produced by `pkg2elf.py`: PT_LOAD segments
    /// whose physical addresses are where package2 would place the image.
    pub fn from_elf(elf: &[u8]) -> Result<Self, ImageError> {
        let (load_address, bytes) = flatten_elf(elf)?;
        Self::from_flat(load_address, bytes)
    }

    pub fn from_flat(load_address: u64, bytes: Vec<u8>) -> Result<Self, ImageError> {
        let blob = Blob {
            load_address,
            bytes,
        };
        let stub = blob.decode_entry_stub()?;
        if stub.image_base != load_address {
            return Err(ImageError::EntryStub(
                "stub image base is not the load address",
            ));
        }
        Ok(Self {
            early_entry: stub.early_entry,
            layout_address: stub.layout,
            layout: blob.decode_layout(stub.layout)?,
            ini1: blob.decode_ini1(stub.ini1),
            loader_entry: stub.loader_entry,
            blob,
        })
    }

    /// Physical address of the first byte, which is also the entry point.
    pub fn load_address(&self) -> u64 {
        self.blob.load_address
    }

    pub fn end(&self) -> u64 {
        self.blob.end()
    }

    pub fn bytes(&self) -> &[u8] {
        &self.blob.bytes
    }

    /// Link-time address of the image page whose contents equal `page`, if
    /// exactly one does. The loader shuffles the physical pages behind the
    /// kernel's virtual mapping, so contents are the only stable way to map a
    /// relocated code address back to the image.
    pub fn locate_page(&self, page: &[u8]) -> Option<u64> {
        let mut matches = self
            .blob
            .bytes
            .as_chunks::<{ PAGE_SIZE as usize }>()
            .0
            .iter()
            .enumerate()
            .filter(|(_, candidate)| candidate.as_slice() == page);
        let (index, _) = matches.next()?;
        matches
            .next()
            .is_none()
            .then(|| self.blob.load_address + index as u64 * PAGE_SIZE)
    }
}

#[derive(Debug)]
struct Blob {
    load_address: u64,
    bytes: Vec<u8>,
}

struct EntryStub {
    early_entry: u64,
    image_base: u64,
    layout: u64,
    ini1: u64,
    loader_entry: u64,
}

impl Blob {
    fn end(&self) -> u64 {
        self.load_address + self.bytes.len() as u64
    }

    fn slice(&self, address: u64, len: usize) -> Option<&[u8]> {
        let offset = usize::try_from(address.checked_sub(self.load_address)?).ok()?;
        self.bytes.get(offset..offset.checked_add(len)?)
    }

    fn u32_at(&self, address: u64) -> Option<u32> {
        Some(u32::from_le_bytes(
            self.slice(address, 4)?.try_into().unwrap(),
        ))
    }

    fn u64_at(&self, address: u64) -> Option<u64> {
        Some(u64::from_le_bytes(
            self.slice(address, 8)?.try_into().unwrap(),
        ))
    }

    /// The entry point is `B early_entry`; early entry ends its setup with
    ///
    /// ```text
    /// ADR X0, image_base        ADR X4, p_loader
    /// ADR X1, layout            LDR X3, [X4]
    /// ADR X4, p_ini1            ADD X3, X3, X4
    /// LDR X2, [X4]              BLR X3
    /// ADD X2, X2, X4
    /// ```
    ///
    /// where `p_ini1`/`p_loader` hold self-relative offsets.
    fn decode_entry_stub(&self) -> Result<EntryStub, ImageError> {
        const LDR_X2_X4: u32 = 0xF940_0082;
        const ADD_X2_X2_X4: u32 = 0x8B04_0042;
        const LDR_X3_X4: u32 = 0xF940_0083;
        const ADD_X3_X3_X4: u32 = 0x8B04_0063;
        const BLR_X3: u32 = 0xD63F_0060;
        const DIFFERS: ImageError = ImageError::EntryStub("loader call sequence differs");

        let first = self
            .u32_at(self.load_address)
            .ok_or(ImageError::EntryStub("image is empty"))?;
        let early_entry = branch_target(first, self.load_address).ok_or(ImageError::EntryStub(
            "first instruction is not an unconditional branch",
        ))?;
        let call = (0..ENTRY_STUB_SCAN_INSNS)
            .map(|index| early_entry.wrapping_add(index * 4))
            .find(|&address| self.u32_at(address) == Some(BLR_X3))
            .ok_or(ImageError::EntryStub("no loader call after early entry"))?;
        // (address, instruction) `back` instructions before the call.
        let before = |back: u64| {
            let address = call.wrapping_sub(back * 4);
            self.u32_at(address).map(|insn| (address, insn))
        };

        let fixed = [
            (5, LDR_X2_X4),
            (4, ADD_X2_X2_X4),
            (2, LDR_X3_X4),
            (1, ADD_X3_X3_X4),
        ];
        if fixed
            .iter()
            .any(|&(back, expected)| before(back).map(|(_, insn)| insn) != Some(expected))
        {
            return Err(DIFFERS);
        }
        let adr = |back: u64, register: u32| {
            before(back)
                .and_then(|(pc, insn)| adr_target(insn, pc, register))
                .ok_or(DIFFERS)
        };
        let self_relative = |pointer: u64| {
            self.u64_at(pointer)
                .map(|offset| pointer.wrapping_add(offset))
                .ok_or(ImageError::EntryStub(
                    "self-relative pointer outside the image",
                ))
        };
        Ok(EntryStub {
            early_entry,
            image_base: adr(8, 0)?,
            layout: adr(7, 1)?,
            ini1: self_relative(adr(6, 4)?)?,
            loader_entry: self_relative(adr(3, 4)?)?,
        })
    }

    fn decode_layout(&self, address: u64) -> Result<KernelLayout, ImageError> {
        let raw = self
            .slice(address, 13 * 4)
            .ok_or(ImageError::Layout("block outside the image"))?;
        let mut fields = raw.as_chunks::<4>().0.iter().map(|field| {
            let offset = i32::from_le_bytes(*field);
            address.wrapping_add_signed(offset.into())
        });
        let mut next = || fields.next().unwrap();
        let layout = KernelLayout {
            text: next()..next(),
            rodata: next()..next(),
            data: next()..next(),
            bss: next()..next(),
            image_end: next(),
            dynamic: next(),
            init_array: next()..next(),
            translation_registers: next(),
        };

        // The loader spins forever on any of these, so fail before the guest does.
        let page_aligned = [
            self.load_address,
            layout.text.start,
            layout.text.end,
            layout.rodata.start,
            layout.rodata.end,
            layout.data.start,
            layout.bss.end,
        ];
        if page_aligned.iter().any(|address| address % PAGE_SIZE != 0) {
            return Err(ImageError::Layout(
                "a bound the loader requires page-aligned is not",
            ));
        }
        let ordered = [
            self.load_address,
            layout.text.start,
            layout.text.end,
            layout.rodata.start,
            layout.rodata.end,
            layout.data.start,
            layout.data.end,
            layout.bss.start,
            layout.bss.end,
            layout.image_end,
            self.end(),
        ];
        if !ordered.is_sorted() {
            return Err(ImageError::Layout(
                "sections are not ordered inside the image",
            ));
        }
        Ok(layout)
    }

    fn decode_ini1(&self, address: u64) -> Option<Ini1Header> {
        if self.slice(address, 4)? != b"INI1" {
            return None;
        }
        let size = self.u32_at(address + 4)?;
        let process_count = self.u32_at(address + 8)?;
        (size <= INI1_MAX_SIZE && self.slice(address, size as usize).is_some()).then_some(
            Ini1Header {
                address,
                size,
                process_count,
            },
        )
    }
}

fn sign_extend(value: u32, bits: u32) -> i64 {
    let shift = 64 - bits;
    (i64::from(value) << shift) >> shift
}

/// Target of `B label`.
fn branch_target(insn: u32, pc: u64) -> Option<u64> {
    (insn & 0xFC00_0000 == 0x1400_0000)
        .then(|| pc.wrapping_add_signed(sign_extend(insn & 0x03FF_FFFF, 26) * 4))
}

/// Target of `ADR X<rd>, label`.
fn adr_target(insn: u32, pc: u64, rd: u32) -> Option<u64> {
    if insn & 0x9F00_001F != 0x1000_0000 | rd {
        return None;
    }
    let immediate = ((insn >> 5) & 0x7_FFFF) << 2 | (insn >> 29) & 3;
    Some(pc.wrapping_add_signed(sign_extend(immediate, 21)))
}

/// Lays every PT_LOAD segment out at its physical address; returns the lowest
/// address and the bytes from there to the end of the highest segment.
fn flatten_elf(elf: &[u8]) -> Result<(u64, Vec<u8>), ImageError> {
    const TRUNCATED: ImageError = ImageError::Elf("truncated");
    let field = |offset: usize, len: usize| {
        offset
            .checked_add(len)
            .and_then(|end| elf.get(offset..end))
            .ok_or(TRUNCATED)
    };
    let u16_at = |offset| field(offset, 2).map(|raw| u16::from_le_bytes(raw.try_into().unwrap()));
    let u32_at = |offset| field(offset, 4).map(|raw| u32::from_le_bytes(raw.try_into().unwrap()));
    let u64_at = |offset| field(offset, 8).map(|raw| u64::from_le_bytes(raw.try_into().unwrap()));

    // Magic, ELFCLASS64, ELFDATA2LSB.
    if field(0, 6)? != b"\x7fELF\x02\x01" {
        return Err(ImageError::Elf("not a little-endian ELF64"));
    }
    if u16_at(0x12)? != EM_AARCH64 {
        return Err(ImageError::Elf("machine is not AArch64"));
    }
    let table = usize::try_from(u64_at(0x20)?).map_err(|_| TRUNCATED)?;
    let entry_size = usize::from(u16_at(0x36)?);
    let entry_count = usize::from(u16_at(0x38)?);

    let mut segments = Vec::new();
    for index in 0..entry_count {
        let header = table.checked_add(index * entry_size).ok_or(TRUNCATED)?;
        if u32_at(header)? != PT_LOAD {
            continue;
        }
        let file_offset = u64_at(header + 0x08)?;
        let address = u64_at(header + 0x18)?;
        let file_size = u64_at(header + 0x20)?;
        let memory_size = u64_at(header + 0x28)?;
        if memory_size < file_size {
            return Err(ImageError::Elf("segment larger in file than in memory"));
        }
        let contents = usize::try_from(file_offset)
            .ok()
            .zip(usize::try_from(file_size).ok())
            .and_then(|(start, len)| elf.get(start..start.checked_add(len)?))
            .ok_or(ImageError::Elf("segment data outside the file"))?;
        segments.push((address, memory_size, contents));
    }

    let start = segments
        .iter()
        .map(|&(address, ..)| address)
        .min()
        .ok_or(ImageError::Elf("no PT_LOAD segment"))?;
    let end = segments
        .iter()
        .try_fold(start, |end, &(address, size, _)| {
            Some(end.max(address.checked_add(size)?))
        })
        .ok_or(ImageError::Elf("segment wraps the address space"))?;
    // A whole kernel image is a few MiB; anything far larger is not one.
    let len = usize::try_from(end - start)
        .ok()
        .filter(|&len| len <= 0x1000_0000)
        .ok_or(ImageError::Elf("segments span an implausible range"))?;
    let mut bytes = vec![0; len];
    for (address, _, contents) in segments {
        let offset = (address - start) as usize;
        bytes[offset..offset + contents.len()].copy_from_slice(contents);
    }
    Ok((start, bytes))
}
