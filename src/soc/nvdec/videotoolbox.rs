//! H.264 decoding by macOS VideoToolbox (`VTDecompressionSession`), the
//! frames returned as 8-bit 4:2:0 luma and interleaved Cb/Cr planes.
//!
//! Samples go in in decoding order and come out in decoding order: without
//! `kVTDecodeFrame_EnableTemporalProcessing` the session does not reorder,
//! and without `kVTDecodeFrame_EnableAsynchronousDecompression` it has
//! decoded a sample when `VTDecompressionSessionDecodeFrame` returns (Apple
//! VideoToolbox reference, `VTDecodeFrameFlags`). The output format asked
//! for is `kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange` ('420v'): the
//! decoded samples unconverted, Y plane then Cb/Cr pairs.

use std::ffi::c_void;

type CFTypeRef = *const c_void;
type CFAllocatorRef = *const c_void;
type CFDictionaryRef = *const c_void;
type CFMutableDictionaryRef = *mut c_void;
type CFStringRef = *const c_void;
type CFNumberRef = *const c_void;
type CMFormatDescriptionRef = *const c_void;
type CMBlockBufferRef = *const c_void;
type CMSampleBufferRef = *const c_void;
type CVImageBufferRef = *const c_void;
type VTDecompressionSessionRef = *const c_void;
type OSStatus = i32;

#[repr(C)]
#[derive(Clone, Copy)]
struct CMTime {
    value: i64,
    timescale: i32,
    flags: u32,
    epoch: i64,
}

type OutputCallback = extern "C" fn(
    refcon: *mut c_void,
    frame_refcon: *mut c_void,
    status: OSStatus,
    info: u32,
    image: CVImageBufferRef,
    presentation: CMTime,
    duration: CMTime,
);

#[repr(C)]
struct OutputCallbackRecord {
    callback: OutputCallback,
    refcon: *mut c_void,
}

#[repr(C)]
struct DictionaryCallbacks {
    _private: [u8; 0],
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    static kCFTypeDictionaryKeyCallBacks: DictionaryCallbacks;
    static kCFTypeDictionaryValueCallBacks: DictionaryCallbacks;
    fn CFRelease(object: CFTypeRef);
    fn CFDictionaryCreateMutable(
        allocator: CFAllocatorRef,
        capacity: isize,
        keys: *const DictionaryCallbacks,
        values: *const DictionaryCallbacks,
    ) -> CFMutableDictionaryRef;
    fn CFDictionarySetValue(dictionary: CFMutableDictionaryRef, key: CFTypeRef, value: CFTypeRef);
    fn CFNumberCreate(allocator: CFAllocatorRef, kind: isize, value: *const c_void) -> CFNumberRef;
}

#[link(name = "CoreMedia", kind = "framework")]
unsafe extern "C" {
    fn CMVideoFormatDescriptionCreateFromH264ParameterSets(
        allocator: CFAllocatorRef,
        count: usize,
        pointers: *const *const u8,
        sizes: *const usize,
        nal_unit_header_length: i32,
        out: *mut CMFormatDescriptionRef,
    ) -> OSStatus;
    fn CMBlockBufferCreateWithMemoryBlock(
        allocator: CFAllocatorRef,
        memory: *mut c_void,
        length: usize,
        block_allocator: CFAllocatorRef,
        custom_source: *const c_void,
        offset: usize,
        data_length: usize,
        flags: u32,
        out: *mut CMBlockBufferRef,
    ) -> OSStatus;
    fn CMBlockBufferReplaceDataBytes(
        source: *const c_void,
        destination: CMBlockBufferRef,
        offset: usize,
        length: usize,
    ) -> OSStatus;
    fn CMSampleBufferCreateReady(
        allocator: CFAllocatorRef,
        data: CMBlockBufferRef,
        format: CMFormatDescriptionRef,
        samples: isize,
        timing_entries: isize,
        timing: *const c_void,
        size_entries: isize,
        sizes: *const usize,
        out: *mut CMSampleBufferRef,
    ) -> OSStatus;
}

#[link(name = "CoreVideo", kind = "framework")]
unsafe extern "C" {
    static kCVPixelBufferPixelFormatTypeKey: CFStringRef;
    fn CVPixelBufferLockBaseAddress(buffer: CVImageBufferRef, flags: u64) -> i32;
    fn CVPixelBufferUnlockBaseAddress(buffer: CVImageBufferRef, flags: u64) -> i32;
    fn CVPixelBufferGetPixelFormatType(buffer: CVImageBufferRef) -> u32;
    fn CVPixelBufferGetPlaneCount(buffer: CVImageBufferRef) -> usize;
    fn CVPixelBufferGetWidthOfPlane(buffer: CVImageBufferRef, plane: usize) -> usize;
    fn CVPixelBufferGetHeightOfPlane(buffer: CVImageBufferRef, plane: usize) -> usize;
    fn CVPixelBufferGetBytesPerRowOfPlane(buffer: CVImageBufferRef, plane: usize) -> usize;
    fn CVPixelBufferGetBaseAddressOfPlane(buffer: CVImageBufferRef, plane: usize) -> *const u8;
}

#[link(name = "VideoToolbox", kind = "framework")]
unsafe extern "C" {
    fn VTDecompressionSessionCreate(
        allocator: CFAllocatorRef,
        format: CMFormatDescriptionRef,
        decoder_specification: CFDictionaryRef,
        image_attributes: CFDictionaryRef,
        callback: *const OutputCallbackRecord,
        out: *mut VTDecompressionSessionRef,
    ) -> OSStatus;
    fn VTDecompressionSessionDecodeFrame(
        session: VTDecompressionSessionRef,
        sample: CMSampleBufferRef,
        flags: u32,
        frame_refcon: *mut c_void,
        info_out: *mut u32,
    ) -> OSStatus;
    fn VTDecompressionSessionWaitForAsynchronousFrames(
        session: VTDecompressionSessionRef,
    ) -> OSStatus;
    fn VTDecompressionSessionInvalidate(session: VTDecompressionSessionRef);
}

/// `kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange`.
const PIXEL_FORMAT_420V: u32 = u32::from_be_bytes(*b"420v");
/// `kCFNumberSInt32Type`.
const CF_NUMBER_SINT32: isize = 3;
/// `kCMBlockBufferAssureMemoryNowFlag`.
const ASSURE_MEMORY_NOW: u32 = 1;

/// A decoded picture: `width`×`height` luma bytes, then `height / 2` rows
/// of `width` bytes of Cb/Cr pairs (Cb first).
#[derive(Debug, Default)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub luma: Vec<u8>,
    pub chroma: Vec<u8>,
}

/// What the output callback leaves for the call that decoded the sample.
#[derive(Default)]
struct Output {
    frame: Option<Result<Frame, String>>,
}

extern "C" fn output(
    _refcon: *mut c_void,
    frame_refcon: *mut c_void,
    status: OSStatus,
    _info: u32,
    image: CVImageBufferRef,
    _presentation: CMTime,
    _duration: CMTime,
) {
    // SAFETY: `frame_refcon` is the `Output` the decoding call passed,
    // alive until the call returns, which is after this callback.
    let out = unsafe { &mut *(frame_refcon as *mut Output) };
    out.frame = Some(if status != 0 {
        Err(format!("VideoToolbox: decoding failed (OSStatus {status})"))
    } else if image.is_null() {
        Err("VideoToolbox: no picture for the sample".into())
    } else {
        // SAFETY: `image` is a valid pixel buffer for the callback's
        // duration; its planes are read only while it is locked.
        unsafe { copy_planes(image) }
    });
}

/// Copies the two planes of a '420v' pixel buffer.
unsafe fn copy_planes(image: CVImageBufferRef) -> Result<Frame, String> {
    // SAFETY (whole function): `image` is a valid CVPixelBuffer.
    unsafe {
        let format = CVPixelBufferGetPixelFormatType(image);
        if format != PIXEL_FORMAT_420V || CVPixelBufferGetPlaneCount(image) != 2 {
            return Err(format!(
                "VideoToolbox: picture in pixel format {format:#x}, not '420v'"
            ));
        }
        const READ_ONLY: u64 = 1;
        if CVPixelBufferLockBaseAddress(image, READ_ONLY) != 0 {
            return Err("VideoToolbox: picture cannot be locked".into());
        }
        let plane = |index: usize, row: usize| {
            let rows = CVPixelBufferGetHeightOfPlane(image, index);
            let stride = CVPixelBufferGetBytesPerRowOfPlane(image, index);
            let base = CVPixelBufferGetBaseAddressOfPlane(image, index);
            let mut bytes = Vec::with_capacity(row * rows);
            for y in 0..rows {
                bytes.extend_from_slice(std::slice::from_raw_parts(base.add(y * stride), row));
            }
            bytes
        };
        let width = CVPixelBufferGetWidthOfPlane(image, 0);
        let height = CVPixelBufferGetHeightOfPlane(image, 0);
        let frame = Frame {
            width: width as u32,
            height: height as u32,
            luma: plane(0, width),
            // Plane 1 is width / 2 Cb/Cr pairs wide.
            chroma: plane(1, CVPixelBufferGetWidthOfPlane(image, 1) * 2),
        };
        CVPixelBufferUnlockBaseAddress(image, READ_ONLY);
        Ok(frame)
    }
}

/// A decompression session for one pair of parameter sets.
pub struct Session {
    format: CMFormatDescriptionRef,
    session: VTDecompressionSessionRef,
}

// SAFETY: the session and its format description are Core Foundation
// objects, usable from any thread; a `Session` is used by one at a time.
unsafe impl Send for Session {}

impl Session {
    /// A session for the stream of `sps` and the picture parameter sets
    /// `pps` (NAL units with their header byte, no start code).
    pub fn new(sps: &[u8], pps: &[&[u8]]) -> Result<Session, String> {
        let pointers: Vec<*const u8> = std::iter::once(sps)
            .chain(pps.iter().copied())
            .map(<[u8]>::as_ptr)
            .collect();
        let sizes: Vec<usize> = std::iter::once(sps)
            .chain(pps.iter().copied())
            .map(<[u8]>::len)
            .collect();
        let mut format = std::ptr::null();
        // SAFETY: the pointers and sizes describe live slices.
        let status = unsafe {
            CMVideoFormatDescriptionCreateFromH264ParameterSets(
                std::ptr::null(),
                pointers.len(),
                pointers.as_ptr(),
                sizes.as_ptr(),
                4,
                &mut format,
            )
        };
        if status != 0 {
            return Err(format!(
                "VideoToolbox: parameter sets refused (OSStatus {status})"
            ));
        }
        let record = OutputCallbackRecord {
            callback: output,
            refcon: std::ptr::null_mut(),
        };
        let mut session = std::ptr::null();
        // SAFETY: Core Foundation calls with valid arguments; the
        // attributes dictionary is released once the session holds it.
        let status = unsafe {
            let attributes = CFDictionaryCreateMutable(
                std::ptr::null(),
                1,
                &kCFTypeDictionaryKeyCallBacks,
                &kCFTypeDictionaryValueCallBacks,
            );
            let pixel_format = PIXEL_FORMAT_420V as i32;
            let number = CFNumberCreate(
                std::ptr::null(),
                CF_NUMBER_SINT32,
                &pixel_format as *const i32 as *const c_void,
            );
            CFDictionarySetValue(attributes, kCVPixelBufferPixelFormatTypeKey, number);
            CFRelease(number);
            let status = VTDecompressionSessionCreate(
                std::ptr::null(),
                format,
                std::ptr::null(),
                attributes,
                &record,
                &mut session,
            );
            CFRelease(attributes);
            status
        };
        if status != 0 {
            // SAFETY: `format` was created above.
            unsafe { CFRelease(format) };
            return Err(format!(
                "VideoToolbox: no decompression session (OSStatus {status})"
            ));
        }
        Ok(Session { format, session })
    }

    /// Decodes one picture: its slice NAL units, each prefixed with its
    /// length (4 bytes, big-endian).
    pub fn decode(&mut self, sample: &[u8]) -> Result<Frame, String> {
        let mut out = Output::default();
        // SAFETY: the block buffer allocates and owns a copy of `sample`;
        // `out` outlives the synchronous decode that writes it.
        let status = unsafe {
            let mut block = std::ptr::null();
            let status = CMBlockBufferCreateWithMemoryBlock(
                std::ptr::null(),
                std::ptr::null_mut(),
                sample.len(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                sample.len(),
                ASSURE_MEMORY_NOW,
                &mut block,
            );
            if status != 0 {
                return Err(format!("VideoToolbox: no block buffer (OSStatus {status})"));
            }
            let status =
                CMBlockBufferReplaceDataBytes(sample.as_ptr().cast(), block, 0, sample.len());
            let mut buffer = std::ptr::null();
            let status = if status == 0 {
                CMSampleBufferCreateReady(
                    std::ptr::null(),
                    block,
                    self.format,
                    1,
                    0,
                    std::ptr::null(),
                    1,
                    &sample.len(),
                    &mut buffer,
                )
            } else {
                status
            };
            CFRelease(block);
            if status != 0 {
                return Err(format!(
                    "VideoToolbox: no sample buffer (OSStatus {status})"
                ));
            }
            let mut info = 0;
            let status = VTDecompressionSessionDecodeFrame(
                self.session,
                buffer,
                0,
                &mut out as *mut Output as *mut c_void,
                &mut info,
            );
            VTDecompressionSessionWaitForAsynchronousFrames(self.session);
            CFRelease(buffer);
            status
        };
        match out.frame {
            Some(frame) => frame,
            None if status != 0 => {
                Err(format!("VideoToolbox: decoding failed (OSStatus {status})"))
            }
            None => Err("VideoToolbox: no picture for the sample".into()),
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // SAFETY: both were created by `Session::new` and are released once.
        unsafe {
            VTDecompressionSessionInvalidate(self.session);
            CFRelease(self.session);
            CFRelease(self.format);
        }
    }
}
