//! H.264 encoding by macOS VideoToolbox (`VTCompressionSession`), from
//! 8-bit 4:2:0 pictures: a luma plane and a plane of interleaved Cb/Cr
//! pairs (NV12).
//!
//! The session is asked for what lets its slices stand in for NVENC's
//! under the driver's parameter sets (see `h264`): no frame reordering
//! (`kVTCompressionPropertyKey_AllowFrameReordering` false), so every
//! picture comes out as it goes in and references only earlier ones (the
//! slices it gives reference one picture, the latest, which `h264`
//! checks; `kVTCompressionPropertyKey_ReferenceBufferCount` is refused
//! by the macOS 26 H.264 encoder, OSStatus -12900);
//! key frames only when asked for (`kVTEncodeFrameOptionKey_ForceKeyFrame`,
//! the interval otherwise unbounded); the entropy coder the driver's
//! picture parameter set names; High profile, whose 8x8 transform the
//! driver's set enables; real-time encoding at the driver's frame rate and
//! bit rate. `VTCompressionSessionCompleteFrames` after each picture makes
//! the call return with that picture encoded (Apple VideoToolbox
//! reference: it "forces the video encoder to emit all pending frames").

use std::ffi::c_void;

type CFTypeRef = *const c_void;
type CFAllocatorRef = *const c_void;
type CFDictionaryRef = *const c_void;
type CFMutableDictionaryRef = *mut c_void;
type CFStringRef = *const c_void;
type CFNumberRef = *const c_void;
type CFBooleanRef = *const c_void;
type CMFormatDescriptionRef = *const c_void;
type CMBlockBufferRef = *const c_void;
type CMSampleBufferRef = *const c_void;
type CVPixelBufferRef = *const c_void;
type CVPixelBufferPoolRef = *const c_void;
type VTCompressionSessionRef = *const c_void;
type OSStatus = i32;

#[repr(C)]
#[derive(Clone, Copy)]
struct CMTime {
    value: i64,
    timescale: i32,
    flags: u32,
    epoch: i64,
}

/// `kCMTimeFlags_Valid`.
const TIME_VALID: u32 = 1;
/// `kCMTimeInvalid`.
const TIME_INVALID: CMTime = CMTime {
    value: 0,
    timescale: 0,
    flags: 0,
    epoch: 0,
};

type OutputCallback = extern "C" fn(
    refcon: *mut c_void,
    frame_refcon: *mut c_void,
    status: OSStatus,
    info: u32,
    sample: CMSampleBufferRef,
);

#[repr(C)]
struct DictionaryCallbacks {
    _private: [u8; 0],
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    static kCFTypeDictionaryKeyCallBacks: DictionaryCallbacks;
    static kCFTypeDictionaryValueCallBacks: DictionaryCallbacks;
    static kCFBooleanTrue: CFBooleanRef;
    static kCFBooleanFalse: CFBooleanRef;
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
    fn CMSampleBufferGetDataBuffer(sample: CMSampleBufferRef) -> CMBlockBufferRef;
    fn CMSampleBufferGetFormatDescription(sample: CMSampleBufferRef) -> CMFormatDescriptionRef;
    fn CMBlockBufferGetDataLength(buffer: CMBlockBufferRef) -> usize;
    fn CMBlockBufferCopyDataBytes(
        buffer: CMBlockBufferRef,
        offset: usize,
        length: usize,
        destination: *mut c_void,
    ) -> OSStatus;
    fn CMVideoFormatDescriptionGetH264ParameterSetAtIndex(
        description: CMFormatDescriptionRef,
        index: usize,
        pointer: *mut *const u8,
        size: *mut usize,
        count: *mut usize,
        nal_unit_header_length: *mut i32,
    ) -> OSStatus;
}

#[link(name = "CoreVideo", kind = "framework")]
unsafe extern "C" {
    static kCVPixelBufferPixelFormatTypeKey: CFStringRef;
    static kCVPixelBufferWidthKey: CFStringRef;
    static kCVPixelBufferHeightKey: CFStringRef;
    fn CVPixelBufferPoolCreatePixelBuffer(
        allocator: CFAllocatorRef,
        pool: CVPixelBufferPoolRef,
        out: *mut CVPixelBufferRef,
    ) -> i32;
    fn CVPixelBufferLockBaseAddress(buffer: CVPixelBufferRef, flags: u64) -> i32;
    fn CVPixelBufferUnlockBaseAddress(buffer: CVPixelBufferRef, flags: u64) -> i32;
    fn CVPixelBufferGetBytesPerRowOfPlane(buffer: CVPixelBufferRef, plane: usize) -> usize;
    fn CVPixelBufferGetBaseAddressOfPlane(buffer: CVPixelBufferRef, plane: usize) -> *const u8;
}

#[link(name = "VideoToolbox", kind = "framework")]
unsafe extern "C" {
    static kVTCompressionPropertyKey_RealTime: CFStringRef;
    static kVTCompressionPropertyKey_AllowFrameReordering: CFStringRef;
    static kVTCompressionPropertyKey_ProfileLevel: CFStringRef;
    static kVTCompressionPropertyKey_H264EntropyMode: CFStringRef;
    static kVTCompressionPropertyKey_MaxKeyFrameInterval: CFStringRef;
    static kVTCompressionPropertyKey_AverageBitRate: CFStringRef;
    static kVTCompressionPropertyKey_ExpectedFrameRate: CFStringRef;
    static kVTProfileLevel_H264_High_AutoLevel: CFStringRef;
    static kVTH264EntropyMode_CABAC: CFStringRef;
    static kVTH264EntropyMode_CAVLC: CFStringRef;
    static kVTEncodeFrameOptionKey_ForceKeyFrame: CFStringRef;
    fn VTCompressionSessionCreate(
        allocator: CFAllocatorRef,
        width: i32,
        height: i32,
        codec: u32,
        encoder_specification: CFDictionaryRef,
        source_attributes: CFDictionaryRef,
        compressed_allocator: CFAllocatorRef,
        callback: OutputCallback,
        refcon: *mut c_void,
        out: *mut VTCompressionSessionRef,
    ) -> OSStatus;
    fn VTSessionSetProperty(session: CFTypeRef, key: CFStringRef, value: CFTypeRef) -> OSStatus;
    fn VTCompressionSessionPrepareToEncodeFrames(session: VTCompressionSessionRef) -> OSStatus;
    fn VTCompressionSessionGetPixelBufferPool(
        session: VTCompressionSessionRef,
    ) -> CVPixelBufferPoolRef;
    fn VTCompressionSessionEncodeFrame(
        session: VTCompressionSessionRef,
        image: CVPixelBufferRef,
        presentation: CMTime,
        duration: CMTime,
        properties: CFDictionaryRef,
        frame_refcon: *mut c_void,
        info_out: *mut u32,
    ) -> OSStatus;
    fn VTCompressionSessionCompleteFrames(
        session: VTCompressionSessionRef,
        until: CMTime,
    ) -> OSStatus;
    fn VTCompressionSessionInvalidate(session: VTCompressionSessionRef);
}

/// `kCMVideoCodecType_H264`.
const CODEC_H264: u32 = u32::from_be_bytes(*b"avc1");
/// `kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange`.
const PIXEL_FORMAT_420V: u32 = u32::from_be_bytes(*b"420v");
/// `kCFNumberSInt32Type`, `kCFNumberFloat64Type`.
const CF_NUMBER_SINT32: isize = 3;
const CF_NUMBER_FLOAT64: isize = 6;
/// `kVTEncodeInfo_FrameDropped`.
const FRAME_DROPPED: u32 = 1 << 1;

/// What the session is made for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    pub width: u32,
    pub height: u32,
    /// CABAC rather than CAVLC.
    pub cabac: bool,
    /// Bits a second.
    pub bit_rate: u32,
    pub frame_rate: f64,
}

/// One encoded picture: its NAL units (header byte first, no start code,
/// emulation prevention bytes in) and the session's parameter sets for it.
#[derive(Debug, Default)]
pub struct Encoded {
    pub sps: Vec<u8>,
    pub pps: Vec<u8>,
    pub nal_units: Vec<Vec<u8>>,
}

/// What the output callback leaves for the call that encoded the picture.
#[derive(Default)]
struct Output {
    encoded: Option<Result<Encoded, String>>,
}

extern "C" fn output(
    _refcon: *mut c_void,
    frame_refcon: *mut c_void,
    status: OSStatus,
    info: u32,
    sample: CMSampleBufferRef,
) {
    // SAFETY: `frame_refcon` is the `Output` the encoding call passed,
    // alive until it has completed the frame, which is after this callback.
    let out = unsafe { &mut *(frame_refcon as *mut Output) };
    out.encoded = Some(if status != 0 {
        Err(format!("VideoToolbox: encoding failed (OSStatus {status})"))
    } else if info & FRAME_DROPPED != 0 || sample.is_null() {
        Err("VideoToolbox: the encoder dropped the picture".into())
    } else {
        // SAFETY: `sample` is a valid sample buffer for the callback.
        unsafe { unpack(sample) }
    });
}

/// The NAL units of an encoded sample (4-byte big-endian lengths before
/// each) and its parameter sets.
unsafe fn unpack(sample: CMSampleBufferRef) -> Result<Encoded, String> {
    // SAFETY (whole function): `sample` is a valid sample buffer.
    unsafe {
        let block = CMSampleBufferGetDataBuffer(sample);
        let length = CMBlockBufferGetDataLength(block);
        let mut data = vec![0u8; length];
        if CMBlockBufferCopyDataBytes(block, 0, length, data.as_mut_ptr().cast()) != 0 {
            return Err("VideoToolbox: encoded data cannot be read".into());
        }
        let mut nal_units = Vec::new();
        let mut rest = data.as_slice();
        while !rest.is_empty() {
            let Some((size, tail)) = rest.split_first_chunk::<4>() else {
                return Err("VideoToolbox: truncated NAL unit length".into());
            };
            let size = u32::from_be_bytes(*size) as usize;
            if size > tail.len() {
                return Err("VideoToolbox: NAL unit longer than the sample".into());
            }
            nal_units.push(tail[..size].to_vec());
            rest = &tail[size..];
        }
        let description = CMSampleBufferGetFormatDescription(sample);
        let set = |index: usize| {
            let (mut pointer, mut size) = (std::ptr::null(), 0);
            let status = CMVideoFormatDescriptionGetH264ParameterSetAtIndex(
                description,
                index,
                &mut pointer,
                &mut size,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            if status != 0 {
                return Err(format!(
                    "VideoToolbox: no parameter set {index} (OSStatus {status})"
                ));
            }
            Ok(std::slice::from_raw_parts(pointer, size).to_vec())
        };
        Ok(Encoded {
            sps: set(0)?,
            pps: set(1)?,
            nal_units,
        })
    }
}

/// A compression session.
pub struct Encoder {
    settings: Settings,
    session: VTCompressionSessionRef,
    /// Pictures given so far: the presentation time of the next.
    count: i64,
}

// SAFETY: the session is a Core Foundation object, usable from any thread;
// an `Encoder` is used by one at a time.
unsafe impl Send for Encoder {}

/// Sets `key` to `value` on `session`, then releases `value` if `owned`.
unsafe fn set(session: CFTypeRef, key: CFStringRef, value: CFTypeRef, owned: bool) -> OSStatus {
    // SAFETY: the caller passes live Core Foundation objects.
    unsafe {
        let status = VTSessionSetProperty(session, key, value);
        if owned {
            CFRelease(value);
        }
        status
    }
}

fn number_i32(value: i32) -> CFNumberRef {
    // SAFETY: a plain Core Foundation constructor.
    unsafe {
        CFNumberCreate(
            std::ptr::null(),
            CF_NUMBER_SINT32,
            (&raw const value).cast(),
        )
    }
}

fn number_f64(value: f64) -> CFNumberRef {
    // SAFETY: a plain Core Foundation constructor.
    unsafe {
        CFNumberCreate(
            std::ptr::null(),
            CF_NUMBER_FLOAT64,
            (&raw const value).cast(),
        )
    }
}

impl Encoder {
    pub fn new(settings: Settings) -> Result<Encoder, String> {
        let mut session = std::ptr::null();
        // SAFETY: Core Foundation and VideoToolbox calls with valid
        // arguments; every object created here is released once.
        unsafe {
            let attributes = CFDictionaryCreateMutable(
                std::ptr::null(),
                3,
                &kCFTypeDictionaryKeyCallBacks,
                &kCFTypeDictionaryValueCallBacks,
            );
            for (key, value) in [
                (kCVPixelBufferPixelFormatTypeKey, PIXEL_FORMAT_420V as i32),
                (kCVPixelBufferWidthKey, settings.width as i32),
                (kCVPixelBufferHeightKey, settings.height as i32),
            ] {
                let number = number_i32(value);
                CFDictionarySetValue(attributes, key, number);
                CFRelease(number);
            }
            let status = VTCompressionSessionCreate(
                std::ptr::null(),
                settings.width as i32,
                settings.height as i32,
                CODEC_H264,
                std::ptr::null(),
                attributes,
                std::ptr::null(),
                output,
                std::ptr::null_mut(),
                &mut session,
            );
            CFRelease(attributes);
            if status != 0 {
                return Err(format!(
                    "VideoToolbox: no compression session for {}x{} (OSStatus {status})",
                    settings.width, settings.height
                ));
            }
            let encoder = Encoder {
                settings,
                session,
                count: 0,
            };
            let entropy = if settings.cabac {
                kVTH264EntropyMode_CABAC
            } else {
                kVTH264EntropyMode_CAVLC
            };
            let mut refused = None;
            for (name, key, value, owned) in [
                (
                    "RealTime",
                    kVTCompressionPropertyKey_RealTime,
                    kCFBooleanTrue,
                    false,
                ),
                (
                    "AllowFrameReordering",
                    kVTCompressionPropertyKey_AllowFrameReordering,
                    kCFBooleanFalse,
                    false,
                ),
                (
                    "ProfileLevel",
                    kVTCompressionPropertyKey_ProfileLevel,
                    kVTProfileLevel_H264_High_AutoLevel,
                    false,
                ),
                (
                    "H264EntropyMode",
                    kVTCompressionPropertyKey_H264EntropyMode,
                    entropy,
                    false,
                ),
                (
                    "MaxKeyFrameInterval",
                    kVTCompressionPropertyKey_MaxKeyFrameInterval,
                    number_i32(i32::MAX),
                    true,
                ),
                (
                    "AverageBitRate",
                    kVTCompressionPropertyKey_AverageBitRate,
                    number_i32(settings.bit_rate.min(i32::MAX as u32) as i32),
                    true,
                ),
                (
                    "ExpectedFrameRate",
                    kVTCompressionPropertyKey_ExpectedFrameRate,
                    number_f64(settings.frame_rate),
                    true,
                ),
            ] {
                // Every value is released, whatever the session says.
                let status = set(encoder.session, key, value, owned);
                if status != 0 && refused.is_none() {
                    refused = Some((name, status));
                }
            }
            if let Some((name, status)) = refused {
                return Err(format!(
                    "VideoToolbox: property {name} refused (OSStatus {status})"
                ));
            }
            let status = VTCompressionSessionPrepareToEncodeFrames(encoder.session);
            if status != 0 {
                return Err(format!(
                    "VideoToolbox: session not ready (OSStatus {status})"
                ));
            }
            Ok(encoder)
        }
    }

    pub fn settings(&self) -> Settings {
        self.settings
    }

    /// Encodes one picture of the session's size: `luma` rows of `width`
    /// bytes, then `chroma` rows of `width` bytes of Cb/Cr pairs, half as
    /// many. `key` asks for a key (IDR) picture.
    pub fn encode(&mut self, luma: &[u8], chroma: &[u8], key: bool) -> Result<Encoded, String> {
        let (width, height) = (self.settings.width as usize, self.settings.height as usize);
        assert_eq!(luma.len(), width * height);
        assert_eq!(chroma.len(), width * height / 2);
        let mut out = Output::default();
        // SAFETY: the pixel buffer comes from the session's pool and is
        // written only while locked; `out` outlives the completed frame
        // whose callback writes it.
        let status = unsafe {
            let pool = VTCompressionSessionGetPixelBufferPool(self.session);
            if pool.is_null() {
                return Err("VideoToolbox: the session has no pixel buffer pool".into());
            }
            let mut buffer = std::ptr::null();
            if CVPixelBufferPoolCreatePixelBuffer(std::ptr::null(), pool, &mut buffer) != 0 {
                return Err("VideoToolbox: no pixel buffer".into());
            }
            if CVPixelBufferLockBaseAddress(buffer, 0) != 0 {
                CFRelease(buffer);
                return Err("VideoToolbox: pixel buffer cannot be locked".into());
            }
            for (plane, bytes) in [luma, chroma].into_iter().enumerate() {
                let stride = CVPixelBufferGetBytesPerRowOfPlane(buffer, plane);
                let base = CVPixelBufferGetBaseAddressOfPlane(buffer, plane).cast_mut();
                for (y, row) in bytes.chunks_exact(width).enumerate() {
                    std::ptr::copy_nonoverlapping(row.as_ptr(), base.add(y * stride), width);
                }
            }
            CVPixelBufferUnlockBaseAddress(buffer, 0);
            let properties = if key {
                let properties = CFDictionaryCreateMutable(
                    std::ptr::null(),
                    1,
                    &kCFTypeDictionaryKeyCallBacks,
                    &kCFTypeDictionaryValueCallBacks,
                );
                CFDictionarySetValue(
                    properties,
                    kVTEncodeFrameOptionKey_ForceKeyFrame,
                    kCFBooleanTrue,
                );
                properties as CFDictionaryRef
            } else {
                std::ptr::null()
            };
            let time = CMTime {
                value: self.count,
                timescale: (self.settings.frame_rate.round() as i32).max(1),
                flags: TIME_VALID,
                epoch: 0,
            };
            self.count += 1;
            let mut info = 0;
            let status = VTCompressionSessionEncodeFrame(
                self.session,
                buffer,
                time,
                TIME_INVALID,
                properties,
                (&raw mut out).cast(),
                &mut info,
            );
            if !properties.is_null() {
                CFRelease(properties);
            }
            CFRelease(buffer);
            if status == 0 {
                VTCompressionSessionCompleteFrames(self.session, TIME_INVALID)
            } else {
                status
            }
        };
        match out.encoded {
            Some(encoded) => encoded,
            None if status != 0 => {
                Err(format!("VideoToolbox: encoding failed (OSStatus {status})"))
            }
            None => Err("VideoToolbox: no output for the picture".into()),
        }
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        // SAFETY: the session was created by `Encoder::new` and is
        // released once.
        unsafe {
            VTCompressionSessionInvalidate(self.session);
            CFRelease(self.session);
        }
    }
}
