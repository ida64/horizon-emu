//! System updates from Nintendo's content delivery network, fetched the way
//! a console's `nim` (program `0100000000000025`) fetches them, so that the
//! firmware for [`crate::nand`] can come from the console it is for.
//!
//! # What `nim` asks (23.5.0)
//!
//! The hosts are templates whose `%` is the environment, `lp1` on a retail
//! console ([`ENVIRONMENT`]):
//!
//! - `https://sun.hac.%.d4c.nintendo.net/v1/system_update_meta?device_id=%016llx`
//!   (`0xf1040`), asked with `Accept: application/json` (`0x11c670`),
//!   answers the newest system update as
//!   `$.system_update_metas[0].title_id` and `.title_version`.
//! - `https://atumn.hac.%.d4c.nintendo.net/t/%c/%016llx/%u?device_id=%016llx`
//!   (`0xf1350`) names the meta content of a title's version in the
//!   header `X-Nintendo-Content-ID` (`0x11c3a0`): `s` for the system
//!   update (meta type 3), `a` for any other title.
//! - `https://atumn.hac.%.d4c.nintendo.net/c/%c/%s` (`0xf17c0`) is a
//!   content by content ID, without the device ID: `s` the meta content of
//!   the system update, `a` that of another title, `c` any other content.
//! - Every request carries `User-Agent: NintendoSDK Firmware/%s-%u.%u
//!   (platform:NX; did:%016llx; eid:%s)` (`0xa5c70`): the system version
//!   the console runs with its revision, the device ID and the environment.
//!
//! No account or device token goes with any of them. What admits the
//! console is TLS: `nim` has `ssl` present the console's own client
//! certificate (`RegisterInternalPki`), whose key is in the calibration
//! data ([`crate::nand::calibration::client_identity`]). A system update's
//! meta lists its titles with their versions; each title's meta lists its
//! contents with their sizes and SHA-256.
//!
//! # Trust
//!
//! The servers present a certificate for `*.hac.lp1.d4c.nintendo.net`
//! (among other names) issued by "Nintendo Class 2 CA - G3", which this
//! module accepts as the only root. The copy here
//! (`cdn/nintendo_class_2_ca_g3.der`) is byte for byte the one at `0xC1C`
//! of `ssl_TrustedCerts.bdf` in system data `0100000000000800` of firmware
//! 23.5.0, the console's own trust store. Host names are checked as well.
//!
//! # What is checked
//!
//! Each meta content hashes to its content ID (the first half of its
//! SHA-256), decrypts (only key generation 0 is in use for them) and
//! describes the title and version asked for; each other content has the
//! size and SHA-256 its title's meta gives it. Nothing lands under its
//! final name before that.

mod http;

use std::{
    collections::HashSet,
    fmt,
    fs::{self, File},
    io::{self, BufWriter, Write},
    num::NonZeroUsize,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    thread,
};

use rsa::pkcs1::EncodeRsaPrivateKey;
use rustls::{
    ClientConfig, RootCertStore,
    pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs1KeyDer},
};
use sha2::{Digest, Sha256};

use self::http::{Connection, Response};
use crate::{
    keys::KeySet,
    nand::{
        calibration::ClientIdentity,
        cnmt::{CnmtError, ContentInfo, META_TYPE_SYSTEM_UPDATE, PackagedContentMeta},
        nca::{self, CONTENT_TYPE_META, Nca, NcaError},
    },
};

/// The environment of a retail console (`nsd`'s environment identifier).
pub const ENVIRONMENT: &str = "lp1";
const SUN: &str = "sun.hac.lp1.d4c.nintendo.net";
const ATUMN: &str = "atumn.hac.lp1.d4c.nintendo.net";
/// The system update title, whose meta names every title of a firmware.
pub const SYSTEM_UPDATE_ID: u64 = 0x0100_0000_0000_0816;
/// The system version the requests say the console runs: 23.5.0, revision
/// 1.0, as system data `0100000000000809` of that firmware holds it.
const RUNNING_VERSION: &str = "23.5.0-1.0";
const NINTENDO_CLASS_2_CA_G3: &[u8] = include_bytes!("cdn/nintendo_class_2_ca_g3.der");

/// Bytes of a content ID.
type ContentId = [u8; 16];

#[derive(Debug)]
pub enum CdnError {
    /// The TLS client cannot be set up with the console's identity.
    Tls(String),
    /// A request failed on the way.
    Request {
        url: String,
        error: io::Error,
    },
    Status {
        url: String,
        status: u16,
    },
    /// The answer is not what a console expects.
    Answer {
        url: String,
        what: &'static str,
    },
    File {
        path: PathBuf,
        error: io::Error,
    },
    Nca {
        content: String,
        error: NcaError,
    },
    Cnmt {
        content: String,
        error: CnmtError,
    },
    /// A content is not what its record says.
    Content {
        content: String,
        what: &'static str,
    },
}

impl fmt::Display for CdnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tls(why) => write!(f, "TLS: {why}"),
            Self::Request { url, error } => write!(f, "{url}: {error}"),
            Self::Status { url, status } => write!(f, "{url}: HTTP status {status}"),
            Self::Answer { url, what } => write!(f, "{url}: {what}"),
            Self::File { path, error } => write!(f, "{}: {error}", path.display()),
            Self::Nca { content, error } => write!(f, "content {content}: {error}"),
            Self::Cnmt { content, error } => write!(f, "content {content}: {error}"),
            Self::Content { content, what } => write!(f, "content {content}: {what}"),
        }
    }
}

impl std::error::Error for CdnError {}

/// What is under way, for a caller that shows it.
#[derive(Debug, Clone, Copy)]
pub enum Progress {
    /// The meta of a title's version was fetched and checked.
    Title { id: u64, version: u32 },
    /// A content is in place, `fetched` or found complete from before.
    Content {
        id: ContentId,
        size: u64,
        fetched: bool,
    },
}

/// What [`Client::download_system_update`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Downloaded {
    /// Titles, the system update included.
    pub titles: usize,
    /// Contents other than metas.
    pub contents: usize,
    pub fetched_bytes: u64,
    /// Bytes of contents found complete from an earlier run.
    pub present_bytes: u64,
}

/// `major.minor.micro.relstep` of a system update's title version, which
/// Nintendo packs as six, six, four and sixteen bits.
pub fn version_string(version: u32) -> String {
    format!(
        "{}.{}.{}.{}",
        version >> 26,
        version >> 20 & 0x3F,
        version >> 16 & 0xF,
        version & 0xFFFF
    )
}

/// The inverse of [`version_string`], or a title version as a number
/// (decimal or `0x` hex).
pub fn parse_version(text: &str) -> Option<u32> {
    if let Some(hex) = text.strip_prefix("0x") {
        return u32::from_str_radix(hex, 16).ok();
    }
    let parts: Vec<u32> = text
        .split('.')
        .map(str::parse)
        .collect::<Result<_, _>>()
        .ok()?;
    match parts[..] {
        [number] => Some(number),
        [major, minor, micro, relstep]
            if major < 64 && minor < 64 && micro < 16 && relstep <= 0xFFFF =>
        {
            Some(major << 26 | minor << 20 | micro << 16 | relstep)
        }
        _ => None,
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn parse_content_id(text: &str) -> Option<ContentId> {
    if text.len() != 32 {
        return None;
    }
    let mut id = [0; 16];
    for (byte, digits) in id.iter_mut().zip(text.as_bytes().chunks(2)) {
        *byte = u8::from_str_radix(std::str::from_utf8(digits).ok()?, 16).ok()?;
    }
    Some(id)
}

/// A writer that hashes what goes through it.
struct Hashing<W> {
    inner: W,
    hasher: Sha256,
}

impl<W: Write> Write for Hashing<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let written = self.inner.write(buf)?;
        self.hasher.update(&buf[..written]);
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

/// Where a file is written before it is checked.
fn partial(path: &Path) -> PathBuf {
    let mut name = path.file_name().expect("a file name").to_owned();
    name.push(".part");
    path.with_file_name(name)
}

/// Writes `bytes` to `path` by way of a temporary file beside it.
fn store(path: &Path, bytes: &[u8]) -> Result<(), CdnError> {
    let file_error = |path: &Path| {
        let path = path.to_owned();
        move |error| CdnError::File { path, error }
    };
    let part = partial(path);
    fs::write(&part, bytes).map_err(file_error(&part))?;
    fs::rename(&part, path).map_err(file_error(path))
}

/// Whether `path` already holds `size` bytes hashing to `hash`.
fn complete(path: &Path, size: u64, hash: &[u8; 32]) -> Result<bool, CdnError> {
    let failed = |error| CdnError::File {
        path: path.to_owned(),
        error,
    };
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(failed(error)),
    };
    if file.metadata().map_err(failed)?.len() != size {
        return Ok(false);
    }
    let mut hashing = Hashing {
        inner: io::sink(),
        hasher: Sha256::new(),
    };
    io::copy(&mut file, &mut hashing).map_err(failed)?;
    Ok(hashing.hasher.finalize()[..] == hash[..])
}

/// The packaged content meta in the meta NCA `data`.
fn content_meta(data: &[u8], keys: &KeySet, content: &str) -> Result<Vec<u8>, CdnError> {
    let nca_error = |error| CdnError::Nca {
        content: content.into(),
        error,
    };
    let archive = Nca::open(data, keys).map_err(nca_error)?;
    if archive.content_type() != CONTENT_TYPE_META {
        return Err(CdnError::Content {
            content: content.into(),
            what: "not a meta content",
        });
    }
    let section = archive.partition_section(0).map_err(nca_error)?;
    let files = nca::pfs0_files(&section).map_err(nca_error)?;
    let (_, cnmt) = files
        .into_iter()
        .find(|(name, _)| name.ends_with(".cnmt"))
        .ok_or(CdnError::Content {
            content: content.into(),
            what: "no .cnmt file in its partition FS",
        })?;
    Ok(cnmt.to_vec())
}

/// The console as Nintendo's content delivery network knows it.
pub struct Client {
    config: Arc<ClientConfig>,
    device_id: u64,
    user_agent: String,
}

impl Client {
    pub fn new(identity: &ClientIdentity) -> Result<Self, CdnError> {
        let tls = |error: rustls::Error| CdnError::Tls(error.to_string());
        let mut roots = RootCertStore::empty();
        roots
            .add(CertificateDer::from(NINTENDO_CLASS_2_CA_G3))
            .map_err(tls)?;
        let key = identity
            .key
            .to_pkcs1_der()
            .map_err(|error| CdnError::Tls(error.to_string()))?;
        let config =
            ClientConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
                .with_safe_default_protocol_versions()
                .map_err(tls)?
                .with_root_certificates(roots)
                .with_client_auth_cert(
                    vec![CertificateDer::from(identity.certificate.clone())],
                    PrivateKeyDer::Pkcs1(PrivatePkcs1KeyDer::from(key.as_bytes().to_vec())),
                )
                .map_err(tls)?;
        Ok(Self {
            config: Arc::new(config),
            device_id: identity.device_id,
            user_agent: format!(
                "NintendoSDK Firmware/{RUNNING_VERSION} (platform:NX; did:{:016x}; eid:{ENVIRONMENT})",
                identity.device_id
            ),
        })
    }

    fn connect(&self, host: &'static str) -> Connection {
        Connection::new(self.config.clone(), host)
    }

    /// Sends a request and accepts only status 200.
    fn request(
        &self,
        connection: &mut Connection,
        method: &str,
        path: &str,
        accept: Option<&str>,
    ) -> Result<Response, CdnError> {
        let host = connection.host();
        let url = || format!("https://{host}{path}");
        let mut headers = vec![("User-Agent", self.user_agent.as_str())];
        if let Some(accept) = accept {
            headers.push(("Accept", accept));
        }
        let response = connection
            .send(method, path, &headers)
            .map_err(|error| CdnError::Request { url: url(), error })?;
        if response.status != 200 {
            connection.close();
            return Err(CdnError::Status {
                url: url(),
                status: response.status,
            });
        }
        Ok(response)
    }

    /// The whole body of a `GET` of `path`.
    fn fetch(
        &self,
        connection: &mut Connection,
        path: &str,
        accept: Option<&str>,
    ) -> Result<Vec<u8>, CdnError> {
        let response = self.request(connection, "GET", path, accept)?;
        let mut body = Vec::new();
        connection
            .read_body(&response, &mut body)
            .map_err(|error| CdnError::Request {
                url: format!("https://{}{path}", connection.host()),
                error,
            })?;
        Ok(body)
    }

    /// The title version of the newest system update, as `sun` answers a
    /// console asking whether there is one.
    pub fn latest_system_update(&self) -> Result<u32, CdnError> {
        let path = format!("/v1/system_update_meta?device_id={:016x}", self.device_id);
        let mut connection = self.connect(SUN);
        let body = self.fetch(&mut connection, &path, Some("application/json"))?;
        let answer = |what| CdnError::Answer {
            url: format!("https://{SUN}{path}"),
            what,
        };
        let json: serde_json::Value =
            serde_json::from_slice(&body).map_err(|_| answer("not JSON"))?;
        let meta = &json["system_update_metas"][0];
        let id = meta["title_id"]
            .as_str()
            .and_then(|id| u64::from_str_radix(id, 16).ok());
        if id != Some(SYSTEM_UPDATE_ID) {
            return Err(answer("no system update in system_update_metas[0]"));
        }
        meta["title_version"]
            .as_u64()
            .and_then(|version| u32::try_from(version).ok())
            .ok_or(answer("no title_version in system_update_metas[0]"))
    }

    /// Fetches the meta content of version `version` of title `id` into
    /// `dir`, checks it and returns its packaged content meta.
    fn fetch_meta(
        &self,
        connection: &mut Connection,
        keys: &KeySet,
        dir: &Path,
        id: u64,
        version: u32,
    ) -> Result<Vec<u8>, CdnError> {
        let kind = if id == SYSTEM_UPDATE_ID { 's' } else { 'a' };
        let path = format!(
            "/t/{kind}/{id:016x}/{version}?device_id={:016x}",
            self.device_id
        );
        let response = self.request(connection, "HEAD", &path, None)?;
        let content_id = response
            .header("x-nintendo-content-id")
            .and_then(parse_content_id)
            .ok_or_else(|| CdnError::Answer {
                url: format!("https://{ATUMN}{path}"),
                what: "no X-Nintendo-Content-ID",
            })?;
        let content = hex(&content_id);
        let data = self.fetch(connection, &format!("/c/{kind}/{content}"), None)?;
        if Sha256::digest(&data)[..16] != content_id {
            return Err(CdnError::Content {
                content,
                what: "does not hash to its content ID",
            });
        }
        let cnmt = content_meta(&data, keys, &content)?;
        let key = PackagedContentMeta::parse(&cnmt)
            .map_err(|error| CdnError::Cnmt {
                content: content.clone(),
                error,
            })?
            .key();
        if key.id != id
            || key.version != version
            || (key.meta_type == META_TYPE_SYSTEM_UPDATE) != (kind == 's')
        {
            return Err(CdnError::Content {
                content,
                what: "describes another title or version than the one asked for",
            });
        }
        store(&dir.join(format!("{content}.cnmt.nca")), &data)?;
        Ok(cnmt)
    }

    /// Puts the content `info`, whose SHA-256 is `hash`, into `dir`, unless
    /// it is there complete; whether it was fetched.
    fn fetch_content(
        &self,
        connection: &mut Connection,
        dir: &Path,
        hash: &[u8; 32],
        info: &ContentInfo,
    ) -> Result<bool, CdnError> {
        let content = hex(&info.id);
        let target = dir.join(format!("{content}.nca"));
        if complete(&target, info.size, hash)? {
            return Ok(false);
        }
        let path = format!("/c/c/{content}");
        let response = self.request(connection, "GET", &path, None)?;
        let part = partial(&target);
        let file_error = |path: &Path| {
            let path = path.to_owned();
            move |error| CdnError::File { path, error }
        };
        let file = File::create(&part).map_err(file_error(&part))?;
        let mut sink = Hashing {
            inner: BufWriter::new(file),
            hasher: Sha256::new(),
        };
        let size = connection
            .read_body(&response, &mut sink)
            .map_err(|error| CdnError::Request {
                url: format!("https://{ATUMN}{path}"),
                error,
            })?;
        sink.flush().map_err(file_error(&part))?;
        let mismatch = if size != info.size {
            Some("is not of the size its record gives")
        } else if sink.hasher.finalize()[..] != hash[..] {
            Some("does not hash to its record")
        } else {
            None
        };
        if let Some(what) = mismatch {
            let _ = fs::remove_file(&part);
            return Err(CdnError::Content { content, what });
        }
        fs::rename(&part, &target).map_err(file_error(&target))?;
        Ok(true)
    }

    /// Fetches version `version` of the system update into `dir`, which is
    /// made if need be: every title's meta content as `<id>.cnmt.nca` and
    /// every other content as `<id>.nca`, the layout
    /// [`crate::nand::install_firmware`] takes. Contents already in `dir`
    /// complete are kept; `jobs` contents are fetched at a time.
    /// `keys` decrypts the metas.
    pub fn download_system_update(
        &self,
        keys: &KeySet,
        version: u32,
        dir: &Path,
        jobs: NonZeroUsize,
        progress: &(dyn Fn(Progress) + Sync),
    ) -> Result<Downloaded, CdnError> {
        fs::create_dir_all(dir).map_err(|error| CdnError::File {
            path: dir.to_owned(),
            error,
        })?;
        let mut connection = self.connect(ATUMN);
        let update = self.fetch_meta(&mut connection, keys, dir, SYSTEM_UPDATE_ID, version)?;
        progress(Progress::Title {
            id: SYSTEM_UPDATE_ID,
            version,
        });
        let titles: Vec<_> = PackagedContentMeta::parse(&update)
            .expect("checked by fetch_meta")
            .content_meta_infos()
            .collect();
        let mut seen = HashSet::new();
        let mut contents = Vec::new();
        for title in &titles {
            let cnmt = self.fetch_meta(&mut connection, keys, dir, title.id, title.version)?;
            progress(Progress::Title {
                id: title.id,
                version: title.version,
            });
            let meta = PackagedContentMeta::parse(&cnmt).expect("checked by fetch_meta");
            contents.extend(
                meta.packaged_contents()
                    .filter(|(_, info)| seen.insert(info.id)),
            );
        }
        drop(connection);

        let next = AtomicUsize::new(0);
        let fetched_bytes = AtomicU64::new(0);
        let present_bytes = AtomicU64::new(0);
        let failure = Mutex::new(None);
        thread::scope(|scope| {
            for _ in 0..jobs.get().min(contents.len()) {
                scope.spawn(|| {
                    let mut connection = self.connect(ATUMN);
                    while failure.lock().unwrap().is_none() {
                        let Some((hash, info)) = contents.get(next.fetch_add(1, Ordering::Relaxed))
                        else {
                            break;
                        };
                        match self.fetch_content(&mut connection, dir, hash, info) {
                            Ok(fetched) => {
                                let total = if fetched {
                                    &fetched_bytes
                                } else {
                                    &present_bytes
                                };
                                total.fetch_add(info.size, Ordering::Relaxed);
                                progress(Progress::Content {
                                    id: info.id,
                                    size: info.size,
                                    fetched,
                                });
                            }
                            Err(error) => {
                                failure.lock().unwrap().get_or_insert(error);
                            }
                        }
                    }
                });
            }
        });
        if let Some(error) = failure.into_inner().unwrap() {
            return Err(error);
        }
        Ok(Downloaded {
            titles: titles.len() + 1,
            contents: contents.len(),
            fetched_bytes: fetched_bytes.into_inner(),
            present_bytes: present_bytes.into_inner(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_pack_as_nintendo_numbers_them() {
        // 23.5.0 as its system update meta names it.
        assert_eq!(parse_version("23.5.0.640"), Some(0x5C50_0280));
        assert_eq!(version_string(0x5C50_0280), "23.5.0.640");
        assert_eq!(parse_version("0x5c500280"), Some(0x5C50_0280));
        assert_eq!(parse_version("23.5.0"), None);
    }
}
