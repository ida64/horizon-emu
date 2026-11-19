//! The access point the emulated radio finds: one 802.11g network on
//! 2.4 GHz channel 6, open or WPA2-Personal (CCMP), configured with
//! `--wifi-ssid` and `--wifi-passphrase`.
//!
//! The BCM4356's firmware hides the air interface from the host: the host
//! sees scan results (`wl_bss_info_t`), association events and Ethernet
//! frames. This module supplies the access point's side of those: the BSS
//! description the firmware would assemble from its beacons and probe
//! responses, and for WPA2-PSK the authenticator of the 4-way handshake.
//! `wlan`'s supplicant runs the handshake on the host: it has no
//! `sup_wpa` iovar (the firmware supplicant), sets `buf_key_b4_m4` and
//! installs the keys with `wsec_key` itself, so the EAPOL-Key frames travel
//! as data frames (ether type 0x888E) through the message rings. After the
//! handshake the radio encrypts and decrypts with CCMP; the host only ever
//! sees plaintext Ethernet, so after it nothing here depends on the keys.
//!
//! Sources: IEEE 802.11-2016 12.7 (key hierarchy, PRF, EAPOL-Key frames,
//! 4-way handshake), 9.4.2 (information elements); IEEE 802.1X-2004 (EAPOL
//! header); RFC 3394 (AES key wrap); RFC 2898 (PBKDF2); public `wlioctl.h`
//! for the `wl_bss_info_t` layout, which `wlan`'s escan parser (`0x688d0`)
//! confirms (BSSID at 8, SSID at 18/19, chanspec at 72, RSSI at 78, flags
//! at 96).

use aes::{
    Aes128,
    cipher::{BlockEncrypt, KeyInit, generic_array::GenericArray},
};
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha1::Sha1;
use std::io::Read;

/// The access point's MAC address (BSSID). A locally administered address
/// in the emulator's own range (the console's is 02:68:45:6D:75:01, from
/// its calibration data): an assumption, as any address works for the
/// console.
pub const BSSID: [u8; 6] = [0x02, 0x68, 0x45, 0x6d, 0x75, 0xf0];
/// The channel: 6, the middle of the 2.4 GHz band and a common default
/// of home routers (assumed); inside the channels the radio reports valid.
pub const CHANNEL: u8 = 6;
/// Received signal strength and noise floor the radio reports, in dBm: an
/// access point a few metres away in a quiet room (assumed values; nifm
/// turns -45 dBm into full signal bars).
pub const RSSI_DBM: i16 = -45;
pub const NOISE_DBM: i8 = -92;
/// Beacon interval (time units) and DTIM period: the near-universal
/// defaults of access points (assumed).
pub const BEACON_PERIOD: u16 = 100;
pub const DTIM_PERIOD: u8 = 1;
/// 802.11b/g rates in 500 kb/s units; the 802.11b rates are the basic
/// rate set (bit 7), as a mixed b/g access point advertises them.
pub const RATES: [u8; 12] = [
    0x82, 0x84, 0x8b, 0x96, 0x0c, 0x12, 0x18, 0x24, 0x30, 0x48, 0x60, 0x6c,
];
/// Capability information: ESS, short preamble, short slot time, and
/// privacy when the network is secured (IEEE 802.11-2016 9.4.1.4).
pub const CAPABILITY_ESS: u16 = 0x0001;
pub const CAPABILITY_PRIVACY: u16 = 0x0010;
pub const CAPABILITY_SHORT_PREAMBLE: u16 = 0x0020;
pub const CAPABILITY_SHORT_SLOT: u16 = 0x0400;
/// `WL_BSS_INFO_VERSION` and the fixed part of `wl_bss_info_t` (public
/// `wlioctl.h`): the information elements follow at offset 128.
const BSS_INFO_VERSION: u32 = 109;
const BSS_INFO_FIXED_LEN: usize = 128;
/// RSN element of WPA2-Personal with CCMP for both ciphers and no
/// management frame protection (IEEE 802.11-2016 9.4.2.25).
const RSN_IE: [u8; 22] = [
    48, 20, 1, 0, 0x00, 0x0f, 0xac, 4, 1, 0, 0x00, 0x0f, 0xac, 4, 1, 0, 0x00, 0x0f, 0xac, 2, 0, 0,
];
/// Ether type of EAPOL (IEEE 802.1X).
pub const ETHERTYPE_EAPOL: u16 = 0x888e;

/// Key information bits of EAPOL-Key frames (IEEE 802.11-2016 12.7.2).
const KEY_INFO_VERSION_AES_HMAC_SHA1: u16 = 2;
const KEY_INFO_PAIRWISE: u16 = 0x0008;
const KEY_INFO_INSTALL: u16 = 0x0040;
const KEY_INFO_ACK: u16 = 0x0080;
const KEY_INFO_MIC: u16 = 0x0100;
const KEY_INFO_SECURE: u16 = 0x0200;
const KEY_INFO_ENCRYPTED_DATA: u16 = 0x1000;
/// Offsets in an EAPOL-Key frame from its EAPOL header.
const EAPOL_HEADER: usize = 4;
const KEY_INFO_AT: usize = 5;
const REPLAY_AT: usize = 9;
const NONCE_AT: usize = 17;
const MIC_AT: usize = 81;
const KEY_DATA_LEN_AT: usize = 97;
const KEY_DATA_AT: usize = 99;

/// The configured network. Saved with the radio's snapshot: its SSID and
/// key follow from the command line, but the group key is drawn at random
/// when it is built and stations hold it once they joined.
#[derive(Serialize, Deserialize)]
pub struct AccessPoint {
    ssid: Vec<u8>,
    /// The pairwise master key of a WPA2-Personal network.
    pmk: Option<[u8; 32]>,
    /// The group key handed to stations in message 3.
    gtk: [u8; 16],
}

impl AccessPoint {
    /// An open network if `passphrase` is `None`, else WPA2-Personal.
    pub fn new(ssid: &str, passphrase: Option<&str>) -> Result<Self, String> {
        if ssid.is_empty() || ssid.len() > 32 {
            return Err(format!("SSID of {} bytes; 1 to 32 are allowed", ssid.len()));
        }
        let pmk = match passphrase {
            None => None,
            Some(p) if (8..=63).contains(&p.len()) && p.bytes().all(|b| (32..127).contains(&b)) => {
                let mut pmk = [0; 32];
                pbkdf2::pbkdf2_hmac::<Sha1>(p.as_bytes(), ssid.as_bytes(), 4096, &mut pmk);
                Some(pmk)
            }
            Some(_) => {
                return Err("a WPA2 passphrase has 8 to 63 printable ASCII characters".into());
            }
        };
        Ok(Self {
            ssid: ssid.as_bytes().to_vec(),
            pmk,
            gtk: random(),
        })
    }

    pub fn ssid(&self) -> &[u8] {
        &self.ssid
    }

    /// Whether the network is WPA2-Personal.
    pub fn secured(&self) -> bool {
        self.pmk.is_some()
    }

    /// 20 MHz chanspec of the channel (802.11ac encoding, public
    /// `bcmwifi_channels.h`).
    pub fn chanspec(&self) -> u16 {
        0x1000 | u16::from(CHANNEL)
    }

    /// The information elements of its probe responses: SSID, supported
    /// rates, DS parameter set, extended supported rates and, secured, the
    /// RSN element.
    fn ies(&self) -> Vec<u8> {
        let mut ies = vec![0, self.ssid.len() as u8];
        ies.extend(&self.ssid);
        ies.extend([1, 8]);
        ies.extend(&RATES[..8]);
        ies.extend([3, 1, CHANNEL]);
        ies.extend([50, (RATES.len() - 8) as u8]);
        ies.extend(&RATES[8..]);
        if self.secured() {
            ies.extend(RSN_IE);
        }
        ies
    }

    /// `wl_bss_info_t` of the network as the firmware reports it in scan
    /// results and `WLC_GET_BSS_INFO`.
    pub fn bss_info(&self) -> Vec<u8> {
        let mut capability = CAPABILITY_ESS | CAPABILITY_SHORT_PREAMBLE | CAPABILITY_SHORT_SLOT;
        if self.secured() {
            capability |= CAPABILITY_PRIVACY;
        }
        bss_info(&Bss {
            bssid: BSSID,
            beacon_period: BEACON_PERIOD,
            capability,
            ssid: &self.ssid,
            channel: CHANNEL,
            dtim_period: DTIM_PERIOD,
            rssi: RSSI_DBM,
            ies: &self.ies(),
        })
    }

    /// Whether a station whose own RSN/WPA element (from `wpaie`) is
    /// `station_ie` and whose `wpa_auth` is `wpa_auth` matches the network's
    /// security: none for an open network; WPA2-PSK (`WPA2_AUTH_PSK`, 0x80,
    /// public `wlioctl_defs.h`) with an RSN element naming CCMP and PSK for
    /// a secured one. A mismatched station is refused at association, as
    /// the access point would.
    pub fn accepts(&self, wpa_auth: u32, station_ie: &[u8]) -> bool {
        if !self.secured() {
            return wpa_auth == 0;
        }
        wpa_auth & 0x80 != 0
            && station_ie.first() == Some(&48)
            && station_ie.windows(4).any(|w| w == [0x00, 0x0f, 0xac, 2])
            && station_ie.windows(4).any(|w| w == [0x00, 0x0f, 0xac, 4])
    }

    /// The authenticator for a newly associated station; `None` on an open
    /// network, where the port opens at association.
    pub fn authenticator(&self, station: [u8; 6]) -> Option<Authenticator> {
        self.pmk.map(|pmk| Authenticator {
            pmk,
            gtk: self.gtk,
            station,
            anonce: random(),
            replay: 0,
            ptk: None,
            done: false,
        })
    }

    /// Takes the state of `saved`, the same network in the machine a
    /// snapshot was taken of: its group key. Refused for a network
    /// configured differently (another SSID or passphrase), which the
    /// saved stations could not be associated with.
    pub fn restore(&mut self, saved: AccessPoint) -> Result<(), String> {
        if saved.ssid != self.ssid || saved.pmk != self.pmk {
            return Err("the access point was configured with another SSID or passphrase".into());
        }
        self.gtk = saved.gtk;
        Ok(())
    }
}

/// What the radio knows of a network it heard: from the beacon or probe
/// response, and the signal it heard it with.
pub struct Bss<'a> {
    pub bssid: [u8; 6],
    pub beacon_period: u16,
    pub capability: u16,
    /// Empty for a network that hides its name.
    pub ssid: &'a [u8],
    pub channel: u8,
    pub dtim_period: u8,
    pub rssi: i16,
    /// The information elements of the frame it was heard in.
    pub ies: &'a [u8],
}

/// `wl_bss_info_t` (version 109) of `bss`, little-endian, length rounded
/// up to a multiple of 4: the fixed part, then the information elements.
/// The rate set is the one the elements carry (supported and extended
/// supported rates), the chanspec a 20 MHz one.
pub fn bss_info(bss: &Bss) -> Vec<u8> {
    let mut rates: Vec<u8> = [1, 50]
        .iter()
        .filter_map(|&id| super::dot11::element(bss.ies, id))
        .flatten()
        .copied()
        .collect();
    rates.truncate(16);
    let mut info = vec![0u8; BSS_INFO_FIXED_LEN];
    let length = (BSS_INFO_FIXED_LEN + bss.ies.len()).next_multiple_of(4);
    let ssid = &bss.ssid[..bss.ssid.len().min(32)];
    let band = if bss.channel > 14 { 0xc000 } else { 0 };
    let chanspec = 0x1000 | band | u16::from(bss.channel);
    info[0..4].copy_from_slice(&BSS_INFO_VERSION.to_le_bytes());
    info[4..8].copy_from_slice(&(length as u32).to_le_bytes());
    info[8..14].copy_from_slice(&bss.bssid);
    info[14..16].copy_from_slice(&bss.beacon_period.to_le_bytes());
    info[16..18].copy_from_slice(&bss.capability.to_le_bytes());
    info[18] = ssid.len() as u8;
    info[19..19 + ssid.len()].copy_from_slice(ssid);
    info[52..56].copy_from_slice(&(rates.len() as u32).to_le_bytes());
    info[56..56 + rates.len()].copy_from_slice(&rates);
    info[72..74].copy_from_slice(&chanspec.to_le_bytes());
    info[76] = bss.dtim_period;
    info[78..80].copy_from_slice(&bss.rssi.to_le_bytes());
    info[80] = NOISE_DBM as u8;
    info[88] = bss.channel;
    info[116..118].copy_from_slice(&(BSS_INFO_FIXED_LEN as u16).to_le_bytes());
    info[120..124].copy_from_slice(&(bss.ies.len() as u32).to_le_bytes());
    info[124..126].copy_from_slice(&(bss.rssi - i16::from(NOISE_DBM)).to_le_bytes());
    info.extend(bss.ies);
    info.resize(length, 0);
    info
}

/// The access point's half of the 4-way handshake with one station: all
/// of it is saved with the association it belongs to.
#[derive(Serialize, Deserialize)]
pub struct Authenticator {
    pmk: [u8; 32],
    gtk: [u8; 16],
    station: [u8; 6],
    anonce: [u8; 32],
    replay: u64,
    /// KCK, KEK and TK once message 2 arrived.
    ptk: Option<Ptk>,
    done: bool,
}

/// A pairwise transient key: KCK, KEK and TK.
#[derive(Clone, Copy, Serialize, Deserialize)]
struct Ptk(#[serde(with = "crate::snapshot::array")] [u8; 48]);

impl Authenticator {
    /// Whether the handshake completed: data frames may flow.
    pub fn authorized(&self) -> bool {
        self.done
    }

    /// Message 1 (or its retransmission, with the next replay counter) as
    /// an Ethernet frame to the station; `None` once message 2 arrived.
    pub fn message_1(&mut self) -> Option<Vec<u8>> {
        if self.ptk.is_some() {
            return None;
        }
        self.replay += 1;
        let info = KEY_INFO_VERSION_AES_HMAC_SHA1 | KEY_INFO_PAIRWISE | KEY_INFO_ACK;
        let anonce = self.anonce;
        Some(self.frame(&key_frame(info, self.replay, &anonce, &[])))
    }

    /// An EAPOL frame (Ethernet) from the station: message 2 or 4. Returns
    /// message 3 after a valid message 2. Frames that do not belong to the
    /// handshake in progress (wrong replay counter, bad MIC) are dropped,
    /// as an authenticator does.
    pub fn receive(&mut self, frame: &[u8]) -> Option<Vec<u8>> {
        let eapol = frame.get(14..)?;
        if eapol.get(1) != Some(&3) || eapol.len() < KEY_DATA_AT {
            return None;
        }
        let length = usize::from(u16::from_be_bytes([eapol[2], eapol[3]]));
        let eapol = eapol.get(..EAPOL_HEADER + length)?;
        let info = u16::from_be_bytes([eapol[KEY_INFO_AT], eapol[KEY_INFO_AT + 1]]);
        let replay = u64::from_be_bytes(eapol[REPLAY_AT..REPLAY_AT + 8].try_into().unwrap());
        if info & KEY_INFO_MIC == 0 || info & KEY_INFO_ACK != 0 || replay != self.replay {
            return None;
        }
        if info & KEY_INFO_SECURE == 0 {
            // Message 2: derive the PTK from its SNonce and check the MIC.
            if self.done {
                return None;
            }
            let snonce: [u8; 32] = eapol[NONCE_AT..NONCE_AT + 32].try_into().unwrap();
            let ptk = self.derive_ptk(&snonce);
            if !mic_valid(&ptk[..16], eapol) {
                return None;
            }
            self.ptk = Some(Ptk(ptk));
            Some(self.message_3())
        } else {
            // Message 4.
            let ptk = self.ptk?.0;
            if !mic_valid(&ptk[..16], eapol) {
                return None;
            }
            self.done = true;
            None
        }
    }

    /// Message 3: install, secure, with the RSN element and the GTK in a
    /// GTK KDE (key ID 1), key-wrapped with the KEK.
    fn message_3(&mut self) -> Vec<u8> {
        let ptk = self.ptk.unwrap().0;
        self.replay += 1;
        let mut data = RSN_IE.to_vec();
        data.extend([0xdd, 22, 0x00, 0x0f, 0xac, 1, 1, 0]);
        data.extend(self.gtk);
        // Padding: 0xdd then zeros up to a multiple of 8 (12.7.2).
        if !data.len().is_multiple_of(8) {
            data.push(0xdd);
            data.resize(data.len().next_multiple_of(8), 0);
        }
        let wrapped = aes_key_wrap(ptk[16..32].try_into().unwrap(), &data);
        let info = KEY_INFO_VERSION_AES_HMAC_SHA1
            | KEY_INFO_PAIRWISE
            | KEY_INFO_INSTALL
            | KEY_INFO_ACK
            | KEY_INFO_MIC
            | KEY_INFO_SECURE
            | KEY_INFO_ENCRYPTED_DATA;
        let anonce = self.anonce;
        let mut eapol = key_frame(info, self.replay, &anonce, &wrapped);
        let mic = hmac_sha1(&ptk[..16], &eapol);
        eapol[MIC_AT..MIC_AT + 16].copy_from_slice(&mic[..16]);
        self.frame(&eapol)
    }

    /// PTK = PRF-384(PMK, "Pairwise key expansion", min(AA,SPA) ||
    /// max(AA,SPA) || min(ANonce,SNonce) || max(ANonce,SNonce)).
    fn derive_ptk(&self, snonce: &[u8; 32]) -> [u8; 48] {
        let (a1, a2) = order(&BSSID, &self.station);
        let (n1, n2) = order(&self.anonce, snonce);
        let mut data = Vec::with_capacity(76);
        data.extend(a1);
        data.extend(a2);
        data.extend(n1);
        data.extend(n2);
        let mut ptk = [0; 48];
        prf(&self.pmk, b"Pairwise key expansion", &data, &mut ptk);
        ptk
    }

    fn frame(&self, eapol: &[u8]) -> Vec<u8> {
        let mut frame = Vec::with_capacity(14 + eapol.len());
        frame.extend(self.station);
        frame.extend(BSSID);
        frame.extend(ETHERTYPE_EAPOL.to_be_bytes());
        frame.extend(eapol);
        frame
    }
}

/// An EAPOL-Key frame (EAPOL version 2, RSN descriptor, key length 16)
/// with a zero MIC.
fn key_frame(info: u16, replay: u64, nonce: &[u8; 32], data: &[u8]) -> Vec<u8> {
    let body = 95 + data.len();
    let mut eapol = vec![0u8; EAPOL_HEADER + body];
    eapol[0] = 2;
    eapol[1] = 3;
    eapol[2..4].copy_from_slice(&(body as u16).to_be_bytes());
    eapol[4] = 2;
    eapol[KEY_INFO_AT..KEY_INFO_AT + 2].copy_from_slice(&info.to_be_bytes());
    eapol[7..9].copy_from_slice(&16u16.to_be_bytes());
    eapol[REPLAY_AT..REPLAY_AT + 8].copy_from_slice(&replay.to_be_bytes());
    eapol[NONCE_AT..NONCE_AT + 32].copy_from_slice(nonce);
    eapol[KEY_DATA_LEN_AT..KEY_DATA_LEN_AT + 2].copy_from_slice(&(data.len() as u16).to_be_bytes());
    eapol[KEY_DATA_AT..].copy_from_slice(data);
    eapol
}

fn mic_valid(kck: &[u8], eapol: &[u8]) -> bool {
    let mut zeroed = eapol.to_vec();
    zeroed[MIC_AT..MIC_AT + 16].fill(0);
    hmac_sha1(kck, &zeroed)[..16] == eapol[MIC_AT..MIC_AT + 16]
}

fn hmac_sha1(key: &[u8], data: &[u8]) -> [u8; 20] {
    let mut mac = <Hmac<Sha1> as Mac>::new_from_slice(key).expect("HMAC takes any key length");
    mac.update(data);
    mac.finalize().into_bytes().into()
}

/// The 802.11 PRF: HMAC-SHA1(K, A || 0 || B || i) for i = 0, 1, ...
fn prf(key: &[u8], label: &[u8], data: &[u8], out: &mut [u8]) {
    for (i, chunk) in out.chunks_mut(20).enumerate() {
        let mut input = label.to_vec();
        input.push(0);
        input.extend(data);
        input.push(i as u8);
        chunk.copy_from_slice(&hmac_sha1(key, &input)[..chunk.len()]);
    }
}

fn order<'a>(a: &'a [u8], b: &'a [u8]) -> (&'a [u8], &'a [u8]) {
    if a <= b { (a, b) } else { (b, a) }
}

/// AES key wrap (RFC 3394) with the default IV.
fn aes_key_wrap(kek: &[u8; 16], plain: &[u8]) -> Vec<u8> {
    let cipher = Aes128::new(GenericArray::from_slice(kek));
    let n = plain.len() / 8;
    let mut a = [0xa6u8; 8];
    let mut r: Vec<[u8; 8]> = plain.chunks(8).map(|c| c.try_into().unwrap()).collect();
    for j in 0..6 {
        for (i, block) in r.iter_mut().enumerate() {
            let mut b = [0u8; 16];
            b[..8].copy_from_slice(&a);
            b[8..].copy_from_slice(block);
            let mut b = GenericArray::from(b);
            cipher.encrypt_block(&mut b);
            let t = (n * j + i + 1) as u64;
            a.copy_from_slice(&b[..8]);
            for (x, y) in a.iter_mut().zip(t.to_be_bytes()) {
                *x ^= y;
            }
            block.copy_from_slice(&b[8..]);
        }
    }
    let mut out = a.to_vec();
    for block in r {
        out.extend(block);
    }
    out
}

/// Random bytes from the host's `/dev/urandom` (nonces and the group key
/// must be unpredictable; this is the host's own CSPRNG).
fn random<const N: usize>() -> [u8; N] {
    let mut bytes = [0; N];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut bytes))
        .expect("the host provides /dev/urandom");
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    /// IEEE 802.11-2016 J.4.2 test vector: PSK from passphrase and SSID.
    #[test]
    fn psk_from_passphrase() {
        let ap = AccessPoint::new("IEEE", Some("password")).unwrap();
        assert_eq!(
            ap.pmk.unwrap()[..8],
            [0xf4, 0x2c, 0x6f, 0xc5, 0x2d, 0xf0, 0xeb, 0xef]
        );
    }

    /// RFC 3394 4.1 test vector.
    #[test]
    fn key_wrap() {
        let kek: [u8; 16] = core::array::from_fn(|i| i as u8);
        let plain: Vec<u8> = (0..16).map(|i| (i * 0x11) as u8).collect();
        assert_eq!(
            aes_key_wrap(&kek, &plain),
            [
                0x1f, 0xa6, 0x8b, 0x0a, 0x81, 0x12, 0xb4, 0x47, 0xae, 0xf3, 0x4b, 0xd8, 0xfb, 0x5a,
                0x7b, 0x82, 0x9d, 0x3e, 0x86, 0x23, 0x71, 0xd2, 0xcf, 0xe5
            ]
        );
    }

    #[test]
    fn bss_info_layout() {
        let ap = AccessPoint::new("emulated", Some("passphrase")).unwrap();
        let info = ap.bss_info();
        assert_eq!(
            u32::from_le_bytes(info[4..8].try_into().unwrap()) as usize,
            info.len()
        );
        assert_eq!(&info[8..14], &BSSID);
        assert_eq!(&info[19..27], b"emulated");
        assert_eq!(u16::from_le_bytes([info[72], info[73]]), 0x1006);
        assert_ne!(
            u16::from_le_bytes([info[16], info[17]]) & CAPABILITY_PRIVACY,
            0
        );
        let ies = &info[128..128 + u32::from_le_bytes(info[120..124].try_into().unwrap()) as usize];
        assert!(ies.windows(RSN_IE.len()).any(|w| w == RSN_IE));
    }

    /// A station's side of the handshake against the authenticator.
    #[test]
    fn four_way_handshake() {
        let ap = AccessPoint::new("emulated", Some("passphrase")).unwrap();
        let station = [0x02, 0x68, 0x45, 0x6d, 0x75, 0x01];
        let mut auth = ap.authenticator(station).unwrap();
        let m1 = auth.message_1().unwrap();
        assert_eq!(&m1[12..14], &[0x88, 0x8e]);
        let anonce: [u8; 32] = m1[14 + NONCE_AT..14 + NONCE_AT + 32].try_into().unwrap();
        let replay = u64::from_be_bytes(m1[14 + REPLAY_AT..14 + REPLAY_AT + 8].try_into().unwrap());
        // Message 2 with the station's SNonce and RSN element.
        let snonce = [7u8; 32];
        let ptk = auth.derive_ptk(&snonce);
        let mut m2 = key_frame(
            KEY_INFO_VERSION_AES_HMAC_SHA1 | KEY_INFO_PAIRWISE | KEY_INFO_MIC,
            replay,
            &snonce,
            &RSN_IE,
        );
        let mic = hmac_sha1(&ptk[..16], &m2);
        m2[MIC_AT..MIC_AT + 16].copy_from_slice(&mic[..16]);
        let mut frame = BSSID.to_vec();
        frame.extend(station);
        frame.extend(ETHERTYPE_EAPOL.to_be_bytes());
        frame.extend(&m2);
        // A corrupted MIC is ignored.
        let mut bad = frame.clone();
        bad[14 + MIC_AT] ^= 1;
        assert!(auth.receive(&bad).is_none());
        let m3 = auth.receive(&frame).unwrap();
        let eapol = &m3[14..];
        assert!(mic_valid(&ptk[..16], eapol));
        assert_eq!(&eapol[NONCE_AT..NONCE_AT + 32], &anonce);
        let replay = u64::from_be_bytes(eapol[REPLAY_AT..REPLAY_AT + 8].try_into().unwrap());
        assert_eq!(replay, 2);
        assert!(!auth.authorized());
        // Message 4.
        let mut m4 = key_frame(
            KEY_INFO_VERSION_AES_HMAC_SHA1 | KEY_INFO_PAIRWISE | KEY_INFO_MIC | KEY_INFO_SECURE,
            replay,
            &[0; 32],
            &[],
        );
        let mic = hmac_sha1(&ptk[..16], &m4);
        m4[MIC_AT..MIC_AT + 16].copy_from_slice(&mic[..16]);
        let mut frame = BSSID.to_vec();
        frame.extend(station);
        frame.extend(ETHERTYPE_EAPOL.to_be_bytes());
        frame.extend(&m4);
        assert!(auth.receive(&frame).is_none());
        assert!(auth.authorized());
        assert_eq!(anonce, auth.anonce);
    }
}
