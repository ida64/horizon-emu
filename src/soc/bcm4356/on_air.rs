//! What the BCM4356's firmware does on the air shared with other emulated
//! radios ([`crate::air`]): the 802.11 MAC (MLME) a full-dongle firmware
//! keeps from its host. The host sees scan results, events and Ethernet
//! frames; here are the frames behind them:
//!
//! * as access point (`WLC_SET_AP` 1, then `WLC_SET_SSID`): beacons every
//!   beacon period carrying what the host configured (SSID, or none for a
//!   closed network, channel, privacy, the vendor elements it added with
//!   `vndr_ie`), probe responses, open-system authentication and
//!   association of stations (up to `maxassoc`), each reported to the host
//!   (`WLC_E_AUTH_IND`, `WLC_E_ASSOC_IND` with the request's elements, and
//!   `WLC_E_DEAUTH_IND`/`WLC_E_DISASSOC_IND` when a station leaves);
//!   Ethernet frames to and from the stations, forwarded between stations
//!   inside the BSS unless `ap_isolate` is set;
//! * as station: escan results for networks heard on the air, the join
//!   (directed probe where the network is not known yet, authentication,
//!   association) ending in the events the infrastructure path reports,
//!   and data frames through the access point;
//! * on both: CCMP with the keys the host installs with `wsec_key` when
//!   `wsec` enables AES, packet numbers counted per key.
//!
//! Frame formats are IEEE 802.11-2016 (9.3.3 management frame bodies);
//! firmware behaviour follows the public brcmfmac driver's view of the
//! firmware (events and their data, `cfg80211.c`) where the host can see
//! it; what only the firmware decides is marked as an assumption.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::super::{
    dot11::{self, Ccmp, Header},
    wifi_ap,
};

/// Time unit of beacon intervals: 1024 microseconds (9.4.1.3).
pub const TU: Duration = Duration::from_micros(1024);
/// Status codes (9.4.1.9): success, and "the AP is unable to handle
/// additional associated STAs".
pub const STATUS_SUCCESS: u16 = 0;
pub const STATUS_TOO_MANY_STATIONS: u16 = 17;
/// Reason code of a station leaving (9.4.1.7): "deauthenticated because
/// sending STA is leaving".
pub const REASON_LEAVING: u16 = 3;
/// How long the station waits for each answer of authentication and
/// association before it gives up, and how long it dwells on a channel
/// probing for a network it does not know yet: the firmware's defaults
/// (assumed; Broadcom's `assoc_retry_max`/`scan_assoc_time` defaults are
/// of this order).
pub const ANSWER_TIMEOUT: Duration = Duration::from_millis(300);
pub const PROBE_DWELL: Duration = Duration::from_millis(40);
/// Signal strength reported for other emulators' radios: consoles side
/// by side (assumed, as for the access point of `--wifi-ssid`).
pub const RSSI_DBM: i16 = -40;
/// `wsec` bit of AES (public `wlioctl_defs.h` `AES_ENABLED`), and
/// `wl_wsec_key_t.algo` of CCMP (`CRYPTO_ALGO_AES_CCM`) and flag of the
/// key used to transmit (`WL_PRIMARY_KEY`).
pub const AES_ENABLED: u32 = 4;
pub const CRYPTO_ALGO_AES_CCM: u32 = 4;
const WL_PRIMARY_KEY: u32 = 2;
/// `vndr_ie` packet flags: in which frames an element goes (public
/// `wlioctl.h` `VNDR_IE_*_FLAG`).
pub const VNDR_IE_BEACON: u32 = 0x01;
pub const VNDR_IE_PROBE_RESPONSE: u32 = 0x02;
pub const VNDR_IE_ASSOC_RESPONSE: u32 = 0x04;
pub const VNDR_IE_PROBE_REQUEST: u32 = 0x10;
pub const VNDR_IE_ASSOC_REQUEST: u32 = 0x20;
/// Events (public `wlioctl_defs.h`).
pub const WLC_E_SET_SSID: u32 = 0;
pub const WLC_E_JOIN: u32 = 1;
pub const WLC_E_AUTH: u32 = 3;
pub const WLC_E_AUTH_IND: u32 = 4;
pub const WLC_E_DEAUTH: u32 = 5;
pub const WLC_E_DEAUTH_IND: u32 = 6;
pub const WLC_E_ASSOC: u32 = 7;
pub const WLC_E_ASSOC_IND: u32 = 8;
pub const WLC_E_REASSOC_IND: u32 = 10;
pub const WLC_E_DISASSOC_IND: u32 = 12;
pub const WLC_E_LINK: u32 = 16;
pub const WLC_E_STATUS_SUCCESS: u32 = 0;
pub const WLC_E_STATUS_FAIL: u32 = 1;
pub const WLC_E_STATUS_NO_NETWORKS: u32 = 3;
pub const WLC_EVENT_MSG_LINK: u16 = 1;

/// What the host configured that the frames depend on, read from the
/// firmware's settings when they are needed.
pub struct Config {
    pub mac: [u8; 6],
    pub wsec: u32,
    /// The RSN or WPA element the host gave (`wpaie`), for association
    /// requests and, as access point, beacons.
    pub wpa_ie: Vec<u8>,
    pub closednet: bool,
    pub max_assoc: u32,
    pub beacon_period: u16,
    pub dtim_period: u8,
    pub ap_isolate: bool,
}

/// A key the host installed.
#[derive(Clone, Serialize, Deserialize)]
pub struct Key {
    pub index: u32,
    /// The peer of a pairwise key; `None` for a default (group) key.
    pub peer: Option<[u8; 6]>,
    pub data: [u8; 16],
    pub primary: bool,
    /// The packet number of the next frame sent with it.
    pub next_pn: u64,
}

/// What `wsec_key` asks: `wl_wsec_key_t` (public `wlioctl.h`: index at 0,
/// length at 4, key data at 8, algorithm at 112, flags at 116, the peer's
/// address at 156; zero for a default key).
pub enum KeyChange {
    Install(Key),
    Remove { index: u32, peer: Option<[u8; 6]> },
}

pub fn key_change(value: &[u8]) -> Result<KeyChange, String> {
    let word = |at: usize| {
        value
            .get(at..at + 4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
    };
    let (Some(index), Some(length), Some(algorithm), Some(flags), Some(peer)) =
        (word(0), word(4), word(112), word(116), value.get(156..162))
    else {
        return Err(format!("wsec_key of {} bytes", value.len()));
    };
    let peer: [u8; 6] = peer.try_into().unwrap();
    let peer = (peer != [0; 6]).then_some(peer);
    match (length, algorithm) {
        (0, _) => Ok(KeyChange::Remove { index, peer }),
        (16, CRYPTO_ALGO_AES_CCM) => Ok(KeyChange::Install(Key {
            index,
            peer,
            data: value[8..24].try_into().unwrap(),
            primary: flags & WL_PRIMARY_KEY != 0,
            next_pn: 1,
        })),
        _ => Err(format!(
            "wsec_key with algorithm {algorithm}, {length}-byte key, flags {flags:#x}"
        )),
    }
}

/// A station of the BSS this radio runs.
#[derive(Clone, Serialize, Deserialize)]
pub struct Peer {
    pub mac: [u8; 6],
    pub aid: u16,
    pub associated: bool,
}

/// The BSS this radio runs as access point.
#[derive(Serialize, Deserialize)]
pub struct OwnBss {
    pub if_id: u8,
    pub ssid: Vec<u8>,
    pub channel: u8,
    pub stations: Vec<Peer>,
    /// Counts BSS starts, to tell stale beacon work from current.
    pub epoch: u32,
}

/// How far a join over the air got.
#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Stage {
    /// Probing for the network on `channels[index]`.
    Probing(usize),
    Authenticating,
    Associating,
    Associated,
}

/// The station's association with an access point on the air.
#[derive(Serialize, Deserialize)]
pub struct Link {
    pub if_id: u8,
    pub epoch: u32,
    pub ssid: Vec<u8>,
    /// Known once the network answered (or the host named it).
    pub bssid: Option<[u8; 6]>,
    pub channel: u8,
    pub channels: Vec<u8>,
    pub stage: Stage,
    /// `wl_bss_info_t` of the network, from what it last sent.
    pub bss_info: Vec<u8>,
}

/// A scan in flight, as far as the air is concerned: the channel it is on.
#[derive(Serialize, Deserialize)]
pub struct Scan {
    pub sync_id: u16,
    pub if_id: u8,
    pub channel: Option<u8>,
    /// The SSID asked for; empty for any.
    pub ssid: Vec<u8>,
    pub passive: bool,
    /// The networks already reported in this scan.
    pub reported: Vec<[u8; 6]>,
}

/// A network heard on the air.
#[derive(Clone, Serialize, Deserialize)]
pub struct Heard {
    pub bssid: [u8; 6],
    pub channel: u8,
    /// Empty for a hidden network until a probe response named it.
    pub ssid: Vec<u8>,
    pub bss_info: Vec<u8>,
}

/// Work on the air that comes due later.
#[derive(Serialize, Deserialize)]
pub enum Work {
    Beacon {
        epoch: u32,
    },
    ScanChannel {
        sync_id: u16,
        channel: u8,
    },
    Probe {
        epoch: u32,
        index: usize,
    },
    Timeout {
        epoch: u32,
        stage: Stage,
    },
    /// The next of the action frames the host asked to be sent
    /// periodically (`nsc_aloe_af_periodic`, with the setting as it was
    /// asked: a changed or withdrawn setting ends the old series).
    PeriodicActionFrame {
        setting: Vec<u8>,
    },
}

/// What the radio does as the result of a request or a frame.
pub enum Effect {
    Transmit {
        channel: u8,
        frame: Vec<u8>,
    },
    Event {
        if_id: u8,
        kind: u32,
        status: u32,
        reason: u32,
        flags: u16,
        addr: [u8; 6],
        data: Vec<u8>,
    },
    /// A partial escan result with this BSS.
    ScanResult {
        if_id: u8,
        sync_id: u16,
        bss_info: Vec<u8>,
    },
    /// An Ethernet frame for the host.
    Deliver {
        if_id: u8,
        frame: Vec<u8>,
    },
    /// An action frame heard, for the host (as an event, if its filters
    /// let it through).
    ActionFrame {
        channel: u8,
        frame: Vec<u8>,
    },
    Later {
        delay: Duration,
        work: Work,
    },
}

fn event(if_id: u8, kind: u32, status: u32, addr: [u8; 6], data: Vec<u8>) -> Effect {
    Effect::Event {
        if_id,
        kind,
        status,
        reason: 0,
        flags: 0,
        addr,
        data,
    }
}

/// The radio's part on the air: saved with the firmware's state.
#[derive(Default, Serialize, Deserialize)]
pub struct Radio {
    pub bss: Option<OwnBss>,
    pub link: Option<Link>,
    pub scan: Option<Scan>,
    pub heard: Vec<Heard>,
    pub keys: Vec<Key>,
    /// Vendor elements the host added, with the frames they go in.
    pub vendor_ies: Vec<(u32, Vec<u8>)>,
    pub sequence: u16,
    pub epoch: u32,
}

/// Supported and extended supported rates elements: 802.11b/g in 2.4 GHz,
/// OFDM only in 5 GHz (basic rates 6, 12, 24 Mb/s).
fn rate_elements(channel: u8) -> Vec<u8> {
    let mut ies = Vec::new();
    if channel > 14 {
        ies.extend([1, 8, 0x8c, 0x12, 0x98, 0x24, 0xb0, 0x48, 0x60, 0x6c]);
    } else {
        ies.extend([1, 8]);
        ies.extend(&wifi_ap::RATES[..8]);
        ies.push(50);
        ies.push((wifi_ap::RATES.len() - 8) as u8);
        ies.extend(&wifi_ap::RATES[8..]);
    }
    ies
}

fn ssid_element(ssid: &[u8]) -> Vec<u8> {
    let mut ie = vec![0, ssid.len() as u8];
    ie.extend(ssid);
    ie
}

impl Radio {
    fn next_sequence(&mut self) -> u16 {
        self.sequence = (self.sequence + 1) & 0xfff;
        self.sequence << 4
    }

    fn frame(&mut self, mut header: Header, body: &[u8]) -> Vec<u8> {
        header.sequence = self.next_sequence();
        header.frame(body)
    }

    fn vendor(&self, flag: u32) -> Vec<u8> {
        self.vendor_ies
            .iter()
            .filter(|(flags, _)| flags & flag != 0)
            .flat_map(|(_, ie)| ie.iter().copied())
            .collect()
    }

    /// The channel the radio listens on: its BSS's, the network it joins,
    /// the one a scan dwells on, else none (it hears nothing addressed to
    /// it, and nothing it would act on).
    pub fn channel(&self) -> Option<u8> {
        if let Some(bss) = &self.bss {
            return Some(bss.channel);
        }
        if let Some(link) = &self.link {
            return Some(match link.stage {
                Stage::Probing(index) => link.channels.get(index).copied()?,
                _ => link.channel,
            });
        }
        self.scan.as_ref().and_then(|s| s.channel)
    }

    // ----- access point -----

    fn capability(config: &Config) -> u16 {
        let mut capability = wifi_ap::CAPABILITY_ESS
            | wifi_ap::CAPABILITY_SHORT_PREAMBLE
            | wifi_ap::CAPABILITY_SHORT_SLOT;
        if config.wsec != 0 {
            capability |= wifi_ap::CAPABILITY_PRIVACY;
        }
        capability
    }

    /// The body of a beacon (`beacon`) or probe response: timestamp, beacon
    /// interval, capability, then SSID (none in beacons of a closed
    /// network: \[assumption\] a zero-length SSID element, as Broadcom's
    /// `closednet` is publicly described), rates, DS parameter set, TIM
    /// (beacons), the host's RSN element and vendor elements.
    fn bss_body(&self, config: &Config, bss: &OwnBss, now: Duration, beacon: bool) -> Vec<u8> {
        let mut body = (now.as_micros() as u64).to_le_bytes().to_vec();
        body.extend(config.beacon_period.to_le_bytes());
        body.extend(Self::capability(config).to_le_bytes());
        let hidden = beacon && config.closednet;
        body.extend(ssid_element(if hidden { &[] } else { &bss.ssid }));
        let rates = rate_elements(bss.channel);
        let (supported, extended) = rates.split_at(2 + usize::from(rates[1]));
        body.extend(supported);
        body.extend([3, 1, bss.channel]);
        if beacon {
            body.extend([5, 4, 0, config.dtim_period.max(1), 0, 0]);
        }
        body.extend(extended);
        if config.wpa_ie.first() == Some(&48) {
            body.extend(&config.wpa_ie);
        }
        let flag = if beacon {
            VNDR_IE_BEACON
        } else {
            VNDR_IE_PROBE_RESPONSE
        };
        body.extend(self.vendor(flag));
        body
    }

    /// The host started a BSS with `ssid` on `channel`: beaconing begins at
    /// once; the host hears that its link is up and that the SSID is set
    /// (\[assumption\]: the events brcmfmac takes as the access point
    /// being up, `brcmf_notify_connect_status_ap`).
    pub fn start_bss(
        &mut self,
        config: &Config,
        if_id: u8,
        ssid: &[u8],
        channel: u8,
    ) -> Vec<Effect> {
        self.epoch += 1;
        self.link = None;
        self.bss = Some(OwnBss {
            if_id,
            ssid: ssid.to_vec(),
            channel,
            stations: Vec::new(),
            epoch: self.epoch,
        });
        let mut link = event(
            if_id,
            WLC_E_LINK,
            WLC_E_STATUS_SUCCESS,
            config.mac,
            Vec::new(),
        );
        if let Effect::Event { flags, .. } = &mut link {
            *flags = WLC_EVENT_MSG_LINK;
        }
        vec![
            link,
            event(
                if_id,
                WLC_E_SET_SSID,
                WLC_E_STATUS_SUCCESS,
                config.mac,
                ssid.to_vec(),
            ),
            Effect::Later {
                delay: Duration::ZERO,
                work: Work::Beacon { epoch: self.epoch },
            },
        ]
    }

    /// The host took the BSS down: its stations are deauthenticated (they
    /// hear a deauthentication), beaconing stops.
    pub fn stop_bss(&mut self, config: &Config) -> Vec<Effect> {
        let Some(bss) = self.bss.take() else {
            return Vec::new();
        };
        let mut effects = Vec::new();
        for station in bss.stations {
            let header = Header::new(dot11::DEAUTH, 0, station.mac, config.mac, config.mac);
            let frame = self.frame(header, &REASON_LEAVING.to_le_bytes());
            effects.push(Effect::Transmit {
                channel: bss.channel,
                frame,
            });
        }
        self.keys.retain(|k| k.peer.is_none());
        effects
    }

    /// Deauthenticates station `mac` for `reason` at the host's request.
    pub fn deauthenticate(&mut self, config: &Config, mac: [u8; 6], reason: u16) -> Vec<Effect> {
        let Some(bss) = self.bss.as_mut() else {
            return Vec::new();
        };
        let (if_id, channel) = (bss.if_id, bss.channel);
        let before = bss.stations.len();
        bss.stations.retain(|s| s.mac != mac);
        if bss.stations.len() == before {
            return Vec::new();
        }
        self.keys.retain(|k| k.peer != Some(mac));
        let header = Header::new(dot11::DEAUTH, 0, mac, config.mac, config.mac);
        let frame = self.frame(header, &reason.to_le_bytes());
        vec![
            Effect::Transmit { channel, frame },
            event(if_id, WLC_E_DEAUTH, WLC_E_STATUS_SUCCESS, mac, Vec::new()),
        ]
    }

    pub fn beacon(&mut self, config: &Config, epoch: u32, now: Duration) -> Vec<Effect> {
        let Some(bss) = self.bss.as_ref().filter(|b| b.epoch == epoch) else {
            return Vec::new();
        };
        let body = self.bss_body(config, bss, now, true);
        let channel = bss.channel;
        let header = Header::new(dot11::BEACON, 0, dot11::BROADCAST, config.mac, config.mac);
        let frame = self.frame(header, &body);
        vec![
            Effect::Transmit { channel, frame },
            Effect::Later {
                delay: TU * u32::from(config.beacon_period.max(1)),
                work: Work::Beacon { epoch },
            },
        ]
    }

    /// A management frame for the BSS this radio runs.
    fn ap_management(
        &mut self,
        config: &Config,
        header: &Header,
        body: &[u8],
        now: Duration,
    ) -> Vec<Effect> {
        let Some(bss) = self.bss.as_ref() else {
            return Vec::new();
        };
        let (if_id, channel) = (bss.if_id, bss.channel);
        let from = header.addr2;
        let reply = |radio: &mut Radio, kind: u8, body: &[u8]| Effect::Transmit {
            channel,
            frame: radio.frame(Header::new(kind, 0, from, config.mac, config.mac), body),
        };
        match header.kind {
            // Answered when it asks for any network (and this one does not
            // hide) or names this one.
            dot11::PROBE_REQUEST => {
                let asked = dot11::element(body, 0).unwrap_or(&[]);
                let named = asked == bss.ssid.as_slice();
                if !(named || (asked.is_empty() && !config.closednet)) {
                    return Vec::new();
                }
                let body = self.bss_body(config, bss, now, false);
                vec![reply(self, dot11::PROBE_RESPONSE, &body)]
            }
            // Open system authentication (algorithm 0, sequence 1).
            dot11::AUTH if header.addr3 == config.mac => {
                if body.get(..4) != Some(&[0, 0, 1, 0]) {
                    return Vec::new();
                }
                let bss = self.bss.as_mut().unwrap();
                if !bss.stations.iter().any(|s| s.mac == from) {
                    bss.stations.push(Peer {
                        mac: from,
                        aid: 0,
                        associated: false,
                    });
                }
                vec![
                    reply(self, dot11::AUTH, &[0, 0, 2, 0, 0, 0]),
                    event(
                        if_id,
                        WLC_E_AUTH_IND,
                        WLC_E_STATUS_SUCCESS,
                        from,
                        Vec::new(),
                    ),
                ]
            }
            // (Re)association of an authenticated station: capability,
            // listen interval (and the current AP for a reassociation),
            // then its elements, which the host gets with the event.
            dot11::ASSOC_REQUEST | 0x20 if header.addr3 == config.mac => {
                let reassoc = header.kind == 0x20;
                let Some(ies) = body.get(if reassoc { 10 } else { 4 }..) else {
                    return Vec::new();
                };
                let bss = self.bss.as_mut().unwrap();
                let associated = bss.stations.iter().filter(|s| s.associated).count();
                let Some(index) = bss.stations.iter().position(|s| s.mac == from) else {
                    // Not authenticated: deauthenticated (class 2 frame
                    // from an unauthenticated station, 11.3.3).
                    return vec![reply(self, dot11::DEAUTH, &7u16.to_le_bytes())];
                };
                let already = bss.stations[index].associated;
                let status = if !already && associated as u32 >= config.max_assoc.max(1) {
                    STATUS_TOO_MANY_STATIONS
                } else {
                    STATUS_SUCCESS
                };
                let aid = match status {
                    STATUS_SUCCESS => {
                        let aid = match bss.stations[index].aid {
                            0 => (1..=2007)
                                .find(|aid| bss.stations.iter().all(|s| s.aid != *aid))
                                .unwrap(),
                            aid => aid,
                        };
                        bss.stations[index].aid = aid;
                        bss.stations[index].associated = true;
                        aid
                    }
                    _ => 0,
                };
                let mut response = Self::capability(config).to_le_bytes().to_vec();
                response.extend(status.to_le_bytes());
                response.extend((aid | 0xc000).to_le_bytes());
                response.extend(rate_elements(channel));
                response.extend(self.vendor(VNDR_IE_ASSOC_RESPONSE));
                let kind = if reassoc { 0x30 } else { dot11::ASSOC_RESPONSE };
                let mut effects = vec![reply(self, kind, &response)];
                if status == STATUS_SUCCESS {
                    let kind = if reassoc {
                        WLC_E_REASSOC_IND
                    } else {
                        WLC_E_ASSOC_IND
                    };
                    effects.push(event(if_id, kind, WLC_E_STATUS_SUCCESS, from, ies.to_vec()));
                }
                effects
            }
            dot11::DEAUTH | dot11::DISASSOC if header.addr1 == config.mac => {
                let bss = self.bss.as_mut().unwrap();
                let before = bss.stations.len();
                bss.stations.retain(|s| s.mac != from);
                if bss.stations.len() == before {
                    return Vec::new();
                }
                self.keys.retain(|k| k.peer != Some(from));
                let kind = if header.kind == dot11::DEAUTH {
                    WLC_E_DEAUTH_IND
                } else {
                    WLC_E_DISASSOC_IND
                };
                let mut e = event(if_id, kind, WLC_E_STATUS_SUCCESS, from, Vec::new());
                if let (Effect::Event { reason, .. }, Some(code)) = (&mut e, body.get(..2)) {
                    *reason = u32::from(u16::from_le_bytes([code[0], code[1]]));
                }
                vec![e]
            }
            _ => Vec::new(),
        }
    }

    // ----- scanning -----

    /// A scan begins: it is on no channel until its first dwell starts.
    pub fn start_scan(&mut self, sync_id: u16, if_id: u8, ssid: &[u8], passive: bool) {
        self.scan = Some(Scan {
            sync_id,
            if_id,
            channel: None,
            ssid: ssid.to_vec(),
            passive,
            reported: Vec::new(),
        });
    }

    pub fn end_scan(&mut self, sync_id: u16) {
        if self.scan.as_ref().is_some_and(|s| s.sync_id == sync_id) {
            self.scan = None;
        }
    }

    /// The scan moves to `channel`; an active scan asks with a probe
    /// request (for the SSID it looks for, or any).
    pub fn scan_channel(&mut self, config: &Config, sync_id: u16, channel: u8) -> Vec<Effect> {
        let Some(scan) = self.scan.as_mut().filter(|s| s.sync_id == sync_id) else {
            return Vec::new();
        };
        scan.channel = Some(channel);
        if scan.passive || self.bss.is_some() {
            return Vec::new();
        }
        let ssid = scan.ssid.clone();
        vec![self.probe_request(config, &ssid, channel)]
    }

    fn probe_request(&mut self, config: &Config, ssid: &[u8], channel: u8) -> Effect {
        let mut body = ssid_element(ssid);
        body.extend(rate_elements(channel));
        body.extend(self.vendor(VNDR_IE_PROBE_REQUEST));
        let header = Header::new(
            dot11::PROBE_REQUEST,
            0,
            dot11::BROADCAST,
            config.mac,
            dot11::BROADCAST,
        );
        Effect::Transmit {
            channel,
            frame: self.frame(header, &body),
        }
    }

    /// A beacon or probe response heard on `channel`: remembered, reported
    /// to a scan on that channel that has not reported it yet, and the
    /// network a probing join looks for.
    fn heard(&mut self, config: &Config, header: &Header, body: &[u8], channel: u8) -> Vec<Effect> {
        let (Some(fixed), Some(ies)) = (body.get(..12), body.get(12..)) else {
            return Vec::new();
        };
        let bssid = header.addr3;
        let ssid = dot11::element(ies, 0).unwrap_or(&[]);
        let bss_info = wifi_ap::bss_info(&wifi_ap::Bss {
            bssid,
            beacon_period: u16::from_le_bytes([fixed[8], fixed[9]]),
            capability: u16::from_le_bytes([fixed[10], fixed[11]]),
            ssid,
            channel,
            dtim_period: dot11::element(ies, 5)
                .and_then(|t| t.get(1).copied())
                .unwrap_or(1),
            rssi: RSSI_DBM,
            ies,
        });
        let known_ssid = match self.heard.iter().find(|h| h.bssid == bssid) {
            Some(h) if ssid.iter().all(|&b| b == 0) => h.ssid.clone(),
            _ => ssid.to_vec(),
        };
        self.heard.retain(|h| h.bssid != bssid);
        self.heard.push(Heard {
            bssid,
            channel,
            ssid: known_ssid.clone(),
            bss_info: bss_info.clone(),
        });
        let mut effects = Vec::new();
        if let Some(scan) = self.scan.as_mut().filter(|s| s.channel == Some(channel))
            && !scan.reported.contains(&bssid)
            && (scan.ssid.is_empty() || scan.ssid == ssid)
        {
            scan.reported.push(bssid);
            effects.push(Effect::ScanResult {
                if_id: scan.if_id,
                sync_id: scan.sync_id,
                bss_info: bss_info.clone(),
            });
        }
        if let Some(link) = self.link.as_mut() {
            if link.bssid == Some(bssid) {
                link.bss_info = bss_info;
            }
            if matches!(link.stage, Stage::Probing(_))
                && header.kind == dot11::PROBE_RESPONSE
                && header.addr1 == config.mac
                && ssid == link.ssid.as_slice()
                && link.bssid.is_none_or(|b| b == bssid)
            {
                link.bssid = Some(bssid);
                link.channel = channel;
                if let Some(h) = self.heard.last_mut() {
                    h.ssid = ssid.to_vec();
                }
                effects.extend(self.authenticate(config));
            }
        }
        effects
    }

    // ----- joining a network -----

    /// The host asks to join `ssid` (at `bssid`, if not zero or broadcast):
    /// a network heard with that SSID or BSSID is authenticated with at
    /// once; otherwise the radio probes for it on `channels` in turn.
    pub fn join(
        &mut self,
        config: &Config,
        if_id: u8,
        epoch: u32,
        ssid: &[u8],
        bssid: [u8; 6],
        channels: Vec<u8>,
    ) -> Vec<Effect> {
        self.bss = None;
        let named = (bssid != [0; 6] && bssid != dot11::BROADCAST).then_some(bssid);
        let known = self
            .heard
            .iter()
            .find(|h| named.map_or(h.ssid == ssid, |b| b == h.bssid))
            .cloned();
        self.link = Some(Link {
            if_id,
            epoch,
            ssid: ssid.to_vec(),
            bssid: known.as_ref().map(|h| h.bssid).or(named),
            channel: known.as_ref().map_or(0, |h| h.channel),
            channels,
            stage: Stage::Probing(0),
            bss_info: known
                .as_ref()
                .map(|h| h.bss_info.clone())
                .unwrap_or_default(),
        });
        match known {
            Some(_) => self.authenticate(config),
            None => self.probe(config, epoch, 0),
        }
    }

    pub fn probe(&mut self, config: &Config, epoch: u32, index: usize) -> Vec<Effect> {
        let Some(link) = self.link.as_mut().filter(|l| l.epoch == epoch) else {
            return Vec::new();
        };
        if !matches!(link.stage, Stage::Probing(_)) {
            return Vec::new();
        }
        let Some(&channel) = link.channels.get(index) else {
            let if_id = link.if_id;
            self.link = None;
            return vec![event(
                if_id,
                WLC_E_SET_SSID,
                WLC_E_STATUS_NO_NETWORKS,
                [0; 6],
                Vec::new(),
            )];
        };
        link.stage = Stage::Probing(index);
        let ssid = link.ssid.clone();
        vec![
            self.probe_request(config, &ssid, channel),
            Effect::Later {
                delay: PROBE_DWELL,
                work: Work::Probe {
                    epoch,
                    index: index + 1,
                },
            },
        ]
    }

    fn authenticate(&mut self, config: &Config) -> Vec<Effect> {
        let link = self.link.as_mut().unwrap();
        link.stage = Stage::Authenticating;
        let (bssid, channel, epoch) = (link.bssid.unwrap(), link.channel, link.epoch);
        let header = Header::new(dot11::AUTH, 0, bssid, config.mac, bssid);
        vec![
            Effect::Transmit {
                channel,
                frame: self.frame(header, &[0, 0, 1, 0, 0, 0]),
            },
            Effect::Later {
                delay: ANSWER_TIMEOUT,
                work: Work::Timeout {
                    epoch,
                    stage: Stage::Authenticating,
                },
            },
        ]
    }

    /// No answer in time: the join fails.
    pub fn timeout(&mut self, epoch: u32, stage: Stage) -> Vec<Effect> {
        let Some(link) = self
            .link
            .as_ref()
            .filter(|l| l.epoch == epoch && l.stage == stage)
        else {
            return Vec::new();
        };
        let if_id = link.if_id;
        self.link = None;
        vec![event(
            if_id,
            WLC_E_SET_SSID,
            WLC_E_STATUS_FAIL,
            [0; 6],
            Vec::new(),
        )]
    }

    /// A management frame from the network the station joins.
    fn station_management(&mut self, config: &Config, header: &Header, body: &[u8]) -> Vec<Effect> {
        let Some(link) = self.link.as_mut() else {
            return Vec::new();
        };
        if Some(header.addr2) != link.bssid || header.addr1 != config.mac {
            return Vec::new();
        }
        let (if_id, bssid, channel, epoch) = (link.if_id, header.addr2, link.channel, link.epoch);
        let status = |at: usize| {
            body.get(at..at + 2)
                .map(|s| u16::from_le_bytes([s[0], s[1]]))
        };
        match (header.kind, link.stage) {
            (dot11::AUTH, Stage::Authenticating) => {
                if body.get(..4) != Some(&[0, 0, 2, 0]) {
                    return Vec::new();
                }
                if status(4) != Some(STATUS_SUCCESS) {
                    self.link = None;
                    return vec![event(
                        if_id,
                        WLC_E_SET_SSID,
                        WLC_E_STATUS_FAIL,
                        bssid,
                        Vec::new(),
                    )];
                }
                link.stage = Stage::Associating;
                let ssid = link.ssid.clone();
                let mut request = Self::capability(config).to_le_bytes().to_vec();
                request.extend(10u16.to_le_bytes());
                request.extend(ssid_element(&ssid));
                request.extend(rate_elements(channel));
                if !config.wpa_ie.is_empty() {
                    request.extend(&config.wpa_ie);
                }
                request.extend(self.vendor(VNDR_IE_ASSOC_REQUEST));
                let header = Header::new(dot11::ASSOC_REQUEST, 0, bssid, config.mac, bssid);
                vec![
                    Effect::Transmit {
                        channel,
                        frame: self.frame(header, &request),
                    },
                    Effect::Later {
                        delay: ANSWER_TIMEOUT,
                        work: Work::Timeout {
                            epoch,
                            stage: Stage::Associating,
                        },
                    },
                ]
            }
            (dot11::ASSOC_RESPONSE, Stage::Associating) => {
                if status(2) != Some(STATUS_SUCCESS) {
                    self.link = None;
                    return vec![event(
                        if_id,
                        WLC_E_SET_SSID,
                        WLC_E_STATUS_FAIL,
                        bssid,
                        Vec::new(),
                    )];
                }
                link.stage = Stage::Associated;
                let ssid = link.ssid.clone();
                let mut effects: Vec<Effect> = [WLC_E_AUTH, WLC_E_ASSOC, WLC_E_JOIN]
                    .into_iter()
                    .map(|kind| event(if_id, kind, WLC_E_STATUS_SUCCESS, bssid, Vec::new()))
                    .collect();
                let mut link = event(if_id, WLC_E_LINK, WLC_E_STATUS_SUCCESS, bssid, Vec::new());
                if let Effect::Event { flags, .. } = &mut link {
                    *flags = WLC_EVENT_MSG_LINK;
                }
                effects.push(link);
                effects.push(event(
                    if_id,
                    WLC_E_SET_SSID,
                    WLC_E_STATUS_SUCCESS,
                    bssid,
                    ssid,
                ));
                effects
            }
            // The network sent the station away: the host hears it and the
            // link going down.
            (dot11::DEAUTH | dot11::DISASSOC, Stage::Associated) => {
                self.link = None;
                self.keys.retain(|k| k.peer.is_none());
                let kind = if header.kind == dot11::DEAUTH {
                    WLC_E_DEAUTH_IND
                } else {
                    WLC_E_DISASSOC_IND
                };
                let reason = u32::from(status(0).unwrap_or(0));
                let mut leave = event(if_id, kind, WLC_E_STATUS_SUCCESS, bssid, Vec::new());
                if let Effect::Event { reason: r, .. } = &mut leave {
                    *r = reason;
                }
                vec![
                    leave,
                    event(if_id, WLC_E_LINK, WLC_E_STATUS_SUCCESS, bssid, Vec::new()),
                ]
            }
            _ => Vec::new(),
        }
    }

    /// The host leaves the network it joined: the access point hears a
    /// deauthentication.
    pub fn leave(&mut self, config: &Config) -> Vec<Effect> {
        let Some(link) = self.link.take() else {
            return Vec::new();
        };
        self.keys.retain(|k| k.peer.is_none());
        let (Some(bssid), Stage::Associated) = (link.bssid, link.stage) else {
            return Vec::new();
        };
        let header = Header::new(dot11::DEAUTH, 0, bssid, config.mac, bssid);
        vec![Effect::Transmit {
            channel: link.channel,
            frame: self.frame(header, &REASON_LEAVING.to_le_bytes()),
        }]
    }

    // ----- frames from the air -----

    /// A frame heard on `channel`.
    pub fn receive(
        &mut self,
        config: &Config,
        channel: u8,
        frame: &[u8],
        now: Duration,
    ) -> Vec<Effect> {
        if self.channel() != Some(channel) {
            return Vec::new();
        }
        let Some((header, at)) = Header::parse(frame) else {
            return Vec::new();
        };
        if header.addr2 == config.mac {
            return Vec::new();
        }
        let body = &frame[at..];
        if header.is_management() {
            let mut effects = match header.kind {
                dot11::BEACON | dot11::PROBE_RESPONSE => self.heard(config, &header, body, channel),
                dot11::ACTION if header.addr1 == config.mac || header.addr1[0] & 1 != 0 => {
                    vec![Effect::ActionFrame {
                        channel,
                        frame: frame.to_vec(),
                    }]
                }
                _ => Vec::new(),
            };
            if self.bss.is_some() {
                effects.extend(self.ap_management(config, &header, body, now));
            } else {
                effects.extend(self.station_management(config, &header, body));
            }
            return effects;
        }
        if header.is_data() {
            return self.receive_data(config, &header, frame, at);
        }
        Vec::new()
    }

    /// An action frame the host sends: to `da`, in the BSS `bssid`, on
    /// `channel`; `body` from the category on.
    pub fn action_frame(
        &mut self,
        config: &Config,
        channel: u8,
        da: [u8; 6],
        bssid: [u8; 6],
        body: &[u8],
    ) -> Effect {
        let header = Header::new(dot11::ACTION, 0, da, config.mac, bssid);
        Effect::Transmit {
            channel,
            frame: self.frame(header, body),
        }
    }

    fn key_for_receiving(&self, header: &Header, key_id: u8) -> Option<&Key> {
        let multicast = header.addr1[0] & 1 != 0;
        let pairwise = (!multicast)
            .then(|| self.keys.iter().find(|k| k.peer == Some(header.addr2)))
            .flatten();
        pairwise.or_else(|| {
            self.keys
                .iter()
                .find(|k| k.peer.is_none() && k.index == u32::from(key_id))
        })
    }

    fn receive_data(
        &mut self,
        config: &Config,
        header: &Header,
        frame: &[u8],
        at: usize,
    ) -> Vec<Effect> {
        let body = if header.flags & dot11::PROTECTED != 0 {
            let Some(key_id) = frame.get(at + 3).map(|b| b >> 6) else {
                return Vec::new();
            };
            let Some(key) = self.key_for_receiving(header, key_id) else {
                return Vec::new();
            };
            match Ccmp::new(&key.data).decrypt(header, frame, at) {
                Some((body, _, _)) => body,
                None => return Vec::new(),
            }
        } else {
            // With AES on, only the port-control frames of the 4-way
            // handshake travel in the clear (12.7.6).
            let body = frame[at..].to_vec();
            let eapol = body.get(6..8) == Some(&wifi_ap::ETHERTYPE_EAPOL.to_be_bytes());
            if config.wsec & AES_ENABLED != 0 && !eapol && !self.keys.is_empty() {
                return Vec::new();
            }
            body
        };
        let ds = header.flags & (dot11::TO_DS | dot11::FROM_DS);
        if let Some(bss) = self.bss.as_ref() {
            // To the access point from one of its stations.
            let associated = bss
                .stations
                .iter()
                .any(|s| s.mac == header.addr2 && s.associated);
            if ds != dot11::TO_DS || header.addr1 != config.mac || !associated {
                return Vec::new();
            }
            let (if_id, da, sa) = (bss.if_id, header.addr3, header.addr2);
            let Some(ethernet) = dot11::ethernet(da, sa, &body) else {
                return Vec::new();
            };
            let multicast = da[0] & 1 != 0;
            let to_station = bss.stations.iter().any(|s| s.mac == da && s.associated);
            let mut effects = Vec::new();
            if (multicast || to_station) && !config.ap_isolate {
                effects.extend(self.ap_transmit(config, &ethernet));
            }
            if multicast || da == config.mac {
                effects.push(Effect::Deliver {
                    if_id,
                    frame: ethernet,
                });
            }
            return effects;
        }
        let Some(link) = self.link.as_ref().filter(|l| l.stage == Stage::Associated) else {
            return Vec::new();
        };
        if ds != dot11::FROM_DS
            || Some(header.addr2) != link.bssid
            || (header.addr1 != config.mac && header.addr1[0] & 1 == 0)
        {
            return Vec::new();
        }
        // A station hears its own multicast again from the access point.
        if header.addr3 == config.mac {
            return Vec::new();
        }
        match dot11::ethernet(header.addr1, header.addr3, &body) {
            Some(frame) => vec![Effect::Deliver {
                if_id: link.if_id,
                frame,
            }],
            None => Vec::new(),
        }
    }

    // ----- frames from the host -----

    /// The key a frame to `receiver` is sent with: the pairwise key of the
    /// peer, else the primary default key (any default key if none is
    /// marked primary).
    fn key_for_sending(&mut self, receiver: [u8; 6]) -> Option<&mut Key> {
        let multicast = receiver[0] & 1 != 0;
        let pairwise = (!multicast)
            .then(|| self.keys.iter().position(|k| k.peer == Some(receiver)))
            .flatten();
        let default = || {
            self.keys
                .iter()
                .position(|k| k.peer.is_none() && k.primary)
                .or_else(|| self.keys.iter().position(|k| k.peer.is_none()))
        };
        let index = pairwise.or_else(default)?;
        Some(&mut self.keys[index])
    }

    fn data_frame(
        &mut self,
        config: &Config,
        header: Header,
        body: &[u8],
        channel: u8,
    ) -> Vec<Effect> {
        let mut header = header;
        header.sequence = self.next_sequence();
        let eapol = body.get(6..8) == Some(&wifi_ap::ETHERTYPE_EAPOL.to_be_bytes());
        let frame = match self.key_for_sending(header.addr1) {
            Some(key) if config.wsec & AES_ENABLED != 0 => {
                let pn = key.next_pn;
                key.next_pn += 1;
                Ccmp::new(&key.data).encrypt(&header, pn, key.index as u8, body)
            }
            _ if config.wsec & AES_ENABLED != 0 && !eapol => return Vec::new(),
            _ => header.frame(body),
        };
        vec![Effect::Transmit { channel, frame }]
    }

    /// An Ethernet frame the access point sends: to a station of its BSS,
    /// or to all of them.
    fn ap_transmit(&mut self, config: &Config, ethernet: &[u8]) -> Vec<Effect> {
        let Some(bss) = self.bss.as_ref() else {
            return Vec::new();
        };
        let (Some(da), Some(sa), Some(body)) = (
            ethernet.get(..6),
            ethernet.get(6..12),
            dot11::snap_body(ethernet),
        ) else {
            return Vec::new();
        };
        let da: [u8; 6] = da.try_into().unwrap();
        let sa: [u8; 6] = sa.try_into().unwrap();
        if da[0] & 1 == 0 && !bss.stations.iter().any(|s| s.mac == da && s.associated) {
            return Vec::new();
        }
        let channel = bss.channel;
        let header = Header::new(dot11::DATA, dot11::FROM_DS, da, config.mac, sa);
        self.data_frame(config, header, &body, channel)
    }

    /// An Ethernet frame the host transmits on the air: through its own
    /// BSS, or to the network it joined. Whether the radio took it (it is
    /// lost if it is on neither).
    pub fn transmit(&mut self, config: &Config, ethernet: &[u8]) -> Option<Vec<Effect>> {
        if self.bss.is_some() {
            return Some(self.ap_transmit(config, ethernet));
        }
        let link = self
            .link
            .as_ref()
            .filter(|l| l.stage == Stage::Associated)?;
        let (bssid, channel) = (link.bssid?, link.channel);
        let da: [u8; 6] = ethernet.get(..6)?.try_into().unwrap();
        let body = dot11::snap_body(ethernet)?;
        let header = Header::new(dot11::DATA, dot11::TO_DS, bssid, config.mac, da);
        Some(self.data_frame(config, header, &body, channel))
    }

    /// Installs or removes a key.
    pub fn change_key(&mut self, change: KeyChange) {
        match change {
            KeyChange::Install(key) => {
                self.keys
                    .retain(|k| !(k.index == key.index && k.peer == key.peer));
                if key.primary && key.peer.is_none() {
                    for k in self.keys.iter_mut().filter(|k| k.peer.is_none()) {
                        k.primary = false;
                    }
                }
                self.keys.push(key);
            }
            KeyChange::Remove { index, peer } => {
                self.keys.retain(|k| !(k.index == index && k.peer == peer));
            }
        }
    }

    /// `vndr_ie`: `vndr_ie_setbuf_t`, a command ("add" or "del"), a count,
    /// then per element its packet flags (32 bits) and the element (ID 221,
    /// length, OUI, data). Public `wlioctl.h`.
    pub fn vendor_elements(&mut self, value: &[u8]) -> Result<(), String> {
        let command = value.get(..4).ok_or("vndr_ie without a command")?;
        let add = match command {
            b"add\0" => true,
            b"del\0" => false,
            _ => return Err(format!("vndr_ie command {command:02x?}")),
        };
        let count = value
            .get(4..8)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .ok_or("vndr_ie without a count")?;
        let mut at = 8;
        for _ in 0..count {
            let flags = value
                .get(at..at + 4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .ok_or("vndr_ie truncated")?;
            let length = usize::from(*value.get(at + 5).ok_or("vndr_ie truncated")?);
            let ie = value
                .get(at + 4..at + 6 + length)
                .ok_or("vndr_ie truncated")?
                .to_vec();
            at += 6 + length;
            if add {
                self.vendor_ies.push((flags, ie));
            } else {
                self.vendor_ies.retain(|(f, e)| !(*f == flags && *e == ie));
            }
        }
        Ok(())
    }

    /// The radio goes down: its BSS and link end without a word on the
    /// air (\[assumption\]: a firmware taken down does not say goodbye),
    /// scans stop.
    pub fn down(&mut self) {
        self.bss = None;
        self.link = None;
        self.scan = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const AP: [u8; 6] = [2, 0, 0, 0, 0, 1];
    const STA: [u8; 6] = [2, 0, 0, 0, 0, 2];

    fn config(mac: [u8; 6]) -> Config {
        Config {
            mac,
            wsec: AES_ENABLED,
            wpa_ie: Vec::new(),
            closednet: true,
            max_assoc: 1,
            beacon_period: 100,
            dtim_period: 1,
            ap_isolate: false,
        }
    }

    fn key() -> Key {
        Key {
            index: 0,
            peer: None,
            data: [0x42; 16],
            primary: true,
            next_pn: 1,
        }
    }

    fn transmitted(effects: &[Effect]) -> Vec<(u8, Vec<u8>)> {
        effects
            .iter()
            .filter_map(|e| match e {
                Effect::Transmit { channel, frame } => Some((*channel, frame.clone())),
                _ => None,
            })
            .collect()
    }

    fn events(effects: &[Effect]) -> Vec<(u32, u32, [u8; 6])> {
        effects
            .iter()
            .filter_map(|e| match e {
                Effect::Event {
                    kind, status, addr, ..
                } => Some((*kind, *status, *addr)),
                _ => None,
            })
            .collect()
    }

    /// Runs frames between two radios until neither sends more.
    fn exchange(
        a: (&mut Radio, &Config),
        b: (&mut Radio, &Config),
        first: Vec<Effect>,
    ) -> (Vec<Effect>, Vec<Effect>) {
        let (mut seen_a, mut seen_b) = (Vec::new(), Vec::new());
        let mut to_b = transmitted(&first);
        seen_a.extend(first);
        let mut to_a: Vec<(u8, Vec<u8>)> = Vec::new();
        while !(to_a.is_empty() && to_b.is_empty()) {
            let mut next_a = Vec::new();
            for (channel, frame) in to_b.drain(..) {
                let effects = b.0.receive(b.1, channel, &frame, Duration::ZERO);
                next_a.extend(transmitted(&effects));
                seen_b.extend(effects);
            }
            for (channel, frame) in to_a.drain(..) {
                let effects = a.0.receive(a.1, channel, &frame, Duration::ZERO);
                to_b.extend(transmitted(&effects));
                seen_a.extend(effects);
            }
            to_a = next_a;
        }
        (seen_a, seen_b)
    }

    /// A hidden BSS: its beacon carries no SSID; a scan on its channel
    /// reports it once; a station that names the SSID joins (probe,
    /// authentication, association), both hosts hear the events, and data
    /// goes both ways under CCMP with the default key.
    #[test]
    fn hidden_network_scan_join_and_data() {
        let (ap_config, sta_config) = (config(AP), config(STA));
        let (mut ap, mut sta) = (Radio::default(), Radio::default());
        ap.change_key(KeyChange::Install(key()));
        sta.change_key(KeyChange::Install(key()));
        let started = ap.start_bss(&ap_config, 0, b"0123456789abcdef", 6);
        assert_eq!(
            events(&started),
            [
                (WLC_E_LINK, WLC_E_STATUS_SUCCESS, AP),
                (WLC_E_SET_SSID, WLC_E_STATUS_SUCCESS, AP)
            ]
        );
        let epoch = ap.bss.as_ref().unwrap().epoch;
        let beacon = transmitted(&ap.beacon(&ap_config, epoch, Duration::ZERO));
        let (header, at) = Header::parse(&beacon[0].1).unwrap();
        assert_eq!(header.kind, dot11::BEACON);
        assert_eq!(dot11::element(&beacon[0].1[at + 12..], 0), Some(&[][..]));

        // A passive scan on another channel hears nothing; on channel 6 it
        // reports the network once.
        sta.start_scan(1, 0, &[], true);
        assert!(sta.scan_channel(&sta_config, 1, 1).is_empty());
        assert!(
            sta.receive(&sta_config, 6, &beacon[0].1, Duration::ZERO)
                .is_empty()
        );
        sta.scan_channel(&sta_config, 1, 6);
        let found = sta.receive(&sta_config, 6, &beacon[0].1, Duration::ZERO);
        assert!(matches!(
            &found[..],
            [Effect::ScanResult { sync_id: 1, .. }]
        ));
        assert!(
            sta.receive(&sta_config, 6, &beacon[0].1, Duration::ZERO)
                .is_empty()
        );
        sta.end_scan(1);

        // Join by SSID: the hidden network is found by a directed probe.
        let joined = sta.join(&sta_config, 0, 7, b"0123456789abcdef", [0; 6], vec![1, 6]);
        let (sta_seen, ap_seen) = exchange((&mut sta, &sta_config), (&mut ap, &ap_config), joined);
        // On channel 1 nobody answers; the station moves on when its dwell
        // there ends.
        assert!(events(&sta_seen).is_empty());
        let next = sta.probe(&sta_config, 7, 1);
        let (sta_seen, ap_seen_2) = exchange((&mut sta, &sta_config), (&mut ap, &ap_config), next);
        assert!(ap_seen.is_empty());
        assert_eq!(
            events(&ap_seen_2),
            [
                (WLC_E_AUTH_IND, WLC_E_STATUS_SUCCESS, STA),
                (WLC_E_ASSOC_IND, WLC_E_STATUS_SUCCESS, STA)
            ]
        );
        assert_eq!(
            events(&sta_seen).iter().map(|e| e.0).collect::<Vec<_>>(),
            [
                WLC_E_AUTH,
                WLC_E_ASSOC,
                WLC_E_JOIN,
                WLC_E_LINK,
                WLC_E_SET_SSID
            ]
        );
        assert!(
            sta.link
                .as_ref()
                .is_some_and(|l| l.stage == Stage::Associated)
        );

        // Data, encrypted on the air.
        let mut ethernet = AP.to_vec();
        ethernet.extend(STA);
        ethernet.extend([0x08, 0x00, 0x45, 1, 2, 3]);
        let sent = transmitted(&sta.transmit(&sta_config, &ethernet).unwrap());
        let (header, _) = Header::parse(&sent[0].1).unwrap();
        assert_ne!(header.flags & dot11::PROTECTED, 0);
        assert!(!sent[0].1.windows(4).any(|w| w == [0x45, 1, 2, 3]));
        let got = ap.receive(&ap_config, 6, &sent[0].1, Duration::ZERO);
        assert!(matches!(&got[..], [Effect::Deliver { frame, .. }] if *frame == ethernet));
        let mut reply = STA.to_vec();
        reply.extend(AP);
        reply.extend([0x08, 0x00, 0x45, 4, 5, 6]);
        let sent = transmitted(&ap.transmit(&ap_config, &reply).unwrap());
        let got = sta.receive(&sta_config, 6, &sent[0].1, Duration::ZERO);
        assert!(matches!(&got[..], [Effect::Deliver { frame, .. }] if *frame == reply));

        // With another key the frame does not decrypt.
        let mut other = Radio::default();
        other.change_key(KeyChange::Install(Key {
            data: [1; 16],
            ..key()
        }));
        other.link = sta.link.take();
        assert!(
            other
                .receive(&sta_config, 6, &sent[0].1, Duration::ZERO)
                .is_empty()
        );
    }

    /// The BSS takes `maxassoc` stations; one more is refused, and a
    /// station that leaves is reported and makes room.
    #[test]
    fn access_point_limits_and_releases_stations() {
        let ap_config = Config {
            wsec: 0,
            closednet: false,
            ..config(AP)
        };
        let mut ap = Radio::default();
        ap.start_bss(&ap_config, 0, b"net", 1);
        let join = |ap: &mut Radio, mac: [u8; 6]| {
            let auth = Header::new(dot11::AUTH, 0, AP, mac, AP).frame(&[0, 0, 1, 0, 0, 0]);
            let mut effects = ap.receive(&ap_config, 1, &auth, Duration::ZERO);
            let mut request = 0x0421u16.to_le_bytes().to_vec();
            request.extend(10u16.to_le_bytes());
            request.extend(ssid_element(b"net"));
            let assoc = Header::new(dot11::ASSOC_REQUEST, 0, AP, mac, AP).frame(&request);
            effects.extend(ap.receive(&ap_config, 1, &assoc, Duration::ZERO));
            effects
        };
        let status = |effects: &[Effect]| {
            transmitted(effects)
                .iter()
                .find_map(|(_, f)| {
                    (f[0] == dot11::ASSOC_RESPONSE).then(|| u16::from_le_bytes([f[26], f[27]]))
                })
                .unwrap()
        };
        let first = join(&mut ap, STA);
        assert_eq!(status(&first), STATUS_SUCCESS);
        let other = [2, 0, 0, 0, 0, 3];
        let second = join(&mut ap, other);
        assert_eq!(status(&second), STATUS_TOO_MANY_STATIONS);
        assert!(!events(&second).iter().any(|e| e.0 == WLC_E_ASSOC_IND));
        let leave = Header::new(dot11::DEAUTH, 0, AP, STA, AP).frame(&REASON_LEAVING.to_le_bytes());
        assert_eq!(
            events(&ap.receive(&ap_config, 1, &leave, Duration::ZERO)),
            [(WLC_E_DEAUTH_IND, WLC_E_STATUS_SUCCESS, STA)]
        );
        assert_eq!(status(&join(&mut ap, other)), STATUS_SUCCESS);
        // An open network answers a wildcard probe; frames on another
        // channel are not heard.
        let probe = Header::new(
            dot11::PROBE_REQUEST,
            0,
            dot11::BROADCAST,
            STA,
            dot11::BROADCAST,
        )
        .frame(&ssid_element(&[]));
        assert_eq!(
            transmitted(&ap.receive(&ap_config, 1, &probe, Duration::ZERO)).len(),
            1
        );
        assert!(ap.receive(&ap_config, 6, &probe, Duration::ZERO).is_empty());
    }

    #[test]
    fn join_without_an_answer_fails_after_every_channel() {
        let sta_config = config(STA);
        let mut sta = Radio::default();
        let effects = sta.join(&sta_config, 0, 3, b"nobody", [0; 6], vec![1, 6]);
        assert_eq!(transmitted(&effects).len(), 1);
        sta.probe(&sta_config, 3, 1);
        let end = sta.probe(&sta_config, 3, 2);
        assert_eq!(
            events(&end),
            [(WLC_E_SET_SSID, WLC_E_STATUS_NO_NETWORKS, [0; 6])]
        );
        assert!(sta.link.is_none());
    }

    #[test]
    fn vendor_elements_are_added_and_deleted() {
        let mut radio = Radio::default();
        let mut add = b"add\0".to_vec();
        add.extend(1u32.to_le_bytes());
        add.extend((VNDR_IE_BEACON | VNDR_IE_PROBE_RESPONSE).to_le_bytes());
        add.extend([221, 5, 0x00, 0x22, 0xaa, 1, 2]);
        radio.vendor_elements(&add).unwrap();
        assert_eq!(
            radio.vendor(VNDR_IE_BEACON),
            [221, 5, 0x00, 0x22, 0xaa, 1, 2]
        );
        assert!(radio.vendor(VNDR_IE_ASSOC_REQUEST).is_empty());
        let mut del = add.clone();
        del[..4].copy_from_slice(b"del\0");
        radio.vendor_elements(&del).unwrap();
        assert!(radio.vendor(VNDR_IE_BEACON).is_empty());
    }
}
