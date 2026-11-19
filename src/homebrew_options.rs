//! The `boot` options that put programs Nintendo did not sign on the
//! console: `--disable-program-verification`, `--homebrew-keys`,
//! `--replace-program` and `--homebrew` with its companions
//! (`horizon_emu::homebrew`).

use std::{fs::File, path::Path};

use horizon_emu::{
    homebrew::{
        self, firmware,
        program::{self, Identity},
    },
    image::KernelImage,
    keys::KeySet,
    nand::{self, Overlay},
    sdcard,
    soc::sdcard::SdCardImage,
};

pub const USAGE: &str = "\
running programs Nintendo did not sign (horizon_emu::homebrew):
  --disable-program-verification
                      have the secure monitor report program verification
                      as disabled (config item 1), as on consoles configured
                      so. Firmware 23.5.0 heeds it only on development
                      hardware, which this console is not
  --homebrew-keys     a modification of the firmware's code: replace the
                      filesystem process's and the loader's public keys of
                      signature key generation 1, which none of the
                      firmware's contents use, with the emulator's
                      homebrew test key, in the kernel image's INI1 before
                      the console starts (the file stays as it is)
  --replace-program PROGRAM_ID=NSP
                      serve, in place of that program's content, a content
                      archive made from the ExeFS NSP (main, main.npdm and
                      any rtld, sdk, subsdk0-9), signed with the homebrew
                      test key. Served from memory: the eMMC image keeps
                      the original, also with --emmc-writable. Needs
                      --homebrew-keys; may be given more than once
  --homebrew NRO      run homebrew the way Atmosphere does: hbloader in
                      place of the Album applet loads NRO as sdmc:/hbmenu.nro,
                      from a 4 GiB microSD card made in memory (no --sd).
                      Implies --homebrew-keys, --disable-program-verification
                      and --replace-program 010000000000100D=HBL
  --homebrew-in PROGRAM_ID
                      with --homebrew: hbloader in place of that program
                      instead of the Album applet, as that program (its ID
                      and application type), the way Atmosphere overrides a
                      title. In place of an application hbloader is one,
                      and what it loads gets an application's memory and
                      services. A console with no games starts one
                      application: starter (0100000000001012), which sets
                      up a console that has not been set up
  --hbl NSP           the hbloader ExeFS for --homebrew (default: hbl.nsp
                      next to NRO)
  --homebrew-app NRO  also put NRO on the --homebrew card, as
                      /switch/<its file name>, where hbmenu lists it; may
                      be given more than once";

/// The Album applet, which Atmosphere has hbloader stand in for.
const ALBUM: u64 = 0x0100_0000_0000_100D;

#[derive(Default)]
pub struct HomebrewOptions {
    verification_disabled: bool,
    keys: bool,
    replacements: Vec<(u64, String)>,
    homebrew: Option<String>,
    homebrew_in: Option<u64>,
    hbl: Option<String>,
    apps: Vec<String>,
}

impl HomebrewOptions {
    /// Takes `flag` if it is one of these options, with its value from
    /// `value`.
    pub fn parse<'a>(
        &mut self,
        flag: &str,
        value: &mut impl FnMut() -> Result<&'a String, String>,
    ) -> Result<bool, String> {
        let program_id = |flag: &str, id: &str| {
            u64::from_str_radix(id.trim_start_matches("0x"), 16)
                .map_err(|_| format!("{flag}: the program ID is in hex, not {id}"))
        };
        match flag {
            "--disable-program-verification" => self.verification_disabled = true,
            "--homebrew-keys" => self.keys = true,
            "--replace-program" => {
                let value = value()?;
                let (program, nsp) = value
                    .split_once('=')
                    .ok_or("--replace-program: expected PROGRAM_ID=NSP, the ID in hex")?;
                self.replacements
                    .push((program_id(flag, program)?, nsp.to_string()));
            }
            "--homebrew" => self.homebrew = Some(value()?.clone()),
            "--homebrew-in" => self.homebrew_in = Some(program_id(flag, value()?)?),
            "--hbl" => self.hbl = Some(value()?.clone()),
            "--homebrew-app" => self.apps.push(value()?.clone()),
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// Whether the secure monitor is to report program verification as
    /// disabled.
    pub fn disables_verification(&self) -> bool {
        self.verification_disabled || self.homebrew.is_some()
    }

    fn installs_keys(&self) -> bool {
        self.keys || self.homebrew.is_some()
    }

    fn replacements(&self) -> Result<Vec<(u64, String, Identity)>, String> {
        let mut replacements: Vec<_> = self
            .replacements
            .iter()
            .map(|(program, nsp)| (*program, nsp.clone(), Identity::Own))
            .collect();
        match (&self.homebrew, &self.hbl, self.homebrew_in) {
            (Some(nro), hbl, place) => {
                let hbl = hbl.clone().unwrap_or_else(|| {
                    let directory = Path::new(nro).parent().unwrap_or(Path::new(""));
                    directory.join("hbl.nsp").to_string_lossy().into_owned()
                });
                replacements.push((place.unwrap_or(ALBUM), hbl, Identity::Replaced));
            }
            (None, Some(_), _) => return Err("--hbl goes with --homebrew".into()),
            (None, None, Some(_)) => return Err("--homebrew-in goes with --homebrew".into()),
            (None, None, None) => {}
        }
        Ok(replacements)
    }

    /// Replaces the keys in `image` if asked to; a line for each.
    pub fn install_keys(&self, image: &mut KernelImage) -> Result<Vec<String>, String> {
        if !self.installs_keys() {
            return Ok(Vec::new());
        }
        let ini1 = image
            .ini1_mut()
            .ok_or("--homebrew-keys: the kernel image has no INI1")?;
        firmware::install_homebrew_keys(ini1, &homebrew::modulus())
            .map_err(|error| format!("--homebrew-keys: {error}"))
    }

    /// What the replaced programs lay over the user area `user`, and a line
    /// for each.
    pub fn overlay(
        &self,
        user: Option<&File>,
        keys: &KeySet,
    ) -> Result<(Overlay, Vec<String>), String> {
        let replacements = self.replacements()?;
        let (mut overlay, mut report) = (Overlay::new(), Vec::new());
        if replacements.is_empty() {
            return Ok((overlay, report));
        }
        let user = user.ok_or("--replace-program needs --emmc")?;
        if !self.installs_keys() {
            return Err(
                "--replace-program: the archive is signed with the homebrew test key, \
                        which the filesystem process accepts only with --homebrew-keys"
                    .into(),
            );
        }
        for (program_id, nsp, identity) in replacements {
            let failed =
                |error: String| format!("--replace-program {program_id:016x}={nsp}: {error}");
            let exefs = std::fs::read(&nsp).map_err(|error| failed(error.to_string()))?;
            let mut dropped = Vec::new();
            let replacement = nand::replace_program(user, keys, program_id, |original| {
                let (archive, changes) =
                    program::build_program_nca(keys, original, program_id, &exefs, identity)
                        .map_err(|error| error.to_string())?;
                dropped = changes;
                Ok(archive)
            })
            .map_err(|error| failed(error.to_string()))?;
            report.extend(
                dropped
                    .into_iter()
                    .map(|line| format!("program {program_id:016x}: {line}")),
            );
            report.push(format!(
                "program {program_id:016x}: SYSTEM:{} ({} bytes) now begins with the {} bytes made from {nsp}",
                replacement.path, replacement.original_size, replacement.size
            ));
            overlay.extend(replacement.overlay);
        }
        Ok((overlay, report))
    }
}

impl HomebrewOptions {
    /// Whether `--homebrew` makes the console's microSD card.
    pub fn makes_sd_card(&self) -> bool {
        self.homebrew.is_some()
    }

    /// For `--homebrew`, a 4 GiB card with the NRO as `/hbmenu.nro`, where
    /// hbloader looks for it, and `/switch/` with the `--homebrew-app`s.
    pub fn sd_card(&self) -> Result<Option<SdCardImage>, String> {
        let Some(nro) = &self.homebrew else {
            if let Some(app) = self.apps.first() {
                return Err(format!("--homebrew-app {app} goes with --homebrew"));
            }
            return Ok(None);
        };
        let read = |path: &String| {
            std::fs::read(path).map_err(|error| format!("--homebrew {path}: {error}"))
        };
        let menu = read(nro)?;
        let mut apps = Vec::new();
        for app in &self.apps {
            let name = Path::new(app)
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| format!("--homebrew-app {app}: no file name"))?;
            apps.push((format!("/switch/{name}"), read(app)?));
        }
        let mut files: Vec<(&str, &[u8])> = vec![("/hbmenu.nro", &menu), ("/switch/", &[])];
        files.extend(
            apps.iter()
                .map(|(path, data)| (path.as_str(), data.as_slice())),
        );
        let image = sdcard::image_with_files(SD_CARD_SIZE, &files)
            .map_err(|error| format!("--homebrew: {error}"))?;
        SdCardImage::memory(image)
            .map(Some)
            .map_err(|error| format!("--homebrew: {error}"))
    }
}

/// Size of the card `--homebrew` makes.
const SD_CARD_SIZE: u64 = 4 << 30;
