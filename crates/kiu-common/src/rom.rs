//! Shared Kid Icarus: Uprising (USA 1.0) ROM discovery and validation
//! shared by both Switch (`uprising_nx`) and PC (`uprising-recompiled`).

use std::path::Path;

use zakuro_fs::{FsError, Title};

use crate::KIU_USA_PROGRAM_ID;

#[derive(Debug, Clone)]
pub struct RomNotice {
    pub heading: String,
    pub paragraphs: Vec<String>,
}

impl RomNotice {
    pub fn missing_rom(dir_hint: &str) -> Self {
        Self {
            heading: "ROM Required".to_owned(),
            paragraphs: vec![
                "Select or provide a decrypted USA copy of Kid Icarus: Uprising (.cia, .3ds, or .cci):".to_owned(),
                dir_hint.to_owned(),
                String::new(),
                "Any filename works (for example: baserom.cia, baserom.3ds, or baserom.cci).".to_owned(),
                "Dump and decrypt your cartridge or eShop title on your 3DS using GodMode9.".to_owned(),
            ],
        }
    }

    pub fn encrypted_rom(name: &str, dir_hint: &str) -> Self {
        Self {
            heading: "Encrypted ROM Detected".to_owned(),
            paragraphs: vec![
                format!("Found '{name}', but it is still encrypted."),
                String::new(),
                "Please export a decrypted .cia, .3ds, or .cci using GodMode9 on your 3DS (NCSD/CIA image options -> Build CIA / Decrypt file) and try again:".to_owned(),
                dir_hint.to_owned(),
            ],
        }
    }

    pub fn wrong_title(name: &str, found_id: u64, dir_hint: &str) -> Self {
        let region_hint = match found_id {
            0x0004_0000_0003_0200 => " (this is the European release)",
            0x0004_0000_0003_0000 => " (this is the Japanese release)",
            _ => "",
        };
        Self {
            heading: "Unsupported ROM Version".to_owned(),
            paragraphs: vec![
                format!("Found '{name}' with Title ID {found_id:016X}{region_hint}."),
                String::new(),
                "This port requires the USA release of Kid Icarus: Uprising (Title ID 0004000000030100, CTR-P-AKDE).".to_owned(),
                dir_hint.to_owned(),
            ],
        }
    }

    pub fn unreadable(name: &str, err: &str) -> Self {
        Self {
            heading: "Could Not Open ROM".to_owned(),
            paragraphs: vec![
                format!("Failed to read '{name}': {err}"),
                String::new(),
                "Verify the file is a valid decrypted .cia, .3ds, or .cci dump.".to_owned(),
            ],
        }
    }
}

/// Validates that `path` is a decrypted USA Kid Icarus: Uprising ROM (`0004000000030100`).
pub fn validate_kiu_usa_rom(path: &Path, dir_hint: &str) -> Result<Title, RomNotice> {
    if !path.is_file() {
        return Err(RomNotice::missing_rom(dir_hint));
    }
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("rom")
        .to_owned();
    match Title::load(path) {
        Ok(title) if title.program_id() == KIU_USA_PROGRAM_ID => Ok(title),
        Ok(title) => Err(RomNotice::wrong_title(&name, title.program_id(), dir_hint)),
        Err(FsError::Encrypted(_) | FsError::EncryptedCia) => {
            Err(RomNotice::encrypted_rom(&name, dir_hint))
        }
        Err(err) => Err(RomNotice::unreadable(&name, &err.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_rom_notice_mentions_title_id_hint() {
        let wrong = RomNotice::wrong_title("test.3ds", 0x0004_0000_0003_0200, "/roms");
        assert!(wrong.paragraphs[0].contains("European release"));
        assert!(wrong.paragraphs[2].contains("0004000000030100"));
    }
}
