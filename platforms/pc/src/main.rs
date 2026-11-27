//! Entry point for Kid Icarus: Uprising Recompiled on PC / Steam Deck.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

const WINDOW_ICON_JPG: &[u8] = include_bytes!("../../nx/c/icon.jpg");

fn main() {
    zakuro::run_kiu_with_icon(Some(WINDOW_ICON_JPG));
}
