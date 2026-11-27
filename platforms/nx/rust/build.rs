use std::path::Path;
use std::process::Command;

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let repo_root = Path::new(&manifest_dir).join("../../..");

    let version_path = repo_root.join("VERSION");
    println!("cargo:rerun-if-changed={}", version_path.display());
    println!("cargo:rerun-if-env-changed=UNX_RELEASE_TAG");

    let git_dir = repo_root.join(".git");
    let head_path = git_dir.join("HEAD");
    if head_path.exists() {
        println!("cargo:rerun-if-changed={}", head_path.display());
        if let Ok(head) = std::fs::read_to_string(&head_path) {
            if let Some(ref_rel) = head.trim().strip_prefix("ref: ") {
                let ref_path = git_dir.join(ref_rel.trim());
                if ref_path.exists() {
                    println!("cargo:rerun-if-changed={}", ref_path.display());
                }
            }
        }
    }

    let version = std::fs::read_to_string(&version_path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| std::env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.1.0".into()));

    let sha = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .current_dir(&repo_root)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into());

    let dirty = Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=no"])
        .current_dir(&repo_root)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .is_some_and(|o| !o.stdout.is_empty());

    let release_tag = std::env::var("UNX_RELEASE_TAG")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    let label = if let Some(ref tag) = release_tag {
        tag.clone()
    } else if dirty {
        format!("v{version} ({sha}*)")
    } else {
        format!("v{version} ({sha})")
    };

    println!("cargo:rustc-env=UNX_VERSION={version}");
    println!("cargo:rustc-env=UNX_GIT_SHA={sha}");
    println!("cargo:rustc-env=UNX_RELEASE_TAG={}", release_tag.as_deref().unwrap_or(""));
    println!("cargo:rustc-env=UNX_VERSION_LABEL={label}");
}
