//! one Vulkan device for the renderer and the presenter, when they would
//! pick the same GPU, so the screens the renderer draws are shown straight
//! from its images rather than copied out to the host and up again.

use ash::vk;

/// a device, the instance it came from and its one queue, gone once the
/// last of the renderer and the presenter lets go of it. the queue is shared,
/// and each locks it, see queue_lock, to submit.
pub struct SharedDevice {
    pub(crate) _entry: ash::Entry,
    pub(crate) instance: ash::Instance,
    pub(crate) physical: vk::PhysicalDevice,
    pub(crate) family: u32,
    pub(crate) device: ash::Device,
    pub(crate) queue: vk::Queue,
    /// held while the queue is submitted to or presented on: the renderer
    /// works on a thread of its own, the presenter on the window's.
    pub(crate) queue_lock: std::sync::Arc<std::sync::Mutex<()>>,
    /// whether the device does logic ops, and counts pipeline statistics.
    pub(crate) logic_ops: bool,
    pub(crate) statistics: bool,
    /// VK_EXT_depth_range_unrestricted: a viewport depth range past 0..1,
    /// so a depth map the viewport can't otherwise hold stays out of the
    /// fragment shader (which writing depth would cost the early depth test)
    pub(crate) unrestricted_depth: bool,
    /// the blend state draws set as they go.
    pub(crate) dynamic: crate::raster::hardware::Dynamic,
    /// maximum supported sampler anisotropy (`1.0` when unsupported).
    pub(crate) max_anisotropy: f32,
}

impl SharedDevice {
    /// a device on the GPU the renderer would pick, with no presenter, to
    /// time showing the screens straight from the GPU without a window.
    pub fn new() -> Result<SharedDevice, String> {
        crate::raster::hardware::own_device()
    }
}

impl std::fmt::Debug for SharedDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedDevice").field("family", &self.family).finish_non_exhaustive()
    }
}

impl Drop for SharedDevice {
    fn drop(&mut self) {
        // SAFETY: whoever made something from the device destroyed it before
        // letting go, the surface of the instance went with the presenter
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// Ordered macOS MoltenVK / Vulkan loader candidate paths probed before
/// falling back to `ash::Entry::load()`.
#[cfg(any(target_os = "macos", test))]
fn macos_loader_candidates(exe_dir: Option<&std::path::Path>) -> Vec<std::path::PathBuf> {
    let mut candidates = Vec::with_capacity(7);
    if let Some(dir) = exe_dir {
        // Inside a `.app` bundle (`Contents/MacOS/../Frameworks/...`) — direct
        // MoltenVK loading requires no `VkLayer` / ICD JSON configuration.
        candidates.push(dir.join("../Frameworks/libMoltenVK.dylib"));
        candidates.push(dir.join("../Frameworks/libvulkan.1.dylib"));
        candidates.push(dir.join("libMoltenVK.dylib"));
    }
    candidates.push(std::path::PathBuf::from("libMoltenVK.dylib"));
    candidates.push(std::path::PathBuf::from("libvulkan.1.dylib"));
    candidates.push(std::path::PathBuf::from("/opt/homebrew/lib/libMoltenVK.dylib"));
    candidates.push(std::path::PathBuf::from("/usr/local/lib/libMoltenVK.dylib"));
    candidates
}

#[cfg(all(not(target_os = "horizon"), any(target_os = "macos", test)))]
fn load_vulkan_from_candidates(candidates: &[std::path::PathBuf]) -> Result<ash::Entry, String> {
    for path in candidates {
        // SAFETY: probing a Vulkan / MoltenVK shared library from standard bundle
        // or system paths.
        match unsafe { ash::Entry::load_from(path) } {
            Ok(entry) => {
                log::info!("loaded Vulkan library from {}", path.display());
                return Ok(entry);
            }
            Err(error) => {
                log::debug!("Vulkan library probe failed for {}: {error}", path.display());
            }
        }
    }
    // SAFETY: falls back to ash's default Vulkan loader search.
    let entry = unsafe { ash::Entry::load() }.map_err(|e| format!("no Vulkan loader, {e}"))?;
    log::info!("loaded Vulkan library via default loader");
    Ok(entry)
}

/// Loads the Vulkan entry points: static linked driver on Horizon, bundled
/// MoltenVK / Vulkan loader discovery on macOS, or the system loader elsewhere.
pub(crate) fn load_vulkan_entry() -> Result<ash::Entry, String> {
    #[cfg(target_os = "horizon")]
    {
        log::info!("loaded linked Horizon Vulkan driver");
        Ok(ash::Entry::linked())
    }
    #[cfg(target_os = "macos")]
    {
        let exe = std::env::current_exe().ok();
        let exe_dir = exe.as_deref().and_then(std::path::Path::parent);
        let candidates = macos_loader_candidates(exe_dir);
        load_vulkan_from_candidates(&candidates)
    }
    #[cfg(all(not(target_os = "horizon"), not(target_os = "macos")))]
    {
        // SAFETY: loads the system's Vulkan library, which has no other
        // requirements.
        let entry = unsafe { ash::Entry::load() }.map_err(|e| format!("no Vulkan loader, {e}"))?;
        log::info!("loaded Vulkan library via system loader");
        Ok(entry)
    }
}

/// Enables `VK_KHR_portability_enumeration` and returns
/// `vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR` when the loader
/// advertises the extension (required to enumerate MoltenVK devices on macOS).
pub(crate) fn enable_portability_enumeration(
    entry: &ash::Entry,
    extensions: &mut Vec<*const std::ffi::c_char>,
) -> vk::InstanceCreateFlags {
    // SAFETY: queries instance extensions from the loaded Vulkan entry.
    let available = unsafe { entry.enumerate_instance_extension_properties(None) }.unwrap_or_default();
    let portability = ash::khr::portability_enumeration::NAME;
    let supported = available
        .iter()
        .any(|ext| ext.extension_name_as_c_str() == Ok(portability));
    if supported {
        let already_present = extensions
            .iter()
            .any(|&ptr| unsafe { std::ffi::CStr::from_ptr(ptr) } == portability);
        if !already_present {
            extensions.push(portability.as_ptr());
        }
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::empty()
    }
}

/// Appends `VK_KHR_portability_subset` to `extensions` whenever the physical
/// device advertises it (required by the Vulkan spec for MoltenVK devices).
pub(crate) fn enable_portability_subset(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    extensions: &mut Vec<*const std::ffi::c_char>,
) {
    // SAFETY: `physical` came from `instance`.
    let available = unsafe { instance.enumerate_device_extension_properties(physical) }.unwrap_or_default();
    let portability = ash::khr::portability_subset::NAME;
    let supported = available
        .iter()
        .any(|ext| ext.extension_name_as_c_str() == Ok(portability));
    if supported {
        let already_present = extensions
            .iter()
            .any(|&ptr| unsafe { std::ffi::CStr::from_ptr(ptr) } == portability);
        if !already_present {
            extensions.push(portability.as_ptr());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::macos_loader_candidates;

    #[test]
    fn macos_loader_candidates_follow_required_probe_order() {
        let exe_dir = Path::new("/Applications/Zakuro.app/Contents/MacOS");
        let candidates = macos_loader_candidates(Some(exe_dir));
        assert_eq!(
            candidates,
            vec![
                PathBuf::from("/Applications/Zakuro.app/Contents/MacOS/../Frameworks/libMoltenVK.dylib"),
                PathBuf::from("/Applications/Zakuro.app/Contents/MacOS/../Frameworks/libvulkan.1.dylib"),
                PathBuf::from("/Applications/Zakuro.app/Contents/MacOS/libMoltenVK.dylib"),
                PathBuf::from("libMoltenVK.dylib"),
                PathBuf::from("libvulkan.1.dylib"),
                PathBuf::from("/opt/homebrew/lib/libMoltenVK.dylib"),
                PathBuf::from("/usr/local/lib/libMoltenVK.dylib"),
            ]
        );

        assert_eq!(
            macos_loader_candidates(None),
            vec![
                PathBuf::from("libMoltenVK.dylib"),
                PathBuf::from("libvulkan.1.dylib"),
                PathBuf::from("/opt/homebrew/lib/libMoltenVK.dylib"),
                PathBuf::from("/usr/local/lib/libMoltenVK.dylib"),
            ]
        );

        #[cfg(not(target_os = "horizon"))]
        {
            let _probe: fn(&[PathBuf]) -> Result<ash::Entry, String> = super::load_vulkan_from_candidates;
        }
    }
}
