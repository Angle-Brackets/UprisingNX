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
