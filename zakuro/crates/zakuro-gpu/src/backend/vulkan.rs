//! Vulkan presentation backend.

use std::ffi::CStr;
use std::mem::ManuallyDrop;
use std::sync::Arc;

use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use super::vulkan_overlay::OverlayPainter;
use super::{fuse_backdrops, place_slices, GpuScreen, Overlay, PresentError, Presenter, ScreenFilter, ScreenImage, ScreenLayout, Shown, Stereo, StereoMode, Viewport};
use crate::raster::hardware::{can_render, render_device};
use crate::SharedDevice;

const VERTEX_SPIRV: &[u8] = include_bytes!("../../shaders/present.vert.spv");
const FRAGMENT_SPIRV: &[u8] = include_bytes!("../../shaders/present.frag.spv");
const STEREO_FRAGMENT_SPIRV: &[u8] = include_bytes!("../../shaders/present_stereo.frag.spv");
const CAULDRON_FRAGMENT_SPIRV: &[u8] = include_bytes!("../../shaders/present_cauldron.frag.spv");
const CHAPTER_FRAGMENT_SPIRV: &[u8] = include_bytes!("../../shaders/present_chapter.frag.spv");
const HUB_FRAGMENT_SPIRV: &[u8] = include_bytes!("../../shaders/present_hub.frag.spv");

/// the bytes of push constants the fragment shader reads: where the screen
/// lies in its image, the bounds its samples keep in and the filter, and
/// for 3D how the eyes show and the right eye's place in its image.
const CONSTANTS: u32 = 36;
const STEREO_CONSTANTS: u32 = 80;
const CAULDRON_CONSTANTS: u32 = 80;
const CHAPTER_CONSTANTS: u32 = 80;
const HUB_CONSTANTS: u32 = 80;

/// how many frames may be recorded before waiting on the oldest.
const FRAMES_IN_FLIGHT: usize = 2;

/// a whole image, as the area of it a screen takes and the bounds its
/// samples keep within, for the fragment shader.
const WHOLE: [f32; 8] = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];

/// what the presenter's surface is made from.
#[derive(Clone, Copy)]
enum SurfaceSource {
    /// a desktop window, through ash-window.
    Window(raw_window_handle::RawDisplayHandle, raw_window_handle::RawWindowHandle),
    /// a Horizon NWindow.
    #[cfg(target_os = "horizon")]
    Vi(*mut std::ffi::c_void),
}

/// what a screen is drawn from in a frame.
#[derive(Clone, Copy)]
enum Source {
    Nothing,
    /// the image of its own the presenter uploads pixels to.
    Own,
    /// an image of the renderer's, on the device they share.
    Gpu(GpuScreen),
}

pub(super) fn fail(message: impl Into<String>) -> PresentError {
    PresentError::Backend(message.into())
}

pub(super) fn vk_fail(context: &str) -> impl Fn(vk::Result) -> PresentError + '_ {
    move |error| PresentError::Backend(format!("{context}: {error}"))
}

/// everything belonging to one emulated screen.
struct Screen {
    width: u32,
    height: u32,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    staging: vk::Buffer,
    staging_memory: vk::DeviceMemory,
    staging_mapping: *mut u8,
    staging_size: vk::DeviceSize,
    descriptor: vk::DescriptorSet,
    /// false until the image has been transitioned out of UNDEFINED.
    initialized: bool,
    /// set when new pixels were written into the staging buffer this frame.
    dirty: bool,
    /// the image holds what the screen showed last, not one of the renderer's.
    showing: bool,
}

pub struct VulkanPresenter {
    _entry: ash::Entry,
    instance: ash::Instance,
    surface_instance: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,

    physical_device: vk::PhysicalDevice,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    device: ash::Device,
    queue_family: u32,
    queue: vk::Queue,

    swapchain_device: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    surface_format: vk::SurfaceFormatKHR,
    extent: vk::Extent2D,
    image_views: Vec<vk::ImageView>,
    framebuffers: Vec<vk::Framebuffer>,

    render_pass: vk::RenderPass,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    /// what draws the top screen's two pictures in one, for 3D in red and
    /// cyan or on alternate rows, with a set per frame in flight.
    stereo_layout: vk::DescriptorSetLayout,
    stereo_pipeline_layout: vk::PipelineLayout,
    stereo_pipeline: vk::Pipeline,
    stereo_sets: Vec<vk::DescriptorSet>,
    cauldron_layout: vk::DescriptorSetLayout,
    cauldron_pipeline_layout: vk::PipelineLayout,
    cauldron_pipeline: vk::Pipeline,
    cauldron_sets: Vec<vk::DescriptorSet>,
    cauldron_state: (f32, f32, u32),
    chapter_pipeline_layout: vk::PipelineLayout,
    chapter_pipeline: vk::Pipeline,
    chapter_title_right: f32,
    weapon_sub: u8,
    pause_sub: u8,
    hub_pipeline_layout: vk::PipelineLayout,
    hub_pipeline: vk::Pipeline,
    sampler: vk::Sampler,
    /// a set per frame in flight and screen, pointed at the renderer's
    /// images when screens are drawn straight from them.
    gpu_sets: Vec<[vk::DescriptorSet; 3]>,
    /// the device and instance, when the renderer draws on them too, which
    /// go with the last of the two rather than with this.
    shared: Option<Arc<SharedDevice>>,
    /// whether a present waits for the display, as FIFO does.
    waits: bool,

    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available: Vec<vk::Semaphore>,
    render_finished: Vec<vk::Semaphore>,
    in_flight: Vec<vk::Fence>,
    frame: usize,

    /// the top screen, the bottom one and the top one's picture for the
    /// right eye.
    screens: [Screen; 3],
    /// what is drawn over the screens, none only while being dropped.
    overlay: Option<OverlayPainter>,
    window: (u32, u32),
    arrangement: ScreenLayout,
    top_width: f32,
    filter: ScreenFilter,
    integer: bool,
    stereo: Stereo,
    safe_area: f32,
    /// set when the swapchain no longer matches the window.
    stale: bool,
    /// consecutive Horizon swapchain acquire timeouts before rebuilding.
    acquire_timeouts: u32,
}

impl VulkanPresenter {
    pub fn new(
        window: &(impl HasDisplayHandle + HasWindowHandle),
        size: (u32, u32),
    ) -> Result<VulkanPresenter, PresentError> {
        let display = window
            .display_handle()
            .map_err(|e| fail(format!("no display handle: {e}")))?;
        let window_handle = window
            .window_handle()
            .map_err(|e| fail(format!("no window handle: {e}")))?;
        VulkanPresenter::create(SurfaceSource::Window(display.as_raw(), window_handle.as_raw()), size)
    }

    /// presents to a Horizon NWindow (libnx's nwindowGetDefault() for the
    /// application's own layer) through VK_NN_vi_surface.
    ///
    /// # Safety
    /// nwindow points to a live NWindow that outlives the presenter.
    #[cfg(target_os = "horizon")]
    pub unsafe fn new_vi(nwindow: *mut std::ffi::c_void, size: (u32, u32)) -> Result<VulkanPresenter, PresentError> {
        VulkanPresenter::create(SurfaceSource::Vi(nwindow), size)
    }

    fn create(source: SurfaceSource, size: (u32, u32)) -> Result<VulkanPresenter, PresentError> {
        let entry = crate::raster::hardware::vulkan_entry().map_err(fail)?;

        // 1.3, which the renderer needs to share the device
        let application = vk::ApplicationInfo::default()
            .application_name(c"Zakuro")
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(c"Zakuro")
            .api_version(vk::API_VERSION_1_3);

        let extensions = match source {
            SurfaceSource::Window(display, _) => ash_window::enumerate_required_extensions(display)
                .map_err(vk_fail("querying surface extensions"))?
                .to_vec(),
            #[cfg(target_os = "horizon")]
            SurfaceSource::Vi(_) => vec![ash::khr::surface::NAME.as_ptr(), ash::nn::vi_surface::NAME.as_ptr()],
        };

        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&application)
            .enabled_extension_names(&extensions);
        let instance = unsafe { entry.create_instance(&instance_info, None) }
            .map_err(vk_fail("creating the instance"))?;

        let surface_instance = ash::khr::surface::Instance::new(&entry, &instance);
        let surface = match source {
            SurfaceSource::Window(display, window) => unsafe {
                ash_window::create_surface(&entry, &instance, display, window, None)
            },
            #[cfg(target_os = "horizon")]
            SurfaceSource::Vi(nwindow) => unsafe {
                ash::nn::vi_surface::Instance::new(&entry, &instance)
                    .create_vi_surface(&vk::ViSurfaceCreateInfoNN::default().window(nwindow), None)
            },
        }
        .map_err(vk_fail("creating the surface"))?;

        let (physical_device, queue_family) =
            pick_device(&instance, &surface_instance, surface)?;

        let properties = unsafe { instance.get_physical_device_properties(physical_device) };
        let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) };

        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        // the renderer draws on this device too when it can, so the screens
        // it draws are shown from its images rather than copied out and up
        // again, ZAKURO_COPY_SCREENS keeps the copies
        let copy = std::env::var_os("ZAKURO_COPY_SCREENS").is_some();
        let swapchain_extension = [ash::khr::swapchain::NAME.as_ptr()];
        let shared = if copy || !can_render(&instance, physical_device, queue_family) {
            None
        } else {
            render_device(&entry, &instance, physical_device, queue_family, &swapchain_extension)
                .inspect_err(|error| log::warn!("could not make a device to share with the renderer, {error}"))
                .ok()
                .map(Arc::new)
        };
        // not destroyed on the way out of a failure below, which leaves what
        // was made from it, as everything else here does
        let shared = ManuallyDrop::new(shared);
        let device = match shared.as_deref() {
            Some(shared) => shared.device.clone(),
            None => {
                let priorities = [1.0f32];
                let queue_info = [vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(queue_family)
                    .queue_priorities(&priorities)];
                let device_info = vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queue_info)
                    .enabled_extension_names(&swapchain_extension);
                unsafe { instance.create_device(physical_device, &device_info, None) }
                    .map_err(vk_fail("creating the device"))?
            }
        };
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let sharing = if shared.is_some() { ", the renderer's device too" } else { "" };
        log::info!("Vulkan on {}{sharing}", name.to_string_lossy());
        // which of the extensions that would let a draw read the image it
        // draws into the driver has
        if let Ok(extensions) = unsafe { instance.enumerate_device_extension_properties(physical_device) } {
            let wanted = ["VK_EXT_attachment_feedback_loop_layout", "VK_KHR_dynamic_rendering_local_read", "VK_EXT_attachment_feedback_loop_dynamic_state", "VK_KHR_dynamic_rendering", "VK_EXT_rasterization_order_attachment_access"];
            let has: Vec<&str> = wanted
                .iter()
                .copied()
                .filter(|want| extensions.iter().any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) }.to_str() == Ok(*want)))
                .collect();
            log::info!("Vulkan device has, of the feedback-loop ones: {has:?} ({} extensions in all)", extensions.len());
        }

        let swapchain_device = ash::khr::swapchain::Device::new(&instance, &device);
        let surface_format = choose_format(&surface_instance, physical_device, surface)?;
        let render_pass = create_render_pass(&device, surface_format.format)?;

        let descriptor_layout = create_descriptor_layout(&device, 1)?;
        let (pipeline_layout, pipeline) =
            create_pipeline(&device, render_pass, descriptor_layout, FRAGMENT_SPIRV, CONSTANTS)?;
        let stereo_layout = create_descriptor_layout(&device, 2)?;
        let (stereo_pipeline_layout, stereo_pipeline) =
            create_pipeline(&device, render_pass, stereo_layout, STEREO_FRAGMENT_SPIRV, STEREO_CONSTANTS)?;
        let cauldron_layout = create_cauldron_descriptor_layout(&device)?;
        let (cauldron_pipeline_layout, cauldron_pipeline) =
            create_pipeline(&device, render_pass, cauldron_layout, CAULDRON_FRAGMENT_SPIRV, CAULDRON_CONSTANTS)?;
        let (chapter_pipeline_layout, chapter_pipeline) =
            create_pipeline(&device, render_pass, cauldron_layout, CHAPTER_FRAGMENT_SPIRV, CHAPTER_CONSTANTS)?;
        let (hub_pipeline_layout, hub_pipeline) =
            create_pipeline(&device, render_pass, cauldron_layout, HUB_FRAGMENT_SPIRV, HUB_CONSTANTS)?;
        let srgb = matches!(surface_format.format, vk::Format::B8G8R8A8_SRGB | vk::Format::R8G8B8A8_SRGB);
        let overlay = OverlayPainter::new(&device, render_pass, FRAMES_IN_FLIGHT, srgb)?;

        let sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    // linear, the screens come drawn at up to four times the
                    // console's resolution and get shrunk as often as grown
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )
        }
        .map_err(vk_fail("creating the sampler"))?;

        let descriptor_pool = unsafe {
            // a screen whose size changes is made again before the old one
            // goes and gives its set back, so there is room for both, then
            // the sets for the renderer's images and the 3D sets, which hold
            // two pictures each, plus Fiend's Cauldron's 2-image + 1-sampler sets
            let sets = 6 + 5 * FRAMES_IN_FLIGHT as u32;
            let sizes = [
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(6 + 5 * FRAMES_IN_FLIGHT as u32),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::SAMPLED_IMAGE)
                    .descriptor_count(2 * FRAMES_IN_FLIGHT as u32),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::SAMPLER)
                    .descriptor_count(FRAMES_IN_FLIGHT as u32),
            ];
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                    .pool_sizes(&sizes)
                    .max_sets(sets),
                None,
            )
        }
        .map_err(vk_fail("creating the descriptor pool"))?;

        let gpu_sets = unsafe {
            let layouts = [descriptor_layout; 3 * FRAMES_IN_FLIGHT];
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&layouts),
            )
        }
        .map_err(vk_fail("allocating descriptor sets"))?
        .chunks(3)
        .map(|sets| [sets[0], sets[1], sets[2]])
        .collect();
        let stereo_sets = unsafe {
            let layouts = [stereo_layout; FRAMES_IN_FLIGHT];
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&layouts),
            )
        }
        .map_err(vk_fail("allocating descriptor sets"))?;
        let cauldron_sets = unsafe {
            let layouts = [cauldron_layout; FRAMES_IN_FLIGHT];
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&layouts),
            )
        }
        .map_err(vk_fail("allocating cauldron descriptor sets"))?;

        let command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .map_err(vk_fail("creating the command pool"))?;

        let command_buffers = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(FRAMES_IN_FLIGHT as u32),
            )
        }
        .map_err(vk_fail("allocating command buffers"))?;

        let mut image_available = Vec::with_capacity(FRAMES_IN_FLIGHT);
        let mut render_finished = Vec::with_capacity(FRAMES_IN_FLIGHT);
        let mut in_flight = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for _ in 0..FRAMES_IN_FLIGHT {
            let semaphore_info = vk::SemaphoreCreateInfo::default();
            image_available.push(
                unsafe { device.create_semaphore(&semaphore_info, None) }
                    .map_err(vk_fail("creating a semaphore"))?,
            );
            render_finished.push(
                unsafe { device.create_semaphore(&semaphore_info, None) }
                    .map_err(vk_fail("creating a semaphore"))?,
            );
            in_flight.push(
                unsafe {
                    device.create_fence(
                        // signalled, so the first frame does not wait forever.
                        &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                        None,
                    )
                }
                .map_err(vk_fail("creating a fence"))?,
            );
        }

        // the screens start at their native sizes, upload grows them if a
        // title ever reports something different.
        let screens = [
            Screen::new(
                &device,
                &memory_properties,
                descriptor_pool,
                descriptor_layout,
                sampler,
                400,
                240,
            )?,
            Screen::new(
                &device,
                &memory_properties,
                descriptor_pool,
                descriptor_layout,
                sampler,
                320,
                240,
            )?,
            Screen::new(
                &device,
                &memory_properties,
                descriptor_pool,
                descriptor_layout,
                sampler,
                400,
                240,
            )?,
        ];

        let mut presenter = VulkanPresenter {
            _entry: entry,
            instance,
            surface_instance,
            surface,
            physical_device,
            memory_properties,
            device,
            queue_family,
            queue,
            swapchain_device,
            swapchain: vk::SwapchainKHR::null(),
            surface_format,
            extent: vk::Extent2D {
                width: size.0,
                height: size.1,
            },
            image_views: Vec::new(),
            framebuffers: Vec::new(),
            render_pass,
            descriptor_layout,
            descriptor_pool,
            pipeline_layout,
            pipeline,
            stereo_layout,
            stereo_pipeline_layout,
            stereo_pipeline,
            stereo_sets,
            cauldron_layout,
            cauldron_pipeline_layout,
            cauldron_pipeline,
            cauldron_sets,
            cauldron_state: (237.4, 0.48, 1),
            chapter_pipeline_layout,
            chapter_pipeline,
            chapter_title_right: 302.0,
            weapon_sub: 1,
            pause_sub: 2,
            hub_pipeline_layout,
            hub_pipeline,
            sampler,
            gpu_sets,
            shared: ManuallyDrop::into_inner(shared),
            waits: true,
            command_pool,
            command_buffers,
            image_available,
            render_finished,
            in_flight,
            frame: 0,
            screens,
            overlay: Some(overlay),
            window: size,
            arrangement: ScreenLayout::default(),
            top_width: super::TOP_WIDTH,
            filter: ScreenFilter::default(),
            integer: false,
            stereo: Stereo::default(),
            safe_area: 1.0,
            stale: false,
            acquire_timeouts: 0,
        };

        presenter.build_swapchain()?;
        Ok(presenter)
    }

    fn build_swapchain(&mut self) -> Result<(), PresentError> {
        let capabilities = unsafe {
            self.surface_instance
                .get_physical_device_surface_capabilities(self.physical_device, self.surface)
        }
        .map_err(vk_fail("querying surface capabilities"))?;

        // a current extent of u32::MAX means the surface takes its size from
        // the swapchain rather than the other way round; on Switch, NWindow
        // also takes the swapchain's extent through nwindowSetDimensions.
        let extent = if capabilities.current_extent.width == u32::MAX || cfg!(target_os = "horizon") {
            vk::Extent2D {
                width: self.window.0.clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: self.window.1.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        } else {
            capabilities.current_extent
        };

        if extent.width == 0 || extent.height == 0 {
            // the window is minimized, nothing to build.
            self.extent = extent;
            return Ok(());
        }

        let mut image_count = capabilities.min_image_count + 1;
        if capabilities.max_image_count > 0 && image_count > capabilities.max_image_count {
            image_count = capabilities.max_image_count;
        }

        let present_modes = unsafe {
            self.surface_instance
                .get_physical_device_surface_present_modes(self.physical_device, self.surface)
        }
        .map_err(vk_fail("querying present modes"))?;
        // mailbox where there is one, the frontend keeps the frames to the
        // console's rate itself, and a present that waits for the display,
        // as FIFO does on Mesa under Wayland, holds the next frame back with
        // it. FIFO is always there otherwise. ZAKURO_PRESENT_MODE=fifo,
        // mailbox or immediate picks one
        let wanted = match std::env::var("ZAKURO_PRESENT_MODE").as_deref() {
            Ok("fifo") => vec![vk::PresentModeKHR::FIFO],
            Ok("immediate") => vec![vk::PresentModeKHR::IMMEDIATE],
            Ok("mailbox") => vec![vk::PresentModeKHR::MAILBOX],
            _ => vec![vk::PresentModeKHR::MAILBOX, vk::PresentModeKHR::FIFO],
        };
        let present_mode = wanted.into_iter().find(|mode| present_modes.contains(mode)).unwrap_or(vk::PresentModeKHR::FIFO);
        self.waits = present_mode == vk::PresentModeKHR::FIFO;
        if self.swapchain == vk::SwapchainKHR::null() {
            let name = |mode: vk::PresentModeKHR| match mode {
                vk::PresentModeKHR::FIFO => "fifo".to_owned(),
                vk::PresentModeKHR::MAILBOX => "mailbox".to_owned(),
                vk::PresentModeKHR::IMMEDIATE => "immediate".to_owned(),
                vk::PresentModeKHR::FIFO_RELAXED => "fifo relaxed".to_owned(),
                other => format!("mode {}", other.as_raw()),
            };
            let all: Vec<String> = present_modes.iter().copied().map(name).collect();
            log::info!("presenting in {} mode, of {}", name(present_mode), all.join(", "));
        }

        let old = self.swapchain;
        let info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(image_count)
            .image_format(self.surface_format.format)
            .image_color_space(self.surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(old);

        let swapchain = unsafe { self.swapchain_device.create_swapchain(&info, None) }
            .map_err(vk_fail("creating the swapchain"))?;

        self.destroy_swapchain_resources();
        if old != vk::SwapchainKHR::null() {
            unsafe { self.swapchain_device.destroy_swapchain(old, None) };
            log::info!("rebuilt swapchain at {}x{}", extent.width, extent.height);
        }

        self.swapchain = swapchain;
        self.extent = extent;

        let images = unsafe { self.swapchain_device.get_swapchain_images(swapchain) }
            .map_err(vk_fail("getting swapchain images"))?;

        for image in images {
            let view = unsafe {
                self.device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(self.surface_format.format)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .level_count(1)
                                .layer_count(1),
                        ),
                    None,
                )
            }
            .map_err(vk_fail("creating a swapchain image view"))?;

            let attachments = [view];
            let framebuffer = unsafe {
                self.device.create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .render_pass(self.render_pass)
                        .attachments(&attachments)
                        .width(extent.width)
                        .height(extent.height)
                        .layers(1),
                    None,
                )
            }
            .map_err(vk_fail("creating a framebuffer"))?;

            self.image_views.push(view);
            self.framebuffers.push(framebuffer);
        }

        self.stale = false;
        self.acquire_timeouts = 0;
        Ok(())
    }

    fn destroy_swapchain_resources(&mut self) {
        unsafe {
            for framebuffer in self.framebuffers.drain(..) {
                self.device.destroy_framebuffer(framebuffer, None);
            }
            for view in self.image_views.drain(..) {
                self.device.destroy_image_view(view, None);
            }
        }
    }

    /// copies pixels into a screen's staging buffer, reallocating the image if
    /// the size changed.
    fn upload(&mut self, index: usize, image: &ScreenImage<'_>) -> Result<(), PresentError> {
        if self.screens[index].width != image.width || self.screens[index].height != image.height
        {
            let queue_lock = self.shared.as_ref().map(|shared| shared.queue_lock.clone());
            let _queue = queue_lock.as_ref().map(|lock| lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner()));
            unsafe { self.device.device_wait_idle() }.map_err(vk_fail("waiting for idle"))?;
            drop(_queue);
            let old = std::mem::replace(
                &mut self.screens[index],
                Screen::new(
                    &self.device,
                    &self.memory_properties,
                    self.descriptor_pool,
                    self.descriptor_layout,
                    self.sampler,
                    image.width,
                    image.height,
                )?,
            );
            old.destroy(&self.device, self.descriptor_pool);
        }

        let screen = &mut self.screens[index];
        let bytes = (image.width * image.height * 4) as usize;
        let count = bytes.min(image.pixels.len());
        // SAFETY: the staging buffer is host visible, persistently mapped and
        // at least bytes long, and no command using it is in flight because
        // the frame's fence was waited on first.
        unsafe {
            std::ptr::copy_nonoverlapping(image.pixels.as_ptr(), screen.staging_mapping, count);
        }
        screen.dirty = true;
        screen.showing = true;
        Ok(())
    }

    /// the device, for the renderer to draw on, when it can share it.
    pub fn shared_device(&self) -> Option<Arc<SharedDevice>> {
        self.shared.clone()
    }

    /// the window's size in pixels as the swapchain has it, what overlay
    /// positions are in
    pub fn extent(&self) -> (u32, u32) {
        (self.extent.width, self.extent.height)
    }

    /// active safe-area rectangle `(x, y, width, height)` inside `self.extent`,
    /// centered inside black borders when `safe_area < 1.0` for TVs with overscan.
    pub fn safe_rect(&self) -> (u32, u32, u32, u32) {
        let f = self.safe_area.clamp(0.80, 1.0);
        if (f - 1.0).abs() < 1e-4 || self.extent.width == 0 || self.extent.height == 0 {
            return (0, 0, self.extent.width, self.extent.height);
        }
        let w = (((self.extent.width as f32 * f).round() as u32) & !1).clamp(1, self.extent.width);
        let h = (((self.extent.height as f32 * f).round() as u32) & !1).clamp(1, self.extent.height);
        let ox = (self.extent.width - w) / 2;
        let oy = (self.extent.height - h) / 2;
        (ox, oy, w, h)
    }

    /// whether a present waits for the display, rather than handing the
    /// frame over and going on.
    pub fn waits_for_display(&self) -> bool {
        self.waits
    }

    /// what a screen is drawn from this frame, pixels uploaded to its own
    /// image or an image of the renderer's the frame's set points at.
    fn source(&mut self, index: usize, image: &ScreenImage<'_>) -> Result<Source, PresentError> {
        if let Some(gpu) = image.gpu {
            self.screens[index].showing = false;
            let info = [vk::DescriptorImageInfo::default()
                .sampler(self.sampler)
                .image_view(gpu.view)
                .image_layout(vk::ImageLayout::GENERAL)];
            let write = vk::WriteDescriptorSet::default()
                .dst_set(self.gpu_sets[self.frame][index])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&info);
            // SAFETY: the frame's fence was waited on, so no command using
            // the set is in flight
            unsafe { self.device.update_descriptor_sets(&[write], &[]) };
            return Ok(Source::Gpu(gpu));
        }
        if !image.is_empty() {
            self.upload(index, image)?;
            return Ok(Source::Own);
        }
        let screen = &self.screens[index];
        Ok(if screen.initialized && screen.showing { Source::Own } else { Source::Nothing })
    }

    /// points the frame's 3D set at the top screen's two pictures, wherever
    /// each is drawn from.
    fn write_stereo_set(&self, sources: &[Source; 3]) {
        let view = |index: usize| match sources[index] {
            Source::Gpu(gpu) => Some((gpu.view, vk::ImageLayout::GENERAL)),
            Source::Own => Some((self.screens[index].view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)),
            Source::Nothing => None,
        };
        let (Some(left), Some(right)) = (view(0), view(2)) else { return };
        let infos = [left, right]
            .map(|(view, layout)| [vk::DescriptorImageInfo::default().sampler(self.sampler).image_view(view).image_layout(layout)]);
        let writes = [0, 1].map(|binding| {
            vk::WriteDescriptorSet::default()
                .dst_set(self.stereo_sets[self.frame])
                .dst_binding(binding as u32)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&infos[binding])
        });
        // SAFETY: the frame's fence was waited on, so no command using the
        // set is in flight
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
    }

    /// points the frame's Fiend's Cauldron set at `top` (`0`), `bottom` (`1`),
    /// and `self.sampler` (`2`).
    fn write_cauldron_set(&self, sources: &[Source; 3]) -> bool {
        let view = |index: usize| match sources[index] {
            Source::Gpu(gpu) => Some((gpu.view, vk::ImageLayout::GENERAL)),
            Source::Own => Some((self.screens[index].view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)),
            Source::Nothing => None,
        };
        let (Some(top), Some(bot)) = (view(0), view(1)) else { return false };
        let top_info = [vk::DescriptorImageInfo::default().image_view(top.0).image_layout(top.1)];
        let bot_info = [vk::DescriptorImageInfo::default().image_view(bot.0).image_layout(bot.1)];
        let samp_info = [vk::DescriptorImageInfo::default().sampler(self.sampler)];
        let set = self.cauldron_sets[self.frame];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(&top_info),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(&bot_info),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .image_info(&samp_info),
        ];
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
        true
    }

    /// keeps the overlay's changes for the next frame drawn.
    fn defer(&mut self, overlay: &Overlay) {
        if let Some(painter) = &mut self.overlay {
            painter.defer(overlay);
        }
    }

    fn record(
        &self,
        command_buffer: vk::CommandBuffer,
        framebuffer: vk::Framebuffer,
        painter: &mut OverlayPainter,
        overlay: &Overlay,
        sources: [Source; 3],
        right: bool,
    ) -> Result<(), PresentError> {
        let device = &self.device;
        unsafe {
            device
                .begin_command_buffer(
                    command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .expect("begin command buffer");
        }
        painter.upload(device, &self.memory_properties, command_buffer, self.frame, overlay)?;
        unsafe {

            for screen in &self.screens {
                if !screen.dirty {
                    continue;
                }
                let old_layout = if screen.initialized {
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                } else {
                    vk::ImageLayout::UNDEFINED
                };
                transition(
                    device,
                    command_buffer,
                    screen.image,
                    old_layout,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                );

                device.cmd_copy_buffer_to_image(
                    command_buffer,
                    screen.staging,
                    screen.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[vk::BufferImageCopy::default()
                        .image_subresource(
                            vk::ImageSubresourceLayers::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .layer_count(1),
                        )
                        .image_extent(vk::Extent3D {
                            width: screen.width,
                            height: screen.height,
                            depth: 1,
                        })],
                );

                transition(
                    device,
                    command_buffer,
                    screen.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                );
            }

            // the renderer's work came first on the queue, its images are
            // read once what it wrote is there
            if sources.iter().any(|source| matches!(source, Source::Gpu(_))) {
                let barrier = vk::MemoryBarrier::default()
                    .src_access_mask(
                        vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                            | vk::AccessFlags::TRANSFER_WRITE
                            | vk::AccessFlags::SHADER_WRITE,
                    )
                    .dst_access_mask(vk::AccessFlags::SHADER_READ);
                device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                        | vk::PipelineStageFlags::TRANSFER
                        | vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[barrier],
                    &[],
                    &[],
                );
            }

            let clear = [vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            }];
            device.cmd_begin_render_pass(
                command_buffer,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(self.render_pass)
                    .framebuffer(framebuffer)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: self.extent,
                    })
                    .clear_values(&clear),
                vk::SubpassContents::INLINE,
            );
            let (ox, oy, safe_w, safe_h) = self.safe_rect();
            let shift = |vp: Viewport| Viewport {
                x: vp.x + ox as f32,
                y: vp.y + oy as f32,
                ..vp
            };
            if self.arrangement == ScreenLayout::FuseSplitDeck {
                let srgb = matches!(self.surface_format.format, vk::Format::B8G8R8A8_SRGB | vk::Format::R8G8B8A8_SRGB);
                let to_clear = |c: f32| {
                    if !srgb {
                        c
                    } else if c <= 0.04045 {
                        c / 12.92
                    } else {
                        ((c + 0.055) / 1.055).powf(2.4)
                    }
                };
                for (vp, [r, g, b, a]) in fuse_backdrops(safe_w, safe_h, self.integer) {
                    let vp = shift(vp);
                    let x0 = vp.x.max(0.0) as u32;
                    let y0 = vp.y.max(0.0) as u32;
                    let x1 = (vp.x + vp.width).max(0.0).min(self.extent.width as f32) as u32;
                    let y1 = (vp.y + vp.height).max(0.0).min(self.extent.height as f32) as u32;
                    if x1 <= x0 || y1 <= y0 {
                        continue;
                    }
                    let att = [vk::ClearAttachment {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        color_attachment: 0,
                        clear_value: vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [to_clear(r), to_clear(g), to_clear(b), a],
                            },
                        },
                    }];
                    let rect = [vk::ClearRect {
                        rect: vk::Rect2D {
                            offset: vk::Offset2D { x: x0 as i32, y: y0 as i32 },
                            extent: vk::Extent2D { width: x1 - x0, height: y1 - y0 },
                        },
                        base_array_layer: 0,
                        layer_count: 1,
                    }];
                    device.cmd_clear_attachments(command_buffer, &att, &rect);
                }
            }
            // each screen's set and place (or sub-crop) in its image
            let crop = |index: usize, uv: [f32; 4]| match sources[index] {
                Source::Nothing => None,
                Source::Own if uv == [0.0, 0.0, 1.0, 1.0] => Some((self.screens[index].descriptor, WHOLE)),
                Source::Own => {
                    let [sx, sy, sw, sh] = uv;
                    let half_u = 0.5 / self.screens[index].width.max(1) as f32;
                    let half_v = 0.5 / self.screens[index].height.max(1) as f32;
                    Some((
                        self.screens[index].descriptor,
                        [sx, sy, sw, sh, sx + half_u, sy + half_v, sx + sw - half_u, sy + sh - half_v],
                    ))
                }
                Source::Gpu(gpu) if uv == [0.0, 0.0, 1.0, 1.0] => {
                    let [x, y, width, height] = gpu.area;
                    let [left, top, right, bottom] = gpu.bounds;
                    Some((self.gpu_sets[self.frame][index], [x, y, width, height, left, top, right, bottom]))
                }
                Source::Gpu(gpu) => {
                    let [x, y, w, h] = gpu.area;
                    let [sx, sy, sw, sh] = uv;
                    let half_u = gpu.bounds[0] - x;
                    let half_v = gpu.bounds[1] - y;
                    let (ax, ay, aw, ah) = (x + sx * w, y + sy * h, sw * w, sh * h);
                    Some((
                        self.gpu_sets[self.frame][index],
                        [ax, ay, aw, ah, ax + half_u, ay + half_v, ax + aw - half_u, ay + ah - half_v],
                    ))
                }
            };
            if self.arrangement == ScreenLayout::FiendsCauldron {
                if let (Some((_, top_area)), Some((_, bot_area)), (Some(stage_vp), _)) = (
                    crop(0, [0.0, 0.0, 1.0, 1.0]),
                    crop(1, [0.0, 0.0, 1.0, 1.0]),
                    super::layout(safe_w, safe_h, ScreenLayout::FiendsCauldron, self.top_width, self.integer),
                ) {
                    let (cursor_x, opacity, hud_flags) = self.cauldron_state;
                    let mut constants = [0u8; CAULDRON_CONSTANTS as usize];
                    let mut off = 0usize;
                    for &c in top_area.iter().chain(bot_area.iter()) {
                        constants[off..off + 4].copy_from_slice(&c.to_le_bytes());
                        off += 4;
                    }
                    constants[off..off + 4].copy_from_slice(&cursor_x.to_le_bytes());
                    constants[off + 4..off + 8].copy_from_slice(&opacity.to_le_bytes());
                    constants[off + 8..off + 12].copy_from_slice(&hud_flags.to_le_bytes());
                    constants[off + 12..off + 16].copy_from_slice(&self.filter.mode().to_le_bytes());
                    set_viewport(device, command_buffer, shift(stage_vp), self.extent);
                    device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.cauldron_pipeline);
                    device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.cauldron_pipeline_layout,
                        0,
                        &[self.cauldron_sets[self.frame]],
                        &[],
                    );
                    device.cmd_push_constants(
                        command_buffer,
                        self.cauldron_pipeline_layout,
                        vk::ShaderStageFlags::FRAGMENT,
                        0,
                        &constants,
                    );
                    device.cmd_draw(command_buffer, 3, 1, 0, 0);
                }
            } else if self.arrangement == ScreenLayout::ChapterSelect {
                if let (Some((_, top_area)), Some((_, bot_area)), (Some(stage_vp), _)) = (
                    crop(0, [0.0, 0.0, 1.0, 1.0]),
                    crop(1, [0.0, 0.0, 1.0, 1.0]),
                    super::layout(safe_w, safe_h, ScreenLayout::ChapterSelect, self.top_width, self.integer),
                ) {
                    let mut constants = [0u8; CHAPTER_CONSTANTS as usize];
                    let mut off = 0usize;
                    for &c in top_area.iter().chain(bot_area.iter()) {
                        constants[off..off + 4].copy_from_slice(&c.to_le_bytes());
                        off += 4;
                    }
                    constants[off..off + 4].copy_from_slice(&self.chapter_title_right.to_le_bytes());
                    constants[off + 12..off + 16].copy_from_slice(&self.filter.mode().to_le_bytes());
                    set_viewport(device, command_buffer, shift(stage_vp), self.extent);
                    device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.chapter_pipeline);
                    device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.chapter_pipeline_layout,
                        0,
                        &[self.cauldron_sets[self.frame]],
                        &[],
                    );
                    device.cmd_push_constants(
                        command_buffer,
                        self.chapter_pipeline_layout,
                        vk::ShaderStageFlags::FRAGMENT,
                        0,
                        &constants,
                    );
                    device.cmd_draw(command_buffer, 3, 1, 0, 0);
                }
            } else if matches!(
                self.arrangement,
                ScreenLayout::ModeSelect
                    | ScreenLayout::SoloHub
                    | ScreenLayout::FileSelect
                    | ScreenLayout::WeaponSelect
                    | ScreenLayout::Pause
            ) {
                if let (Some((_, top_area)), Some((_, bot_area)), (Some(stage_vp), _)) = (
                    crop(0, [0.0, 0.0, 1.0, 1.0]),
                    crop(1, [0.0, 0.0, 1.0, 1.0]),
                    super::layout(safe_w, safe_h, self.arrangement, self.top_width, self.integer),
                ) {
                    let kind: i32 = match self.arrangement {
                        ScreenLayout::SoloHub => 1,
                        ScreenLayout::FileSelect => 2,
                        ScreenLayout::WeaponSelect => 3,
                        ScreenLayout::Pause => 4,
                        _ => 0,
                    };
                    let (pad0, pad1): (f32, f32) = if self.arrangement == ScreenLayout::WeaponSelect {
                        match self.weapon_sub {
                            2 => (0.0, 0.78),
                            3 => (4.0, 0.78),
                            4 => (4.0, -0.80),
                            _ => (4.0, 0.77),
                        }
                    } else if self.arrangement == ScreenLayout::Pause {
                        (self.pause_sub as f32, 0.0)
                    } else {
                        (0.0, 0.0)
                    };
                    let mut constants = [0u8; HUB_CONSTANTS as usize];
                    let mut off = 0usize;
                    for &c in top_area.iter().chain(bot_area.iter()) {
                        constants[off..off + 4].copy_from_slice(&c.to_le_bytes());
                        off += 4;
                    }
                    constants[off..off + 4].copy_from_slice(&pad0.to_le_bytes());
                    constants[off + 4..off + 8].copy_from_slice(&pad1.to_le_bytes());
                    constants[off + 8..off + 12].copy_from_slice(&kind.to_le_bytes());
                    constants[off + 12..off + 16].copy_from_slice(&self.filter.mode().to_le_bytes());
                    set_viewport(device, command_buffer, shift(stage_vp), self.extent);
                    device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.hub_pipeline);
                    device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.hub_pipeline_layout,
                        0,
                        &[self.cauldron_sets[self.frame]],
                        &[],
                    );
                    device.cmd_push_constants(
                        command_buffer,
                        self.hub_pipeline_layout,
                        vk::ShaderStageFlags::FRAGMENT,
                        0,
                        &constants,
                    );
                    device.cmd_draw(command_buffer, 3, 1, 0, 0);
                }
            } else {
                let places = place_slices(safe_w, safe_h, self.arrangement, self.top_width, self.integer, self.stereo, right);
                for slice in places {
                    let (viewport, shown, uv) = (slice.viewport, slice.shown, slice.uv);
                    let index = match shown {
                        Shown::Top | Shown::Both => 0,
                        Shown::Bottom => 1,
                        Shown::Right => 2,
                    };
                    let Some((descriptor, area)) = crop(index, uv) else { continue };
                    let mut constants = [0u8; STEREO_CONSTANTS as usize];
                    let mut off = 0usize;
                    for &c in &area {
                        constants[off..off + 4].copy_from_slice(&c.to_le_bytes());
                        off += 4;
                    }
                    constants[off..off + 4].copy_from_slice(&self.filter.mode().to_le_bytes());
                    off += 4;
                    let (pipeline, layout, set) = if shown == Shown::Both {
                        // then how the eyes show, and the right eye's place
                        let Some((_, right)) = crop(2, uv) else { continue };
                        let stereo = if self.stereo.mode == StereoMode::Anaglyph { 1i32 } else { 2 };
                        for word in [stereo, self.stereo.swap as i32, 0] {
                            constants[off..off + 4].copy_from_slice(&word.to_le_bytes());
                            off += 4;
                        }
                        for &c in &right {
                            constants[off..off + 4].copy_from_slice(&c.to_le_bytes());
                            off += 4;
                        }
                        (self.stereo_pipeline, self.stereo_pipeline_layout, self.stereo_sets[self.frame])
                    } else {
                        (self.pipeline, self.pipeline_layout, descriptor)
                    };
                    set_viewport(device, command_buffer, shift(viewport), self.extent);
                    device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);
                    device.cmd_bind_descriptor_sets(command_buffer, vk::PipelineBindPoint::GRAPHICS, layout, 0, &[set], &[]);
                    device.cmd_push_constants(command_buffer, layout, vk::ShaderStageFlags::FRAGMENT, 0, &constants[..off]);
                    device.cmd_draw(command_buffer, 3, 1, 0, 0);
                }
            }

            painter.draw(device, command_buffer, self.frame, overlay, self.extent);
            device.cmd_end_render_pass(command_buffer);
            device.end_command_buffer(command_buffer).expect("end command buffer");
        }
        Ok(())
    }
}

impl Presenter for VulkanPresenter {
    fn name(&self) -> &'static str {
        "vulkan"
    }

    fn present(
        &mut self,
        top: ScreenImage<'_>,
        bottom: ScreenImage<'_>,
        right: ScreenImage<'_>,
        overlay: &Overlay,
    ) -> Result<(), PresentError> {
        if self.stale {
            // the swapchain no longer matches the window. rebuilding it is
            // what lets a minimized window come back: while minimized the
            // surface has no size, nothing is built, and stale stays set
            // until the window is restored to a size again, so the guard
            // below for an empty extent must not come first.
            let queue_lock = self.shared.as_ref().map(|shared| shared.queue_lock.clone());
            let _queue = queue_lock.as_ref().map(|lock| lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner()));
            unsafe { self.device.device_wait_idle() }.map_err(vk_fail("waiting for idle"))?;
            drop(_queue);
            self.build_swapchain()?;
            if self.extent.width == 0 || self.extent.height == 0 {
                // still minimized, keep waiting for a size.
                self.defer(overlay);
                return Ok(());
            }
        }
        if self.extent.width == 0 || self.extent.height == 0 {
            self.defer(overlay);
            return Ok(());
        }

        // Finite wait on Horizon so `present()` returns to `appletMainLoop()` on
        // focus loss, sleep/wake, or a missed `NWindow` buffer-queue wakeup
        // instead of blocking forever inside `wsi_hos_acquire_next_image`.
        #[cfg(target_os = "horizon")]
        const PRESENT_WAIT_NS: u64 = 250_000_000;
        #[cfg(not(target_os = "horizon"))]
        const PRESENT_WAIT_NS: u64 = u64::MAX;

        let frame = self.frame;
        let fence = self.in_flight[frame];
        match unsafe { self.device.wait_for_fences(&[fence], true, PRESENT_WAIT_NS) } {
            Ok(()) => {}
            Err(vk::Result::TIMEOUT) => {
                self.defer(overlay);
                return Ok(());
            }
            Err(error) => {
                self.defer(overlay);
                return Err(vk_fail("waiting for a fence")(error));
            }
        }

        let acquired = unsafe {
            self.swapchain_device.acquire_next_image(
                self.swapchain,
                PRESENT_WAIT_NS,
                self.image_available[frame],
                vk::Fence::null(),
            )
        };
        let (image_index, suboptimal) = match acquired {
            Ok(result) => {
                self.acquire_timeouts = 0;
                result
            }
            Err(vk::Result::TIMEOUT | vk::Result::NOT_READY) => {
                self.defer(overlay);
                self.acquire_timeouts = self.acquire_timeouts.saturating_add(1);
                if self.acquire_timeouts >= 4 {
                    log::warn!(
                        "swapchain acquire timed out {} times, rebuilding swapchain",
                        self.acquire_timeouts
                    );
                    self.acquire_timeouts = 0;
                    self.stale = true;
                    return Err(PresentError::OutOfDate);
                }
                return Ok(());
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.acquire_timeouts = 0;
                self.stale = true;
                self.defer(overlay);
                return Err(PresentError::OutOfDate);
            }
            Err(error) => {
                self.defer(overlay);
                return Err(vk_fail("acquiring an image")(error));
            }
        };

        // only reset the fence once we know we are going to submit.
        unsafe {
            self.device
                .reset_fences(&[fence])
                .map_err(vk_fail("resetting a fence"))?;
        }

        for screen in &mut self.screens {
            screen.dirty = false;
        }
        // the right eye's picture shows only in a frame that has one
        let has_right = !right.is_empty();
        let sources = [self.source(0, &top)?, self.source(1, &bottom)?, self.source(2, &right)?];
        if has_right && matches!(self.stereo.mode, StereoMode::Anaglyph | StereoMode::Interlaced) {
            self.write_stereo_set(&sources);
        }
        if matches!(
            self.arrangement,
            ScreenLayout::FiendsCauldron
                | ScreenLayout::ChapterSelect
                | ScreenLayout::ModeSelect
                | ScreenLayout::SoloHub
                | ScreenLayout::FileSelect
                | ScreenLayout::WeaponSelect
                | ScreenLayout::Pause
        ) {
            self.write_cauldron_set(&sources);
        }

        let command_buffer = self.command_buffers[frame];
        unsafe {
            self.device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .map_err(vk_fail("resetting a command buffer"))?;
        }
        // the painter is taken out while it records, it and the presenter
        // both being borrowed
        let mut painter = self.overlay.take().expect("the overlay painter lives as long as the presenter");
        painter.begin_frame(&self.device, frame);
        let recorded = self.record(command_buffer, self.framebuffers[image_index as usize], &mut painter, overlay, sources, has_right);
        painter.free(frame, overlay);
        self.overlay = Some(painter);
        recorded?;

        for screen in &mut self.screens {
            if screen.dirty {
                screen.initialized = true;
            }
        }

        let wait = [self.image_available[frame]];
        let signal = [self.render_finished[frame]];
        let stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let buffers = [command_buffer];
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&wait)
            .wait_dst_stage_mask(&stages)
            .command_buffers(&buffers)
            .signal_semaphores(&signal);
        // the renderer submits on its own thread to the same queue
        let queue_lock = self.shared.as_ref().map(|shared| shared.queue_lock.clone());
        let _queue = queue_lock.as_ref().map(|lock| lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner()));
        let held = std::time::Instant::now();
        unsafe {
            self.device
                .queue_submit(self.queue, &[submit], fence)
                .map_err(vk_fail("submitting"))?;
        }

        let swapchains = [self.swapchain];
        let indices = [image_index];
        let present = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal)
            .swapchains(&swapchains)
            .image_indices(&indices);
        let result = unsafe { self.swapchain_device.queue_present(self.queue, &present) };
        drop(_queue);
        crate::stats::QUEUE_HELD_NS.fetch_add(held.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);

        self.frame = (self.frame + 1) % FRAMES_IN_FLIGHT;

        match result {
            Ok(false) if !suboptimal => Ok(()),
            Ok(_) => {
                self.stale = true;
                Ok(())
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.stale = true;
                Err(PresentError::OutOfDate)
            }
            Err(error) => Err(vk_fail("presenting")(error)),
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        let window = (width.max(1), height.max(1));
        if self.window != window {
            self.window = window;
            self.stale = true;
        }
    }

    fn set_layout(&mut self, arrangement: ScreenLayout) {
        self.arrangement = arrangement;
    }

    fn set_top_width(&mut self, width: f32) {
        self.top_width = width;
    }

    fn set_scaling(&mut self, filter: ScreenFilter, integer: bool) {
        self.filter = filter;
        self.integer = integer;
    }

    fn set_stereo(&mut self, stereo: Stereo) {
        self.stereo = stereo;
    }

    fn set_cauldron_state(&mut self, cursor_x: f32, opacity: f32, hud_flags: u32) {
        self.cauldron_state = (cursor_x, opacity, hud_flags);
    }

    fn set_chapter_state(&mut self, title_right: f32) {
        self.chapter_title_right = title_right;
    }

    fn set_weapon_state(&mut self, sub: u8) {
        self.weapon_sub = sub;
    }

    fn set_pause_state(&mut self, sub: u8) {
        self.pause_sub = sub;
    }

    fn set_safe_area(&mut self, factor: f32) {
        self.safe_area = factor.clamp(0.80, 1.0);
    }
}

impl Drop for VulkanPresenter {
    fn drop(&mut self) {
        let queue_lock = self.shared.as_ref().map(|shared| shared.queue_lock.clone());
        let _queue = queue_lock.as_ref().map(|lock| lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner()));
        unsafe {
            let _ = self.device.device_wait_idle();
        }
        drop(_queue);
        unsafe {
            for screen in std::mem::replace(&mut self.screens, [Screen::null(), Screen::null(), Screen::null()]) {
                screen.destroy(&self.device, self.descriptor_pool);
            }
            self.destroy_swapchain_resources();
            if self.swapchain != vk::SwapchainKHR::null() {
                self.swapchain_device.destroy_swapchain(self.swapchain, None);
            }
            for semaphore in self.image_available.drain(..) {
                self.device.destroy_semaphore(semaphore, None);
            }
            for semaphore in self.render_finished.drain(..) {
                self.device.destroy_semaphore(semaphore, None);
            }
            for fence in self.in_flight.drain(..) {
                self.device.destroy_fence(fence, None);
            }
            if let Some(mut painter) = self.overlay.take() {
                painter.destroy(&self.device);
            }
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_sampler(self.sampler, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_pipeline(self.stereo_pipeline, None);
            self.device.destroy_pipeline_layout(self.stereo_pipeline_layout, None);
            self.device.destroy_descriptor_set_layout(self.stereo_layout, None);
            self.device.destroy_pipeline(self.hub_pipeline, None);
            self.device.destroy_pipeline_layout(self.hub_pipeline_layout, None);
            self.device.destroy_pipeline(self.chapter_pipeline, None);
            self.device.destroy_pipeline_layout(self.chapter_pipeline_layout, None);
            self.device.destroy_pipeline(self.cauldron_pipeline, None);
            self.device.destroy_pipeline_layout(self.cauldron_pipeline_layout, None);
            self.device.destroy_descriptor_set_layout(self.cauldron_layout, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);
            // a shared device and its instance go with the last of the
            // renderer and this, the surface before
            if self.shared.is_none() {
                self.device.destroy_device(None);
            }
            self.surface_instance.destroy_surface(self.surface, None);
            if self.shared.is_none() {
                self.instance.destroy_instance(None);
            }
        }
        let _ = self.queue_family;
    }
}

// ---------------------------------------------------------------------------
// Setup helpers
// ---------------------------------------------------------------------------

fn pick_device(
    instance: &ash::Instance,
    surface_instance: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32), PresentError> {
    let devices = unsafe { instance.enumerate_physical_devices() }
        .map_err(vk_fail("enumerating devices"))?;

    let mut best: Option<(vk::PhysicalDevice, u32, u32)> = None;
    for device in devices {
        let families = unsafe { instance.get_physical_device_queue_family_properties(device) };
        for (index, family) in families.iter().enumerate() {
            if !family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                continue;
            }
            let supported = unsafe {
                surface_instance.get_physical_device_surface_support(
                    device,
                    index as u32,
                    surface,
                )
            }
            .unwrap_or(false);
            if !supported {
                continue;
            }
            // prefer a discrete GPU, then integrated, then anything.
            let properties = unsafe { instance.get_physical_device_properties(device) };
            let score = match properties.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 3,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
                _ => 1,
            };
            if best.is_none_or(|(_, _, current)| score > current) {
                best = Some((device, index as u32, score));
            }
            break;
        }
    }

    best.map(|(device, family, _)| (device, family))
        .ok_or_else(|| fail("no Vulkan device can present to this window"))
}

fn choose_format(
    surface_instance: &ash::khr::surface::Instance,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
) -> Result<vk::SurfaceFormatKHR, PresentError> {
    let formats = unsafe {
        surface_instance.get_physical_device_surface_formats(physical_device, surface)
    }
    .map_err(vk_fail("querying surface formats"))?;

    // the emulator produces straight sRGB-ish bytes, so an UNORM target keeps
    // them looking the way the console does.
    formats
        .iter()
        .copied()
        .find(|f| {
            f.format == vk::Format::B8G8R8A8_UNORM
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .or_else(|| formats.first().copied())
        .ok_or_else(|| fail("the surface reports no formats"))
}

fn create_render_pass(
    device: &ash::Device,
    format: vk::Format,
) -> Result<vk::RenderPass, PresentError> {
    let attachments = [vk::AttachmentDescription::default()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)];

    let color = [vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
    let subpasses = [vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color)];

    let dependencies = [vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];

    unsafe {
        device.create_render_pass(
            &vk::RenderPassCreateInfo::default()
                .attachments(&attachments)
                .subpasses(&subpasses)
                .dependencies(&dependencies),
            None,
        )
    }
    .map_err(vk_fail("creating the render pass"))
}

/// a set of pictures the fragment shader samples, one or the top screen's
/// two for 3D.
fn create_descriptor_layout(
    device: &ash::Device,
    pictures: u32,
) -> Result<vk::DescriptorSetLayout, PresentError> {
    let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..pictures)
        .map(|binding| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        })
        .collect();
    unsafe {
        device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )
    }
    .map_err(vk_fail("creating the descriptor set layout"))
}

fn create_cauldron_descriptor_layout(
    device: &ash::Device,
) -> Result<vk::DescriptorSetLayout, PresentError> {
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    ];
    unsafe {
        device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )
    }
    .map_err(vk_fail("creating the cauldron descriptor set layout"))
}

fn create_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    descriptor_layout: vk::DescriptorSetLayout,
    fragment_spirv: &[u8],
    constants_size: u32,
) -> Result<(vk::PipelineLayout, vk::Pipeline), PresentError> {
    let vertex = create_shader_module(device, VERTEX_SPIRV)?;
    let fragment = create_shader_module(device, fragment_spirv)?;

    let layouts = [descriptor_layout];
    let constants = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .size(constants_size)];
    let pipeline_layout = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&layouts)
                .push_constant_ranges(&constants),
            None,
        )
    }
    .map_err(vk_fail("creating the pipeline layout"))?;

    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment)
            .name(c"main"),
    ];

    // the vertex shader builds its own triangle, so there is no input state.
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)];
    let blend =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0);

    let pipelines = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
    }
    .map_err(|(_, error)| vk_fail("creating the pipeline")(error))?;

    unsafe {
        device.destroy_shader_module(vertex, None);
        device.destroy_shader_module(fragment, None);
    }

    Ok((pipeline_layout, pipelines[0]))
}

pub(super) fn create_shader_module(
    device: &ash::Device,
    spirv: &[u8],
) -> Result<vk::ShaderModule, PresentError> {
    if !spirv.len().is_multiple_of(4) {
        return Err(fail("SPIR-V is not a whole number of words"));
    }
    let words: Vec<u32> = spirv
        .as_chunks::<4>().0.iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&words),
            None,
        )
    }
    .map_err(vk_fail("creating a shader module"))
}

pub(super) fn find_memory_type(
    properties: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    wanted: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..properties.memory_type_count).find(|&i| {
        type_bits & (1 << i) != 0
            && properties.memory_types[i as usize]
                .property_flags
                .contains(wanted)
    })
}

pub(super) fn transition(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    from: vk::ImageLayout,
    to: vk::ImageLayout,
) {
    let (src_access, dst_access, src_stage, dst_stage) = match (from, to) {
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
        ),
        (vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
            vk::AccessFlags::SHADER_READ,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::PipelineStageFlags::TRANSFER,
        ),
        _ => (
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        ),
    };

    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(from)
        .new_layout(to)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1),
        )
        .src_access_mask(src_access)
        .dst_access_mask(dst_access);

    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}

fn set_viewport(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    viewport: Viewport,
    extent: vk::Extent2D,
) {
    unsafe {
        device.cmd_set_viewport(
            command_buffer,
            0,
            &[vk::Viewport {
                x: viewport.x,
                y: viewport.y,
                width: viewport.width,
                height: viewport.height,
                min_depth: 0.0,
                max_depth: 1.0,
            }],
        );
        // the scissor has to be clamped to the framebuffer or validation
        // complains when the window is smaller than the layout wants.
        let x = viewport.x.max(0.0) as i32;
        let y = viewport.y.max(0.0) as i32;
        let width = (viewport.width as u32).min(extent.width.saturating_sub(x as u32));
        let height = (viewport.height as u32).min(extent.height.saturating_sub(y as u32));
        device.cmd_set_scissor(
            command_buffer,
            0,
            &[vk::Rect2D {
                offset: vk::Offset2D { x, y },
                extent: vk::Extent2D { width, height },
            }],
        );
    }
}

impl Screen {
    fn null() -> Screen {
        Screen {
            width: 0,
            height: 0,
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            staging: vk::Buffer::null(),
            staging_memory: vk::DeviceMemory::null(),
            staging_mapping: std::ptr::null_mut(),
            staging_size: 0,
            descriptor: vk::DescriptorSet::null(),
            initialized: false,
            dirty: false,
            showing: false,
        }
    }

    fn new(
        device: &ash::Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        pool: vk::DescriptorPool,
        layout: vk::DescriptorSetLayout,
        sampler: vk::Sampler,
        width: u32,
        height: u32,
    ) -> Result<Screen, PresentError> {
        let size = (width as vk::DeviceSize) * (height as vk::DeviceSize) * 4;

        let image = unsafe {
            device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
        }
        .map_err(vk_fail("creating a screen image"))?;

        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let type_index = find_memory_type(
            memory_properties,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .ok_or_else(|| fail("no device-local memory for a screen image"))?;
        let memory = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(type_index),
                None,
            )
        }
        .map_err(vk_fail("allocating image memory"))?;
        unsafe { device.bind_image_memory(image, memory, 0) }
            .map_err(vk_fail("binding image memory"))?;

        let view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1),
                    ),
                None,
            )
        }
        .map_err(vk_fail("creating a screen image view"))?;

        let staging = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }
        .map_err(vk_fail("creating a staging buffer"))?;

        let staging_requirements = unsafe { device.get_buffer_memory_requirements(staging) };
        let staging_type = find_memory_type(
            memory_properties,
            staging_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .ok_or_else(|| fail("no host-visible memory for a staging buffer"))?;
        let staging_memory = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(staging_requirements.size)
                    .memory_type_index(staging_type),
                None,
            )
        }
        .map_err(vk_fail("allocating staging memory"))?;
        unsafe { device.bind_buffer_memory(staging, staging_memory, 0) }
            .map_err(vk_fail("binding staging memory"))?;

        // mapped once and kept mapped, the buffer is written every frame.
        let staging_mapping = unsafe {
            device.map_memory(
                staging_memory,
                0,
                staging_requirements.size,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(vk_fail("mapping staging memory"))? as *mut u8;

        let layouts = [layout];
        let descriptors = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts),
            )
        }
        .map_err(vk_fail("allocating a descriptor set"))?;
        let descriptor = descriptors[0];

        let image_info = [vk::DescriptorImageInfo::default()
            .sampler(sampler)
            .image_view(view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        unsafe {
            device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(descriptor)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&image_info)],
                &[],
            );
        }

        Ok(Screen {
            width,
            height,
            image,
            memory,
            view,
            staging,
            staging_memory,
            staging_mapping,
            staging_size: staging_requirements.size,
            descriptor,
            initialized: false,
            dirty: false,
            showing: false,
        })
    }

    fn destroy(self, device: &ash::Device, pool: vk::DescriptorPool) {
        if self.image == vk::Image::null() {
            return;
        }
        unsafe {
            let _ = device.free_descriptor_sets(pool, &[self.descriptor]);
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
            device.unmap_memory(self.staging_memory);
            device.destroy_buffer(self.staging, None);
            device.free_memory(self.staging_memory, None);
        }
        let _ = self.staging_size;
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn cauldron_shader_compiles_to_valid_spirv() {
        use naga::back::spv;
        use naga::front::glsl;
        use naga::valid::{Capabilities, ValidationFlags, Validator};

        let src_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders/present_cauldron.frag");
        let spv_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders/present_cauldron.frag.spv");
        let source = std::fs::read_to_string(&src_path).expect("read present_cauldron.frag");
        let module = glsl::Frontend::default()
            .parse(&glsl::Options::from(naga::ShaderStage::Fragment), &source)
            .unwrap_or_else(|e| panic!("GLSL parse failed:\n{}", e.emit_to_string(&source)));
        let info = Validator::new(ValidationFlags::all(), Capabilities::IMMEDIATES)
            .validate(&module)
            .unwrap_or_else(|e| panic!("Naga validation failed:\n{}", e.emit_to_string(&source)));
        let options = spv::Options {
            lang_version: (1, 0),
            flags: spv::WriterFlags::empty(),
            force_loop_bounding: false,
            ..spv::Options::default()
        };
        let pipeline = spv::PipelineOptions {
            shader_stage: naga::ShaderStage::Fragment,
            entry_point: "main".into(),
        };
        let words = spv::write_vec(&module, &info, &options, Some(&pipeline)).expect("emit SPIR-V");
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(words[0], 0x0723_0203);
        if std::fs::read(&spv_path).ok().as_deref() != Some(bytes.as_slice()) {
            std::fs::write(&spv_path, &bytes).expect("write present_cauldron.frag.spv");
        }
    }

    #[test]
    fn chapter_shader_compiles_to_valid_spirv() {
        use naga::back::spv;
        use naga::front::glsl;
        use naga::valid::{Capabilities, ValidationFlags, Validator};

        let src_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders/present_chapter.frag");
        let spv_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders/present_chapter.frag.spv");
        let source = std::fs::read_to_string(&src_path).expect("read present_chapter.frag");
        let module = glsl::Frontend::default()
            .parse(&glsl::Options::from(naga::ShaderStage::Fragment), &source)
            .unwrap_or_else(|e| panic!("GLSL parse failed:\n{}", e.emit_to_string(&source)));
        let info = Validator::new(ValidationFlags::all(), Capabilities::IMMEDIATES)
            .validate(&module)
            .unwrap_or_else(|e| panic!("Naga validation failed:\n{}", e.emit_to_string(&source)));
        let options = spv::Options {
            lang_version: (1, 0),
            flags: spv::WriterFlags::empty(),
            force_loop_bounding: false,
            ..spv::Options::default()
        };
        let pipeline = spv::PipelineOptions {
            shader_stage: naga::ShaderStage::Fragment,
            entry_point: "main".into(),
        };
        let words = spv::write_vec(&module, &info, &options, Some(&pipeline)).expect("emit SPIR-V");
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(words[0], 0x0723_0203);
        if std::fs::read(&spv_path).ok().as_deref() != Some(bytes.as_slice()) {
            std::fs::write(&spv_path, &bytes).expect("write present_chapter.frag.spv");
        }
    }

    #[test]
    fn hub_shader_compiles_to_valid_spirv() {
        use naga::back::spv;
        use naga::front::glsl;
        use naga::valid::{Capabilities, ValidationFlags, Validator};

        let src_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders/present_hub.frag");
        let spv_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders/present_hub.frag.spv");
        let source = std::fs::read_to_string(&src_path).expect("read present_hub.frag");
        let module = glsl::Frontend::default()
            .parse(&glsl::Options::from(naga::ShaderStage::Fragment), &source)
            .unwrap_or_else(|e| panic!("GLSL parse failed:\n{}", e.emit_to_string(&source)));
        let info = Validator::new(ValidationFlags::all(), Capabilities::IMMEDIATES)
            .validate(&module)
            .unwrap_or_else(|e| panic!("Naga validation failed:\n{}", e.emit_to_string(&source)));
        let options = spv::Options {
            lang_version: (1, 0),
            flags: spv::WriterFlags::empty(),
            force_loop_bounding: false,
            ..spv::Options::default()
        };
        let pipeline = spv::PipelineOptions {
            shader_stage: naga::ShaderStage::Fragment,
            entry_point: "main".into(),
        };
        let words = spv::write_vec(&module, &info, &options, Some(&pipeline)).expect("emit SPIR-V");
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(words[0], 0x0723_0203);
        if std::fs::read(&spv_path).ok().as_deref() != Some(bytes.as_slice()) {
            std::fs::write(&spv_path, &bytes).expect("write present_hub.frag.spv");
        }
    }
}
