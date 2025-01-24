// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Simple example.

use image::GenericImageView;
use std::sync::Arc;
use tilelib::types::{Display, DrawInstance, Layer, Sprite, Texture};
use wgpu::hal::auxil::db;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes};

use std::time::{Duration, Instant};

struct SimpleVelloApp {
    display: Option<tilelib::types::Display>,
    window: Option<Arc<winit::window::Window>>,

    start_time: Option<Instant>,
    last_time: Option<Instant>,
    interval: spin_sleep_util::Interval,

    textures: [Texture; 2],
}

impl ApplicationHandler for SimpleVelloApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Get the winit window cached in a previous Suspended event or else create a new window
        let window = Arc::new(
            event_loop
                .create_window(WindowAttributes::default())
                .unwrap(),
        );

        self.display = Some(Display::from_winit(window.clone()));
        self.window = Some(window);
        self.start_time = Some(Instant::now());
        self.last_time = self.start_time;
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            // Exit the event loop when a close is requested (e.g. window's close button is pressed)
            WindowEvent::CloseRequested => event_loop.exit(),

            // Resize the surface when the window is resized
            WindowEvent::Resized(size) => {
                self.display.as_mut().unwrap().resize(size);
            }

            // This is where all the rendering happens
            WindowEvent::RedrawRequested => {
                self.interval.tick();
                let start_render_time = Instant::now();

                let renderer = self.display.as_mut().unwrap().get_renderer();

                let passed = self.start_time.unwrap().elapsed();
                let fps =
                    Duration::from_secs(1).div_duration_f32(self.last_time.unwrap().elapsed());
                self.last_time = Some(Instant::now());

                dbg!(fps);

                for i in 0..1 {
                    let mut layer = Layer::square_tile_grid(0.25 * (i as f32 + 1.0));

                    for _ in 0..10 {
                        layer.draw_sprite(
                            &Sprite::new(self.textures[0 + i].clone()),
                            DrawInstance {
                                position: [
                                    0.0 + (passed.as_millis() as f32 % 100000.0) / 100000.0,
                                    0.0,
                                ],
                                size: [1.0, 1.0],
                                animation_frame: ((passed.as_millis() / 100) % 10) as u32,
                            },
                        );

                        layer.draw_sprite(
                            &Sprite::new(self.textures[1 - i].clone()),
                            DrawInstance {
                                position: [1.0, 1.0],
                                size: [1.0, 1.0],
                                animation_frame: 0,
                            },
                        );
                    }

                    renderer.draw(&layer);
                }

                self.display.as_mut().unwrap().finish_frame();

                self.window.as_ref().unwrap().request_redraw();

                dbg!(start_render_time.elapsed());
            }
            _ => {}
        }
    }
}

pub fn main() {
    let diffuse_bytes = include_bytes!("code.png");
    let diffuse_image = image::load_from_memory(diffuse_bytes).unwrap();
    let diffuse_rgba = diffuse_image.to_rgba8().into_vec();

    let dimensions = diffuse_image.dimensions();

    // Setup a bunch of state:
    let mut app = SimpleVelloApp {
        start_time: None,
        last_time: None,
        display: None,
        window: None,
        interval: spin_sleep_util::interval(Duration::from_secs(1) / 144),
        textures: [
            Texture::new(10, diffuse_rgba, dimensions),
            Texture::default(),
        ],
    };

    // Create and run a winit event loop
    let event_loop = EventLoop::new().unwrap();
    event_loop
        .run_app(&mut app)
        .expect("Couldn't run event loop");
}
